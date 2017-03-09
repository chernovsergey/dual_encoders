import array
import numpy as np
import tensorflow as tf
from collections import defaultdict
import codecs

from tensorflow.contrib.metrics import streaming_sparse_recall_at_k as recall_at_k


def loadVOCAB(filename):
    vocab = None

    with open(filename) as f:
        vocab = f.read().splitlines()

    dct = defaultdict(int)
    for idx, word in enumerate(vocab):
        dct[word] = idx

    return [vocab, dct]


def loadGLOVE(filename, vocab):
    """
    Load glove vectors from a .txt file.
    Optionally limit the vocabulary to save memory. `vocab` should be a set.
    """
    dct = {}
    vectors = array.array('d')
    current_idx = 0
    with codecs.open(filename, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            tokens = line.split(" ")
            word = tokens[0]
            entries = tokens[1:]
            if not vocab or word in vocab:
                dct[word] = current_idx
                vectors.extend(float(x) for x in entries)
                current_idx += 1
        word_dim = len(entries)
        num_vectors = len(dct)
        tf.logging.info("Found {} out of {} vectors in Glove".format(num_vectors, len(vocab)))
        return [np.array(vectors).reshape(num_vectors, word_dim), dct]


def buildEMBMatrix(vocab_dict, glove_dict, glove_vectors, embedding_dim):
    initial_embeddings = np.random.uniform(-0.25, 0.25, (len(vocab_dict), embedding_dim)).astype("float32")
    for word, glove_word_idx in glove_dict.items():
        word_idx = vocab_dict.get(word)
        initial_embeddings[word_idx, :] = glove_vectors[glove_word_idx]
    return initial_embeddings


FLAGS = tf.flags.FLAGS


def get_embeddings(hparams):
    if hparams.glove_path and hparams.vocab_path:
        tf.logging.info("Loading Glove embeddings...")
        vocab_array, vocab_dict = loadVOCAB(hparams.vocab_path)
        glove_vectors, glove_dict = loadGLOVE(hparams.glove_path, vocab=set(vocab_array))
        initializer = buildEMBMatrix(vocab_dict, glove_dict, glove_vectors, hparams.embedding_dim)
    else:
        tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
        initializer = tf.random_uniform_initializer(-0.25, 0.25)

    return tf.get_variable("word_embeddings", shape=[hparams.vocab_size, hparams.embedding_dim],
                           initializer=initializer)


class DualEncoders:
    def __init__(self, hparams):
        self.hparams = hparams

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.learning_rate = tf.train.exponential_decay(
            self.hparams.learning_rate,  # Base learning rate.
            self.global_step,  # Current index into the dataset.
            self.hparams.decay_step,  # Decay step.
            self.hparams.decay_rate,  # Decay rate.
            staircase=self.hparams.staircase, name="learning_rate_decay")

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.context = tf.placeholder(tf.int64, [None, hparams.max_context_len], name="Context")
        self.context_len = tf.placeholder(tf.int64, [None], name="ContextLenValue")
        self.utterance = tf.placeholder(tf.int64, [None, hparams.max_context_len], name="Utterance")
        self.utterance_len = tf.placeholder(tf.int64, [None], name="UtteranceLenValue")
        self.targets = tf.placeholder(tf.int64, [None], name="TargetLabels")
        self.val_targets = tf.placeholder(tf.int64, [None], name="ValidationLabels")

        logits = self.inference()
        probs = tf.sigmoid(logits, name="probs_op")

        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(self.targets), name="CrossEntropy")
        mean_loss = tf.reduce_mean(losses, name="Mean_CE_Loss")

        train_op = tf.contrib.layers.optimize_loss(loss=mean_loss,
                                                   global_step=self.global_step,
                                                   learning_rate=self.learning_rate,
                                                   clip_gradients=self.hparams.max_grad_norm,
                                                   optimizer=hparams.optimizer)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        mean_loss_ema_op = ema.apply([mean_loss])
        with tf.control_dependencies([self.targets]):  # update only when train targets passed
            train_op_group = tf.group(train_op, mean_loss_ema_op)

        self.probs_op = probs
        self.train_loss_op = ema.average(mean_loss)
        self.train_op = train_op_group

        self.train_summaries = tf.summary.merge([tf.summary.scalar("loss", mean_loss),
                                                 tf.summary.scalar("learning_rate", self.learning_rate)])

        self.val_probs, self.val_summary = self.validation_accuracy(probs, self.val_targets, mean_loss)

    def inference(self):
        W_emb = get_embeddings(self.hparams)

        context_emb = tf.nn.embedding_lookup(W_emb, self.context, name="ContextEmbedding")
        utterance_emb = tf.nn.embedding_lookup(W_emb, self.utterance, name="UtteranceEmbedding")

        with tf.variable_scope("BidirectionalLSTM"):
            argsdict = {"forget_bias": 2.0, "use_peepholes": True, "state_is_tuple": True}
            fw_cell = tf.contrib.rnn.LSTMCell(self.hparams.rnn_dim, **argsdict)
            bw_cell = tf.contrib.rnn.LSTMCell(self.hparams.rnn_dim, **argsdict)
            seq = tf.concat([context_emb, utterance_emb],axis=0)
            seqlen = tf.concat([self.context_len,self.utterance_len], axis=0)
            _, rnn_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                            inputs=seq,
                                                            sequence_length=seqlen,
                                                            dtype=tf.float32)

            fw_encoding_context, fw_encoding_utter = tf.split(rnn_states[0].h, 2, axis=0)
            bw_encoding_context, bw_encoding_utter = tf.split(rnn_states[1].h, 2, axis=0)

            encoding_context = tf.concat([fw_encoding_context, bw_encoding_context], axis=1)
            encoding_utterance = tf.concat([fw_encoding_utter, bw_encoding_utter], axis=1)

        with tf.variable_scope("Prediction"):
            M = tf.get_variable(name="M", shape=[2 * self.hparams.rnn_dim, 2 * self.hparams.rnn_dim],
                                initializer=tf.random_uniform_initializer(-0.25, 0.25))

        generated_response = tf.matmul(encoding_context, M)
        generated_response = tf.expand_dims(generated_response, 2)
        encoding_utterance = tf.expand_dims(encoding_utterance, 2)
        logits = tf.matmul(generated_response, encoding_utterance, True)
        logits = tf.reshape(logits, [-1])
        return logits

    def validation_accuracy(self, pred_labels, val_labels, val_loss):
        shaped_probs = tf.reshape(pred_labels, [-1, 10])

        def get_top(k):
            return tf.reduce_mean(tf.cast(tf.nn.in_top_k(shaped_probs, val_labels, k=k), tf.float32))

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        top1, top2, top3, top5 = [get_top(k) for k in [1, 2, 3, 5]]
        maintain_averages = ema.apply([top1, top2, top3, top5, val_loss])

        with tf.control_dependencies([self.val_targets]):  # update only when validation targets passed
            self.update_averages = tf.group(maintain_averages)

        # TODO reset shadow variables between validation sessions
        self.val_loss = ema.average(val_loss)
        self.top1_av = ema.average(top1)
        self.top2_av = ema.average(top2)
        self.top3_av = ema.average(top3)
        self.top5_av = ema.average(top5)

        val_summary = tf.summary.merge([tf.summary.scalar("validation_loss", self.val_loss),
                                        tf.summary.scalar("top1", self.top1_av),
                                        tf.summary.scalar("top2", self.top2_av),
                                        tf.summary.scalar("top3", self.top3_av),
                                        tf.summary.scalar("top5", self.top5_av),
                                        tf.summary.histogram("correct_probs_distribution", shaped_probs[:, 0]),
                                        tf.summary.histogram("incorrect_probs_distribution", shaped_probs[:, 1:])])
        return shaped_probs, val_summary

    def setSession(self, session):
        self._sess = session

    def save_model(self, saver, location, step):
        saver.save(self._sess, location, global_step=step)

    def load_model(self, saver, location):
        print("Variable initializaion")
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self._sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(location)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring model')
            saver.restore(self._sess, ckpt.model_checkpoint_path)

    def batch_fit(self, batch_dict):
        feed_dict = {self.context: batch_dict["context"],
                     self.context_len: batch_dict["context_len"],
                     self.utterance: batch_dict["utterance"],
                     self.utterance_len: batch_dict["utterance_len"],
                     self.targets: batch_dict["label"]}

        train_summary, step, _, loss = self._sess.run([self.train_summaries, self.global_step,
                                                       self.train_op, self.train_loss_op], feed_dict=feed_dict)
        return loss, step, train_summary

    def predict(self, batch_dict):
        feed_dict = {self.context: batch_dict["context"],
                     self.context_len: batch_dict["context_len"],
                     self.utterance: batch_dict["utterance"],
                     self.utterance_len: batch_dict["utterance_len"]}

        return self._sess.run([self.probs_op], feed_dict=feed_dict)

    def validate(self, batch_dict):
        val_targets = np.zeros([len(batch_dict["label"]) / 10], dtype=np.int64)
        feed_dict = {self.context: batch_dict["context"],
                     self.context_len: batch_dict["context_len"],
                     self.utterance: batch_dict["utterance"],
                     self.utterance_len: batch_dict["utterance_len"],
                     self.targets: batch_dict["label"],
                     self.val_targets: val_targets}

        _, val_loss, t1, t2, t3, t5, val_probs, val_summary = self._sess.run([self.update_averages, self.val_loss,
                                                                              self.top1_av, self.top2_av,
                                                                              self.top3_av, self.top5_av,
                                                                              self.val_probs,
                                                                              self.val_summary],
                                                                             feed_dict=feed_dict)

        return [t1, t2, t3, t5], val_loss, val_summary
