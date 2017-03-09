import os
import time

from dual_encoder import *
from utils import BatchGenerator, getHyperParams

tf.flags.DEFINE_string("input_dir", "../data", "Directory with input data. ")
tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 5, "Evaluate after this many train steps")
tf.flags.DEFINE_integer("cpkt_every", 4999, "Evaluate after this many train steps")
tf.flags.DEFINE_integer("save_summary_every", 5, "Evaluate after this many train steps")
tf.flags.DEFINE_integer("max_to_keep", 5, "Keep saved last X models")

# Model Params
tf.flags.DEFINE_integer("vocab_size", 28081, "The size of the vocabulary.")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("rnn_dim", 256, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_context_len", 160, "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 160, "Truncate utterance to this length")

# Exponential learning rate decay params
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("decay_step", 20000, "Dropout keep probability")
tf.flags.DEFINE_float("decay_rate", 0.5, "Dropout keep probability")
tf.flags.DEFINE_bool("staircase", True, "Staircase decay switcher")

# Regularization
tf.flags.DEFINE_float("dropout", 1.0, "Dropout keep probability")
tf.flags.DEFINE_float("max_grad_norm", 10.0, "Gradients will be clipped by this value")
tf.flags.DEFINE_integer("rnn_type", 0, "Types: 1 - GRU with layer normalization, 2 - Phased LSTM, Otherwise - LSTM")
tf.flags.DEFINE_integer("extra_negatives_count", 2, "Number of extra negative examples for each context in batch")

# Pre-trained embeddings
tf.flags.DEFINE_string("glove_path", None, "Path to pre-trained Glove vectors")  # '../data/glove.840B.300d.txt'
tf.flags.DEFINE_string("vocab_path", None, "Path to vocabulary.txt file")  # '../data/vocabulary2M.txt'

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 10, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 10, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")

FLAGS = tf.flags.FLAGS

if FLAGS.model_dir:
    MODEL_DIR = FLAGS.model_dir
else:
    MODEL_DIR = os.path.abspath(os.path.join("../runs", str(int(time.time()))))

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_DATA = FLAGS.input_dir + "/train.ids.npy"
VALIDATION_DATA = FLAGS.input_dir + "/validation.ids.10.npy"

recalls_best = [0.0, 0.0, 0.0, 0.0]  # top 1,2,3,5


def maybe_save(recalls):
    for idx, recall in enumerate(recalls):
        if recall > recalls_best[idx]:
            return True
    return False


def evaluate(step, sess, model, summary_writer):
    batch_gen = BatchGenerator(VALIDATION_DATA, FLAGS, isTest=True)
    eval_step = 0
    summary = None
    while True:
        batch = batch_gen.get_batch()
        if batch is None:
            break

        recalls, loss, summary = model.validate(batch)
        if eval_step != 0 and eval_step % FLAGS.save_summary_every == 0:
            print("Loss:{0:.5f}  Recall@1 = {1:.5f}  Recall@2 = {2:.5f}  "
                  "Recall@3 = {3:.5f}  Recall@5 = {4:.5f}".format(loss, recalls[0], recalls[1],
                                                                  recalls[2], recalls[3]))

        eval_step += 1

    summary_writer.add_summary(summary, step)


def train_loop(sess, model, summary_writer, model_saver):
    batch_gen = BatchGenerator(TRAIN_DATA, FLAGS)
    epoch = 0
    while True:
        batch = batch_gen.get_batch()
        if batch is None:
            tf.logging.info("Epoch {0} is over".format(epoch))
            epoch += 1
            del batch_gen
            batch_gen = BatchGenerator(TRAIN_DATA, FLAGS)  # create batch generator again and proceed
            continue

        loss, step, summary = model.batch_fit(batch)

        if step % FLAGS.save_summary_every == 0:
            summary_writer.add_summary(summary, step)
            tf.logging.info("Step: {0} Loss: {1}".format(step, loss))

        if step != 0 and step % FLAGS.eval_every == 0:
            save_signal = evaluate(step, sess, model, summary_writer)
            if save_signal:
                model.save_model(model_saver, MODEL_DIR, step)


def main(argv):
    with tf.Session() as sess:
        hparams = getHyperParams(FLAGS)
        model = DualEncoders(hparams)
        model.setSession(sess)
        modelSaver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        model.load_model(modelSaver, "./runs/")
        summaryWriter = tf.summary.FileWriter(MODEL_DIR, sess.graph)

        train_loop(sess, model, summaryWriter, modelSaver)


if __name__ == "__main__":
    tf.app.run()
