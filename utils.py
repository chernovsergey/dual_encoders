from collections import namedtuple

import numpy as np
import tensorflow as tf
from itertools import islice, chain

def getHyperParams(FLAGS):
    HParams = namedtuple(
        "HParams",
        [
            "batch_size", "embedding_dim", "eval_batch_size", "learning_rate",
            "max_context_len", "max_utterance_len", "optimizer", "rnn_dim", "vocab_size",
            "glove_path", "dropout", "vocab_path", "num_epochs", "max_grad_norm",
            "decay_step", "decay_rate", "staircase", "rnn_type", "extra_negatives_count"
        ])
    return HParams(batch_size=FLAGS.batch_size,
                   embedding_dim=FLAGS.embedding_dim,
                   eval_batch_size=FLAGS.eval_batch_size,
                   learning_rate=FLAGS.learning_rate,
                   max_context_len=FLAGS.max_context_len,
                   max_utterance_len=FLAGS.max_utterance_len,
                   optimizer=FLAGS.optimizer,
                   rnn_dim=FLAGS.rnn_dim,
                   vocab_size=FLAGS.vocab_size,
                   glove_path=FLAGS.glove_path,
                   dropout=FLAGS.dropout,
                   vocab_path=FLAGS.vocab_path,
                   num_epochs=FLAGS.num_epochs,
                   max_grad_norm=FLAGS.max_grad_norm,
                   decay_step=FLAGS.decay_step,
                   decay_rate=FLAGS.decay_rate,
                   staircase=FLAGS.staircase,
                   rnn_type=FLAGS.rnn_type,
                   extra_negatives_count=FLAGS.extra_negatives_count)


class BatchGenerator:
    def __init__(self, data, hparams, isTest=False):
        """

        :param data: string file or numpy array
        """
        if isinstance(data, basestring):
            self.allData = np.load(data)
        else:
            self.allData = data

        self.hparams = hparams
        self.isTest = isTest
        if isTest:
            self.batch_size = self.hparams.eval_batch_size
        else:
            self.batch_size = self.hparams.batch_size

        self.sourceiter = iter(self.allData)

    def batch(self, size):
        while True:
            batchiter = islice(self.sourceiter, size)
            yield chain([batchiter.next()], batchiter)

    def _get_train_batch(self):
        batch_gen = self.batch(self.batch_size)

        if batch_gen is not None:
            for next_batch in batch_gen:
                next_batch = list(next_batch)
                np.random.shuffle(next_batch)

                # Indexes:
                # 0 - context
                # 1 - utterance
                # 2, 3 context and utterance lengths respectively
                # 4 - label

                contexts = np.array([np.array(x[0]) for x in next_batch])
                utterances = np.array([np.array(x[1]) for x in next_batch])
                context_lens = np.array([min(x[2], self.hparams.max_context_len) for x in next_batch])
                utterance_lens = np.array([min(x[3], self.hparams.max_utterance_len) for x in next_batch])
                labels = np.array([x[4] for x in next_batch])

                if self.hparams.extra_negatives_count > 0:
                    positives = np.where(labels == 1)
                    num_positives = len(positives[0])
                    pos_contexts = contexts[positives]
                    pos_contexts_lens = context_lens[positives]
                    for _ in range(self.hparams.extra_negatives_count):
                        pass
                        ids = np.random.randint(0, len(self.allData) - 1, num_positives)

                        fake_utterances = self.allData[ids][:, 1]
                        fake_utterances_lens = list(
                            map(lambda x: min(x, self.hparams.max_utterance_len), self.allData[ids][:, 3]))

                        fake_labels = [0] * num_positives

                        contexts = np.array(list(chain(contexts, pos_contexts)))
                        context_lens = np.array(list(chain(context_lens, pos_contexts_lens)))
                        utterances = np.array(list(chain(utterances, fake_utterances)))
                        utterance_lens = np.array(list(chain(utterance_lens, fake_utterances_lens)))
                        labels = np.array(list(chain(labels, fake_labels)))

                    # Damn ungly! Use sklearn.utils.shuffle if possible
                    permut = np.random.permutation(len(contexts))
                    contexts = contexts[permut]
                    utterances = utterances[permut]
                    context_lens = context_lens[permut]
                    utterance_lens = utterance_lens[permut]
                    labels = labels[permut]

                return {"context": contexts,
                        "context_len": context_lens,
                        "utterance": utterances,
                        "utterance_len": utterance_lens,
                        "label": labels}
        else:
            return None

    def _get_validation_batch(self):

        batch_gen = self.batch(self.batch_size)

        if batch_gen is not None:
            for next_batch in batch_gen:

                next_batch = list(next_batch)

                contexts = []
                utterances = []
                context_lens = []
                utterance_lens = []
                labels = []

                for row in next_batch:
                    assert (len(row) == 22)

                    context, utterance = row[0], row[1]
                    context_len, utterance_len = row[2], row[3]
                    distractors = row[[x for x in range(4, 22) if x % 2 == 0]]
                    distractor_lens = row[[x for x in range(4, 22) if x % 2 != 0]]

                    contexts.extend([context] * 10)
                    context_lens.extend([context_len] * 10)

                    labels.append(1)

                    utterances.append(utterance)
                    utterance_lens.append(utterance_len)

                    utterances.extend(distractors)
                    utterance_lens.extend(distractor_lens)

                    labels.extend([0] * len(distractors))

                contexts = np.array([np.array(x) for x in contexts])
                utterances = np.array([np.array(x) for x in utterances])
                context_lens = np.array(list(map(lambda x: min(x, self.hparams.max_context_len), context_lens)))
                utterance_lens = np.array(list(map(lambda x: min(x, self.hparams.max_utterance_len), utterance_lens)))
                labels = np.array([x for x in labels])

                return {"context": contexts,
                        "context_len": context_lens,
                        "utterance": utterances,
                        "utterance_len": utterance_lens,
                        "label": labels,
                        "size": len(next_batch)}
        else:
            return None

    def get_batch(self):
        if self.isTest:
            return self._get_validation_batch()
        else:
            return self._get_train_batch()


def recall_summary(recalls, loss):
    tf.logging.info("Loss:{0:.5f}  Recall@1 = {1:.5f}  Recall@2 = {2:.5f}  "
                    "Recall@3 = {3:.5f}  Recall@5 = {4:.5f}".format(loss, recalls[0], recalls[1],
                                                                    recalls[2], recalls[3]))
