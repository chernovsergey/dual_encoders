import os
import unicodecsv as csv
import itertools
import functools
import tensorflow as tf
import numpy as np
import array
import cPickle
import re
import codecs

tf.flags.DEFINE_integer("min_word_frequency", 10, "Minimum frequency of words in the vocabulary")
tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")
tf.flags.DEFINE_string("input_dir", os.path.abspath("../data"), "Directory containing original CSV data files")
tf.flags.DEFINE_string("output_dir", os.path.abspath("../data"), "Output directory for .npy files")

FLAGS = tf.flags.FLAGS

# Data to be converted
TRAIN_PATH = os.path.join(FLAGS.input_dir, "train.csv")
VALIDATION_PATH = os.path.join(FLAGS.input_dir, "valid_500.csv")
TEST_PATH = os.path.join(FLAGS.input_dir, "test.csv")

# Specify it vocabulary has been created earlier
VOCAB_PATH = os.path.join(FLAGS.output_dir, "vocab_processor.bin")


class CustomVocabularyProcessor(tf.contrib.learn.preprocessing.VocabularyProcessor):
    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None,
                 tokenizer_fn=None):

        super(CustomVocabularyProcessor, self).__init__(max_document_length, min_frequency, vocabulary, tokenizer_fn)
        self.max_document_length = max_document_length
        self.min_frequency = min_frequency
        self.vocabulary_ = tf.contrib.learn.preprocessing.CategoricalVocabulary()

    def tokenizer(iterator):
        TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                                  re.UNICODE)
        for value in iterator:
            yield TOKENIZER_RE.findall(value)

    def fit(self, raw_documents, unused_y=None):
        for tokens in self._tokenizer(raw_documents):
            for token in tokens:
                self.vocabulary_.add(token)
        if self.min_frequency > 0:
            self.vocabulary_.trim(self.min_frequency)
        self.vocabulary_.freeze()
        return self

    def find_first(self, item, vec):
        """return the index of the first occurence of item in vec"""
        for i in xrange(len(vec)):
            if item == vec[i]:
                return i
        return -1

    def transform(self, raw_documents):
        for tokens in self._tokenizer(raw_documents):

            word_ids = []
            for idx, token in enumerate(tokens):
                word_ids.append(self.vocabulary_.get(token))

            if len(tokens) >= FLAGS.max_sentence_len:
                # Get last FLAGS.max_sentence_len tokens
                word_ids = np.array(word_ids[-FLAGS.max_sentence_len:], np.int64)
                assert (len(word_ids) == FLAGS.max_sentence_len)
            else:
                word_ids160 = np.zeros(FLAGS.max_sentence_len, np.int64)
                for idx, val in enumerate(word_ids):
                    word_ids160[idx] = val
                word_ids = word_ids160
                assert (len(word_ids) == FLAGS.max_sentence_len)

            yield word_ids


def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)


def create_csv_iter(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        # Skip the header
        next(reader)
        for row in reader:
            yield row


def create_vocabs(input_iter, min_frequency):
    vocab_processor_my = CustomVocabularyProcessor(FLAGS.max_sentence_len,
                                                   min_frequency=min_frequency,
                                                   tokenizer_fn=tokenizer_fn)
    vocab_processor_my.fit(input_iter)

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_len,
                                                                         min_frequency=min_frequency,
                                                                         tokenizer_fn=tokenizer_fn,
                                                                         vocabulary=vocab_processor_my.vocabulary_)
    vocab_processor.fit(input_iter)

    return vocab_processor, vocab_processor_my


def transform_sentence(sequence, vocab_processor):
    return next(vocab_processor.transform([sequence])).tolist()


def create_example_train(row, vocab, custom_vocab):
    context, utterance, label = row

    # Get last X tokens here
    context_transformed = transform_sentence(context, custom_vocab)
    utterance_transformed = transform_sentence(utterance, vocab)
    context_len = min(len(next(custom_vocab._tokenizer([context]))), FLAGS.max_sentence_len)
    utterance_len = len(next(vocab._tokenizer([utterance])))

    assert (context_len <= FLAGS.max_sentence_len)

    label = int(float(label))

    return [context_transformed, utterance_transformed, context_len, utterance_len, label]


def create_example_test(row, vocab, custom_vocab):
    """
    :param custom_vocab: Vocabulary processor that gets last X tokents
    """

    context, utterance = row[:2]
    distractors = row[2:]
    context_len = min(len(next(custom_vocab._tokenizer([context]))), FLAGS.max_sentence_len)
    utterance_len = len(next(vocab._tokenizer([utterance])))
    context_transformed = transform_sentence(context, custom_vocab)
    utterance_transformed = transform_sentence(utterance, vocab)

    assert (context_len <= FLAGS.max_sentence_len)

    result = [context_transformed, utterance_transformed, context_len, utterance_len]
    for i, distractor in enumerate(distractors):
        dis_len = len(next(vocab._tokenizer([distractor])))
        dis_transformed = transform_sentence(distractor, vocab)
        result.append(dis_transformed)
        result.append(dis_len)

    return result


def convert_file(input_filename, output_filename, example_fn):
    storage = []
    for i, row in enumerate(create_csv_iter(input_filename)):
        x = example_fn(row)
        storage.append(x)
    np.save(output_filename, storage)
    print("Converted to ", output_filename)


def write_vocabulary(vocab_processor, outfile):
    vocab_size = len(vocab_processor.vocabulary_)
    with codecs.open(outfile, "w", encoding='utf-8') as vocabfile:
        for id in range(vocab_size):
            word = vocab_processor.vocabulary_._reverse_mapping[id]
            vocabfile.write(word + "\n")
    print("Saved vocabulary to {}".format(outfile))


def loadVocab():
    if os.path.exists(VOCAB_PATH):
        vocab = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(VOCAB_PATH)
        print("Loaded vocabulary from pickle. Num words: ", len(vocab.vocabulary_))

        custom_vocab = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(VOCAB_PATH)
        print("Loaded CUSTOM vocabulary from pickle. Num words: ", len(custom_vocab.vocabulary_))

        return vocab, custom_vocab
    else:
        print("Creating vocabularies ...")
        input_iter = create_csv_iter(TRAIN_PATH)
        input_iter = (x[0] + " " + x[1] for x in input_iter)

        vocab, custom_vocab = create_vocabs(input_iter, min_frequency=FLAGS.min_word_frequency)
        print("Total vocabulary sizes: {0}".format(len(vocab.vocabulary_)))

        write_vocabulary(vocab, os.path.join(FLAGS.output_dir, "vocabulary.txt"))
        vocab.save(os.path.join(FLAGS.output_dir, "vocab_processor.bin"))

        return vocab, custom_vocab


if __name__ == "__main__":
    vocab, custom_vocab = loadVocab()

    convert_file(input_filename=VALIDATION_PATH,
                 output_filename=os.path.join(FLAGS.output_dir, "validation.ids.500"),
                 example_fn=functools.partial(create_example_test, vocab=vocab, custom_vocab=custom_vocab))
    #
    # convert_file(input_filename=TEST_PATH,
    #              output_filename=os.path.join(FLAGS.output_dir, "test.ids"),
    #              example_fn=functools.partial(create_example_test, vocab=vocab, custom_vocab=custom_vocab))

    # convert_file(input_filename=TRAIN_PATH,
    #              output_filename=os.path.join(FLAGS.output_dir, "train.ids"),
    #              example_fn=functools.partial(create_example_train, vocab=vocab, custom_vocab=custom_vocab))
