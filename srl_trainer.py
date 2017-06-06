import argparse
import subprocess
import sys
import tempfile
from random import shuffle

import numpy as np
import tensorflow as tf

from srl_data import PAD_INDEX
from srl_data import load_instances
from srl_data import load_model_files
from srl_reader import chunk
from srl_model import build_graph

FLAGS = None


class DeepSrlTrainer(object):
    def __init__(self, flags):
        super(DeepSrlTrainer, self).__init__()
        self.training_iterator = SrlDataIterator(load_instances(flags.train))
        self.validation_iterator = SrlDataIterator(load_instances(flags.valid))
        self.test_iterator = SrlDataIterator(load_instances(flags.test))
        self.vectors, self.word_vocab, self.label_vocab = load_model_files(flags.vocab)
        self.reverse_word_vocab = {val: key for key, val in self.word_vocab.iteritems()}
        self.reverse_label_vocab = {val: key for key, val in self.label_vocab.iteritems()}
        self.max_epochs = 999
        self.batch_size = 30
        self.script_path = flags.script

    def train(self):
        with tf.Session() as sess:
            graph = build_graph(len(self.word_vocab), 100, 100, self.batch_size, len(self.label_vocab))
            sess.run(tf.global_variables_initializer())

            step, accuracy = 0, 0
            current_epoch = 0
            while current_epoch < self.max_epochs:
                step += 1
                batch = self.training_iterator.next_batch(self.batch_size)
                feed = {graph['x']: batch[1], graph['y']: batch[0], graph['seqlen']: batch[3],
                        graph['predicates']: batch[2]}
                sess.run(graph['ts'], feed_dict=feed)

                if self.training_iterator.epochs > current_epoch:
                    current_epoch += 1
                    step, accuracy = 0, 0

                    # eval test set
                    te_epoch = self.validation_iterator.epochs

                    xs = []
                    gold_ys = []
                    pred_ys = []
                    lengths = []
                    indices = []
                    while self.validation_iterator.epochs == te_epoch:
                        step += 1
                        batch = self.validation_iterator.next_batch(self.batch_size)
                        feed = {graph['x']: batch[1], graph['predicates']: batch[2],
                                graph['y']: batch[0], graph['seqlen']: batch[3]}
                        predictions = sess.run([graph['preds']], feed_dict=feed)[0]
                        xs.extend(batch[1])
                        gold_ys.extend(batch[0])
                        pred_ys.extend(predictions)
                        lengths.extend(batch[3])
                        indices.extend(batch[4])

                    step, accuracy = 0, 0
                    print('Epoch %s results:' % current_epoch)
                    print(self.evaluate(xs, pred_ys, gold_ys, lengths, indices))

    def evaluate(self, xs, pred_ys, gold_ys, lens, indices):
        with tempfile.NamedTemporaryFile() as gold_temp, tempfile.NamedTemporaryFile() as pred_temp:
            self._write_to_file(gold_temp, xs, gold_ys, lens, indices)
            self._write_to_file(pred_temp, xs, pred_ys, lens, indices)
            return subprocess.check_output(["perl", self.script_path, gold_temp.name, pred_temp.name]).decode('utf-8')

    def _write_to_file(self, output_file, xs, ys, lens, indices):
        for sentence, labels, length, index in zip(xs, ys, lens, indices):
            line = ''
            i = 0
            for word, predicted in zip(sentence[:length], chunk([self.reverse_label_vocab[l] for l in labels[:length]],
                                                                conll=True)):
                line += '{} {}\n'.format(i == index and word or "-", predicted)
                i += 1
            output_file.write(line + '\n')
        output_file.flush()
        output_file.seek(0)


class SrlDataIterator(object):
    def __init__(self, data, pad_index=PAD_INDEX, num_buckets=5):
        super(SrlDataIterator, self).__init__()
        self.num_buckets = num_buckets
        self.pad_index = pad_index
        self.epochs = 0

        data.sort(key=lambda x: x['length'])
        self.bucket_size = len(data) / num_buckets
        self.data = []
        for bucket in range(num_buckets):
            self.data.append(data[bucket * self.bucket_size: (bucket + 1) * self.bucket_size - 1])

        self.pointer = np.array([0] * num_buckets)
        self.shuffle()

    def shuffle(self):
        for i in range(self.num_buckets):
            shuffle(self.data[i])
            self.pointer[i] = 0

    def next_batch(self, n):
        # noinspection PyTypeChecker
        if np.any(self.pointer + n + 1 > self.bucket_size):
            self.epochs += 1
            self.shuffle()

        bucket = np.random.randint(0, self.num_buckets)

        batch = self.data[bucket][self.pointer[bucket]:self.pointer[bucket] + n]
        self.pointer[bucket] += n

        lens = [instance['length'] for instance in batch]
        max_length = max(lens)
        return self._pad_vals('labels', batch, max_length), self._pad_vals('words', batch, max_length), self._pad_vals(
            'is_predicate', batch, max_length), lens, [instance['index'] for instance in batch]

    def _pad_vals(self, key, batch, maxlen):
        padded = np.empty([len(batch), maxlen], dtype=np.int32)
        padded.fill(self.pad_index)
        for i, sentence in enumerate(padded):
            sentence[:batch[i]['length']] = batch[i][key]
        return padded


def main(_):
    srl_trainer = DeepSrlTrainer(FLAGS)
    srl_trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='models', help='Directory to save models/checkpoints.')
    parser.add_argument('--train', required=True, type=str, help='Binary (*.pkl) train file path.')
    parser.add_argument('--valid', required=True, type=str, help='Binary (*.pkl) validation file path.')
    parser.add_argument('--test', required=True, type=str, help='Binary (*.pkl) test file path.')
    parser.add_argument('--vocab', required=True, type=str, help='Path to directory containing vocabulary files.')
    parser.add_argument('--script', required=True, type=str, help='Path to evaluation script.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
