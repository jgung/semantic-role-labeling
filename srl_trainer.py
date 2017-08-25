import argparse
import logging
import random
import subprocess
import sys
import tempfile
import time
from random import shuffle

import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from tqdm import tqdm

from features import LABEL_KEY, LENGTH_KEY, PAD_INDEX
from features import get_features_from_config
from srl_feature_extractor import MARKER_KEY
from srl_feature_extractor import SrlFeatureExtractor
from srl_reader import chunk
from srl_utils import configure_logger
from srl_utils import deserialize
from srl_utils import read_json
from tagger import DBLSTMTagger
from tagger import KEEP_PROB_KEY

FLAGS = None


class DeepSrlTrainer(object):
    def __init__(self, flags):
        super(DeepSrlTrainer, self).__init__()
        self.save_path = flags.save
        self.load_path = flags.load

        self.script_path = flags.script

        self._read_conf(flags.config)

        extractor = SrlFeatureExtractor(feats=self.features)
        extractor.load(flags.vocab)
        self.marker_index = extractor.extractors[MARKER_KEY].indices['1']

        self.label_vocab = extractor.extractors[LABEL_KEY].indices

        self.reverse_label_vocab = [None] * len(self.label_vocab)
        for key, val in self.label_vocab.iteritems():
            self.reverse_label_vocab[val] = key

        self.transition_params = self._create_transition_matrix(self.reverse_label_vocab)

        if flags.train:
            self.training_iterator = SrlDataIterator(deserialize(flags.train), self.batch_size, features=self.features,
                                                     num_buckets=self.num_buckets, max_length=self.max_length)
        if flags.valid:
            self.validation_iterator = SrlDataIterator(deserialize(flags.valid), self.batch_size, features=self.features)
        if flags.test:
            self.test_iterator = SrlDataIterator(deserialize(flags.test), self.batch_size, features=self.features)

    def _read_conf(self, conf_json):
        conf = read_json(conf_json)
        logging.info(conf)
        self.max_epochs = conf['max_epochs']
        self.batch_size = conf['batch_size']
        self.keep_prob = conf['keep_prob']
        self.lstm_hidden_dim = conf['lstm_hidden_dim']
        self.lstm_num_layers = conf['lstm_num_layers']
        self.max_length = conf['max_length']
        self.num_buckets = conf['num_buckets']
        self.features = get_features_from_config(conf_json)

    def _load_graph(self):
        return DBLSTMTagger(features=self.features, num_classes=len(self.label_vocab), num_layers=self.lstm_num_layers,
                            state_dim=self.lstm_hidden_dim)

    def train(self):
        with tf.Session() as sess:
            graph = self._load_graph()
            graph.train()
            if self.load_path:
                graph.saver.restore(sess, self.load_path)
            else:
                sess.run(tf.global_variables_initializer())
                graph.initialize_embeddings(sess)

            current_epoch, step, max_score = 0, 0, float('-inf')
            patience = 0
            while current_epoch < self.max_epochs:
                logging.info('Epoch %d', current_epoch)
                then = time.time()
                with tqdm(total=self.training_iterator.size, leave=False, unit=' instances') as bar:
                    for batch in self.training_iterator.epoch():
                        feed = {graph.feed_dict[k]: batch[k] for k in batch.keys()}
                        feed[graph.feed_dict[KEEP_PROB_KEY]] = self.keep_prob
                        sess.run(graph.train_step, feed_dict=feed)
                        step += 1
                        bar.update(len(batch[LABEL_KEY]))
                logging.info('Training for epoch %d completed in %f seconds.', current_epoch, time.time() - then)

                score = self._test(graph=graph, sess=sess, iterator=self.validation_iterator)
                if score >= max_score:
                    max_score = score
                    patience = 0
                    if self.save_path:
                        save_path = graph.saver.save(sess, self.save_path)
                        logging.info("Model to file: %s" % save_path)
                else:
                    patience += 1

                logging.info('Epoch %d F1: %f (best: %f, %d epoch(s) ago)', current_epoch, score, max_score, patience)
                current_epoch += 1

    def test(self):
        with tf.Session() as sess:
            graph = self._load_graph()
            graph.train()
            graph.saver.restore(sess, self.load_path)
            self._test(graph, sess, self.test_iterator)

    def _test(self, graph, sess, iterator):
        then = time.time()
        pred_ys, gold_ys, words, indices = [], [], [], []
        with tqdm(total=iterator.size, leave=False, unit=' instances') as bar:
            for batch in iterator.epoch():
                feed = {graph.feed_dict[k]: batch[k] for k in batch.keys()}
                feed[graph.feed_dict[KEEP_PROB_KEY]] = 1.0
                logits = sess.run(graph.scores, feed_dict=feed)

                gold_ys.extend([gold[:stop] for (gold, stop) in zip(batch[LABEL_KEY], batch[LENGTH_KEY])])
                pred_ys.extend(
                    [viterbi_decode(score=pred[:stop], transition_params=self.transition_params)[0] for
                     (pred, stop) in zip(logits, batch[LENGTH_KEY])])
                words.extend(batch['words'])
                indices.extend(batch[MARKER_KEY])
                bar.update(len(batch[LABEL_KEY]))
        logging.info('Evaluation completed in %d seconds.', time.time() - then)
        return self.evaluate(words, pred_ys, gold_ys, indices)

    def evaluate(self, words, pred_ys, gold_ys, indices):
        with tempfile.NamedTemporaryFile() as gold_temp, tempfile.NamedTemporaryFile() as pred_temp:
            self._write_to_file(gold_temp, words, gold_ys, indices)
            self._write_to_file(pred_temp, words, pred_ys, indices)
            result = subprocess.check_output(['perl', self.script_path, gold_temp.name, pred_temp.name]).decode('utf-8')
            logging.info('\n%s', result)
            return float(result.strip().split('\n')[6].strip().split()[6])

    def _write_to_file(self, output_file, xs, ys, indices):
        for words, labels, markers in zip(xs, ys, indices):
            line = ''
            for word, predicted, marker in zip(
                    words, chunk([self.reverse_label_vocab[l] for l in labels], conll=True), markers):
                line += '{} {}\n'.format(marker == self.marker_index and word or '-', predicted)
            output_file.write(line + '\n')
        output_file.flush()
        output_file.seek(0)

    @staticmethod
    def _create_transition_matrix(labels):
        num_tags = len(labels)
        transition_params = np.zeros([num_tags, num_tags], dtype=np.float32)
        for i, prev_label in enumerate(labels):
            for j, label in enumerate(labels):
                if i != j and label[0] == 'I' and not prev_label == 'B' + label[1:]:
                    transition_params[i, j] = np.NINF
        return transition_params


class SrlDataIterator(object):
    def __init__(self, data, batch_size, features, num_buckets=5, max_length=99999):
        super(SrlDataIterator, self).__init__()
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.size = len(data)
        self.features = features

        data = [x for x in data if x[LENGTH_KEY] <= max_length]
        data.sort(key=lambda inst: inst[LENGTH_KEY])
        self.bucket_size = self.size / num_buckets
        self.data = []
        for bucket in range(num_buckets):
            self.data.append(data[bucket * self.bucket_size: (bucket + 1) * self.bucket_size])
        self.data[-1].extend(data[self.bucket_size * num_buckets:])  # add remaining instances
        self.pointer = np.array([0] * num_buckets)

    def max_steps(self):
        return len(self.data) / self.batch_size

    def epoch(self):
        self._reset()
        while not self._has_next():
            # select a random bucket (from remaining buckets)
            bucket = random.choice([i for (i, p) in enumerate(self.pointer.tolist()) if p + 1 < self.bucket_size])

            batch = self.data[bucket][self.pointer[bucket]:self.pointer[bucket] + self.batch_size]
            self.pointer[bucket] += len(batch)
            yield self._prepare_batch(batch)

    def _has_next(self):
        # noinspection PyTypeChecker
        return np.all(self.pointer >= self.bucket_size)

    def _reset(self):
        for i in range(self.num_buckets):
            shuffle(self.data[i])
            self.pointer[i] = 0

    def _prepare_batch(self, batch):
        lengths = [instance[LENGTH_KEY] for instance in batch]
        max_length = max(lengths)
        labels = self._pad_vals(LABEL_KEY, batch, max_length)
        feed_dict = {LABEL_KEY: labels, LENGTH_KEY: lengths}
        for feature in self.features:
            if not feature.function:
                feed_dict[feature.name] = self._pad_vals(feature.name, batch, max_length)
            else:
                feed_dict[feature.name] = self._pad_list_feature(feature.name, batch, max_length, feature.function.max_length)
        return feed_dict

    @staticmethod
    def _pad_vals(key, batch, maxlen):
        padded = np.empty([len(batch), maxlen], dtype=np.int32)
        padded.fill(PAD_INDEX)
        for i, sentence in enumerate(padded):
            sentence[:batch[i][LENGTH_KEY]] = batch[i][key]
        return padded

    @staticmethod
    def _pad_list_feature(key, batch, maxlen, max_feat_length):
        padded = np.empty([len(batch), maxlen, max_feat_length], dtype=np.int32)
        padded.fill(PAD_INDEX)
        for i, sentence in enumerate(padded):
            features = batch[i][key]
            for index, word in enumerate(features):
                sentence[index, :word.size] = word[:max_feat_length]
        return padded


def main(_):
    configure_logger(FLAGS.log)
    srl_trainer = DeepSrlTrainer(FLAGS)
    if FLAGS.train:
        if not FLAGS.valid:
            raise ValueError('Missing required validation (dev) set. Use "--valid path/to/valid.pkl" to specify validation data.')
        srl_trainer.train()
    if FLAGS.test:
        srl_trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='data/models/model', help='Path to save models/checkpoints.')
    parser.add_argument('--load', type=str, help='Path to load previously saved model.')
    parser.add_argument('--train', type=str, help='Binary (*.pkl) train file path.')
    parser.add_argument('--valid', type=str, help='Binary (*.pkl) validation file path.')
    parser.add_argument('--test', required=False, type=str, help='Binary (*.pkl) test file path.')
    parser.add_argument('--vocab', required=True, type=str, help='Path to directory containing vocabulary files.')
    parser.add_argument('--script', required=True, type=str, help='Path to evaluation script.')
    parser.add_argument('--log', default='srl_trainer.log', type=str, help='Path to output log.')
    parser.add_argument('--config', required=True, type=str, help='Path to json configuration file.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
