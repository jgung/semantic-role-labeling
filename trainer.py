import logging
import random
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from features import LABEL_KEY, LENGTH_KEY, END_INDEX, START_INDEX
from features import PAD_INDEX
from features import get_features_from_config
from srl_utils import deserialize
from srl_utils import read_json
from tagger import DBLSTMTagger
from tagger import KEEP_PROB_KEY


class TaggerTrainer(object):
    def __init__(self, flags, extractor):
        super(TaggerTrainer, self).__init__()
        self.save_path = flags.save
        self.load_path = flags.load
        self.script_path = flags.script

        self._read_conf(flags.config)
        self.features = get_features_from_config(flags.config)

        self.extractor = extractor(feats=self.features)
        self.extractor.load(flags.vocab)

        self.label_vocab = self.extractor.extractors[LABEL_KEY].indices

        self.reverse_label_vocab = [None] * len(self.label_vocab)
        for key, val in self.label_vocab.iteritems():
            self.reverse_label_vocab[val] = key

        self.transition_params = TaggerTrainer._create_transition_matrix(self.reverse_label_vocab)

        if flags.train:
            self.training_iterator = BatchIterator(deserialize(flags.train), self.batch_size, features=self.features,
                                                   num_buckets=self.num_buckets, max_length=self.max_length, end_pad=self.crf)
        if flags.valid:
            self.validation_iterator = BatchIterator(deserialize(flags.valid), self.batch_size, features=self.features,
                                                     end_pad=self.crf)
        if flags.test:
            self.test_iterator = BatchIterator(deserialize(flags.test), self.batch_size, features=self.features, end_pad=self.crf)

    def train(self):
        with tf.Session() as sess:
            graph = self._load_graph()
            graph.train()
            if self.load_path:
                graph.saver.restore(sess, self.load_path)
            else:
                sess.run(tf.global_variables_initializer())
                graph.initialize_embeddings(sess)

            current_epoch, step, max_score = 1, 0, float('-inf')
            patience = 0
            while current_epoch <= self.max_epochs:
                logging.info('Epoch %d', current_epoch)
                then = time.time()
                with tqdm(total=self.training_iterator.size, leave=False, unit=' instances') as bar:
                    for batch in self.training_iterator.epoch():
                        feed = {graph.feed_dict[k]: batch[k] for k in batch.keys()}
                        feed[graph.feed_dict[KEEP_PROB_KEY]] = self.keep_prob
                        sess.run(graph.train_step, feed_dict=feed)
                        step += 1
                        bar.update(len(batch[LABEL_KEY]))
                sess.run(graph.global_step_increment)  # increment global step variable associated with graph
                logging.info('Training for epoch %d completed in %f seconds.', current_epoch, time.time() - then)

                score = self._test(graph=graph, sess=sess, iterator=self.validation_iterator)
                if score >= max_score:
                    max_score = score
                    patience = 0
                    if self.save_path:
                        save_path = graph.saver.save(sess, self.save_path, global_step=graph.global_step)
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
        raise NotImplementedError

    def _read_conf(self, conf_json):
        conf = read_json(conf_json)
        logging.info(conf)
        self.crf = conf.get('crf')
        self.max_epochs = conf['max_epochs']
        self.batch_size = conf['batch_size']
        self.keep_prob = conf['keep_prob']
        self.lstm_hidden_dim = conf['lstm_hidden_dim']
        self.lstm_num_layers = conf['lstm_num_layers']
        self.max_length = conf['max_length']
        self.num_buckets = conf['num_buckets']
        self.dblstm = conf.get('dblstm') or False

    def _load_graph(self):
        return DBLSTMTagger(features=self.features, num_classes=len(self.label_vocab), num_layers=self.lstm_num_layers,
                            state_dim=self.lstm_hidden_dim, transition_params=self.transition_params,
                            crf=self.crf, dblstm=self.dblstm)

    @staticmethod
    def _create_transition_matrix(labels):
        num_tags = len(labels)
        transition_params = np.zeros([num_tags, num_tags], dtype=np.float32)
        for i, prev_label in enumerate(labels):
            for j, label in enumerate(labels):
                if i != j and label[0] == 'I' and not prev_label == 'B' + label[1:]:
                    transition_params[i, j] = np.NINF
        return transition_params


class BatchIterator(object):
    def __init__(self, data, batch_size, features, num_buckets=5, max_length=99999, end_pad=False, lr_pad=2):
        super(BatchIterator, self).__init__()
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.size = len(data)
        self.features = features
        self.end_pad = 1 if end_pad else 0
        self.lr_pad = lr_pad

        data = [x for x in data if x[LENGTH_KEY] <= max_length]
        data.sort(key=lambda inst: inst[LENGTH_KEY])
        self.bucket_size = self.size // num_buckets
        self.data = []
        for bucket in range(num_buckets):
            self.data.append(data[bucket * self.bucket_size: (bucket + 1) * self.bucket_size])
        self.data[-1].extend(data[self.bucket_size * num_buckets:])  # add remaining instances
        self.pointer = np.array([0] * num_buckets)

    def max_steps(self):
        return len(self.data) // self.batch_size

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
            random.shuffle(self.data[i])
            self.pointer[i] = 0

    def _prepare_batch(self, batch):
        lengths = [instance[LENGTH_KEY] + self.end_pad for instance in batch]
        max_length = max(lengths)  # minimum length 2 due to https://github.com/tensorflow/tensorflow/issues/7751
        labels = self._pad_vals(LABEL_KEY, batch, max_length)
        feed_dict = {LABEL_KEY: labels, LENGTH_KEY: lengths}
        for feature in self.features:
            if feature.function is None:
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

    def _pad_list_feature(self, key, batch, maxlen, max_feat_length):
        padded = np.empty([len(batch), maxlen, max_feat_length], dtype=np.int32)
        padded.fill(PAD_INDEX)
        for i, sentence in enumerate(padded):
            features = batch[i][key]
            for index, word in enumerate(features):
                sentence[index, 0:self.lr_pad] = START_INDEX
                sentence[index, self.lr_pad:word.size + self.lr_pad] = word[:max_feat_length - self.lr_pad]
                end_word = word.size + self.lr_pad
                sentence[index, end_word:end_word + self.lr_pad] = END_INDEX
        return padded
