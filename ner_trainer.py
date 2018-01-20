import argparse
import logging
import re
import subprocess
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from tqdm import tqdm

from constants import LABEL_KEY, LENGTH_KEY
from ner_feature_extractor import NerFeatureExtractor
from srl_utils import configure_logger
from tagger import DBLSTMTagger
from trainer import TaggerTrainer

FLAGS = None


class NerTagger(DBLSTMTagger):
    def __init__(self, features, num_layers, state_dim, num_classes, transition_params=None, crf=True, dblstm=False):
        super(NerTagger, self).__init__(features, num_layers, state_dim, num_classes, transition_params, crf, dblstm)

    def training_op(self):
        # Use optimization algorithm described in Ma and Hovy 2016
        learning_rate = tf.train.inverse_time_decay(learning_rate=0.015,
                                                    global_step=self.global_step,
                                                    decay_steps=1,
                                                    decay_rate=0.05)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        tvars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, tvars)
        return optimizer.apply_gradients(zip(gradients, tvars))


class DeepNerTrainer(TaggerTrainer):
    def __init__(self, flags, extractor):
        super(DeepNerTrainer, self).__init__(flags, extractor)

    def _load_graph(self):
        return NerTagger(features=self.features, num_classes=len(self.label_vocab), num_layers=self.lstm_num_layers,
                         state_dim=self.lstm_hidden_dim, transition_params=self.transition_params,
                         crf=self.crf, dblstm=self.dblstm)

    def _test(self, graph, sess, iterator):
        then = time.time()
        pred_ys, gold_ys = [], []
        with tqdm(total=iterator.size, leave=False, unit=' instances') as bar:
            for batch in iterator.epoch():
                feed = {graph.feed_dict[k]: batch[k] for k in batch.keys()}
                for key in graph.dropout_keys:
                    feed[graph.feed_dict[key]] = 1.0
                logits = sess.run(graph.scores, feed_dict=feed)
                lengths = [l - 1 for l in batch[LENGTH_KEY]] if self.crf else batch[LENGTH_KEY]  # remove padding before eval
                gold_ys.extend([gold[:stop] for (gold, stop) in zip(batch[LABEL_KEY], lengths)])
                transition = graph.transition_matrix()
                pred_ys.extend([viterbi_decode(score=pred[:stop], transition_params=transition)[0] for
                                (pred, stop) in zip(logits, lengths)])
                bar.update(len(batch[LABEL_KEY]))
        logging.info('Evaluation completed in %d seconds.', time.time() - then)
        return self.evaluate(gold_ys, pred_ys)

    def evaluate(self, gold_seqs, pred_seqs):
        with tempfile.NamedTemporaryFile(mode='w') as temp:
            for gold_labels, pred_labels in zip(gold_seqs, pred_seqs):
                for x, y in zip(gold_labels, pred_labels):
                    temp.write("x {} {}\n".format(self.reverse_label_vocab[x], self.reverse_label_vocab[y]))
                temp.write("\n")  # sequence break
            temp.flush()
            temp.seek(0)
            result = subprocess.check_output(["perl", self.script_path], stdin=temp).decode('utf-8')
            logging.info(result)
            return float(re.split('\s+', re.split('\n', result)[1].strip())[7])


def main(_):
    configure_logger(FLAGS.log)
    ner = DeepNerTrainer(FLAGS, NerFeatureExtractor)
    if FLAGS.train:
        if not FLAGS.valid:
            raise ValueError('Missing required validation (dev) set. Use "--valid path/to/valid.pkl" to specify validation data.')
        ner.train()
    if FLAGS.test:
        ner.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='data/models/model', help='Path to save models/checkpoints.')
    parser.add_argument('--load', type=str, help='Path to load previously saved model.')
    parser.add_argument('--train', type=str, help='Binary (*.pkl) train file path.')
    parser.add_argument('--valid', type=str, help='Binary (*.pkl) validation file path.')
    parser.add_argument('--test', required=False, type=str, help='Binary (*.pkl) test file path.')
    parser.add_argument('--vocab', required=True, type=str, help='Path to directory containing vocabulary files.')
    parser.add_argument('--script', required=True, type=str, help='Path to evaluation script.')
    parser.add_argument('--log', default='ner_trainer.log', type=str, help='Path to output log.')
    parser.add_argument('--config', required=True, type=str, help='Path to json configuration file.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
