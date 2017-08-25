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

from features import LABEL_KEY, LENGTH_KEY
from ner_feature_extractor import NerFeatureExtractor
from srl_utils import configure_logger
from tagger import KEEP_PROB_KEY
from trainer import TaggerTrainer

FLAGS = None


class DeepNerTrainer(TaggerTrainer):
    def __init__(self, flags, extractor):
        super(DeepNerTrainer, self).__init__(flags, extractor)

    def _test(self, graph, sess, iterator):
        then = time.time()
        pred_ys, gold_ys = [], []
        with tqdm(total=iterator.size, leave=False, unit=' instances') as bar:
            for batch in iterator.epoch():
                feed = {graph.feed_dict[k]: batch[k] for k in batch.keys()}
                feed[graph.feed_dict[KEEP_PROB_KEY]] = 1.0
                logits = sess.run(graph.scores, feed_dict=feed)
                gold_ys.extend([gold[:stop] for (gold, stop) in zip(batch[LABEL_KEY], batch[LENGTH_KEY])])
                pred_ys.extend(
                    [viterbi_decode(score=pred[:stop], transition_params=self.transition_params)[0] for
                     (pred, stop) in zip(logits, batch[LENGTH_KEY])])
                bar.update(len(batch[LABEL_KEY]))
        logging.info('Evaluation completed in %d seconds.', time.time() - then)
        return self.evaluate(pred_ys, gold_ys)

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
