import argparse
import logging
import subprocess
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from tqdm import tqdm

from srl.common.constants import LABEL_KEY, LENGTH_KEY, MARKER_KEY
from srl.common.srl_utils import configure_logger
from srl.data.srl_feature_extractor import SrlFeatureExtractor
from srl.data.srl_reader import chunk
from srl.model.trainer import TaggerTrainer

FLAGS = None


class DeepSrlTrainer(TaggerTrainer):
    def __init__(self, flags, extractor):
        super(DeepSrlTrainer, self).__init__(flags, extractor)
        self.marker_index = self.extractor.extractors[MARKER_KEY].indices['1']

    def _test(self, graph, sess, iterator):
        then = time.time()
        pred_ys, gold_ys, words, indices = [], [], [], []
        with tqdm(total=iterator.size, leave=False, unit=' instances') as bar:
            for batch in iterator.epoch():
                feed = {graph.feed_dict[k]: batch[k] for k in batch.keys()}
                for key in graph.dropout_keys:
                    feed[graph.feed_dict[key]] = 1.0
                logits = sess.run(graph.scores, feed_dict=feed)

                gold_ys.extend([gold[:stop] for (gold, stop) in zip(batch[LABEL_KEY], batch[LENGTH_KEY])])
                pred_ys.extend(
                    [viterbi_decode(score=pred[:stop], transition_params=graph.transition_matrix())[0] for
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
                line += '{} {}\n'.format(marker == self.marker_index and 'x' or '-', predicted)
            output_file.write(line + '\n')
        output_file.flush()
        output_file.seek(0)


def main(_):
    configure_logger(FLAGS.log)
    srl_trainer = DeepSrlTrainer(FLAGS, SrlFeatureExtractor)
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
