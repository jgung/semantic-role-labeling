import argparse
import logging
import subprocess
import sys
import tempfile
import time

import tensorflow as tf
from tqdm import tqdm

from srl.common.constants import LABEL_KEY, LENGTH_KEY, MARKER_KEY, SENTENCE_INDEX, WORDS
from srl.common.srl_utils import configure_logger
from srl.data.features import SequenceInstanceProcessor
from srl.data.readers import chunk
from srl.model.trainer import TaggerTrainer

FLAGS = None


class DeepSrlTrainer(TaggerTrainer):

    def __init__(self, flags, extractor):
        super(DeepSrlTrainer, self).__init__(flags, extractor)
        self.marker_index = self.extractor.extractors[MARKER_KEY].indices['1']

    def _test(self, iterator):
        then = time.time()
        pred_ys, gold_ys, words, indices, ids = [], [], [], [], []
        with tqdm(total=iterator.size, leave=False, unit=' instances') as bar:
            for batch in iterator.epoch():
                gold_ys.extend([gold[:stop] for (gold, stop) in zip(batch[LABEL_KEY], batch[LENGTH_KEY])])
                pred_ys.extend([self._decode(pred, stop) for (pred, stop) in zip(self._logits(batch), batch[LENGTH_KEY])])
                words.extend([sentence[:stop] for sentence, stop in zip(batch[WORDS], batch[LENGTH_KEY])])
                indices.extend([sentence[:stop] for sentence, stop in zip(batch[MARKER_KEY], batch[LENGTH_KEY])])
                ids.extend(batch[SENTENCE_INDEX])
                bar.update(len(batch[LABEL_KEY]))
        logging.info('Evaluation completed in %d seconds.', time.time() - then)
        return self.evaluate(words, pred_ys, gold_ys, indices, ids)

    def evaluate(self, words, pred_ys, gold_ys, indices, ids):
        output_file = open(self.output_file, 'w+b') if self.output_file else tempfile.NamedTemporaryFile()
        with tempfile.NamedTemporaryFile() as gold_temp, output_file as pred_temp:
            self._write_to_file(gold_temp, words, gold_ys, indices, ids)
            self._write_to_file(pred_temp, words, pred_ys, indices, ids)
            result = subprocess.check_output(['perl', self.script_path, gold_temp.name, pred_temp.name]).decode('utf-8')
            logging.info('\n%s', result)
            return float(result.strip().split('\n')[6].strip().split()[6])

    def _write_to_file(self, output_file, xs, ys, indices, ids):
        prev_sentence = -1

        predicates = []
        args = []
        for words, labels, markers, sentence in zip(xs, ys, indices, ids):
            if prev_sentence != sentence:
                prev_sentence = sentence
                if predicates:
                    line = ''
                    for index, predicate in enumerate(predicates):
                        line += '{} {}\n'.format(predicate, " ".join([prop[index] for prop in args]))
                    output_file.write(line + '\n')
                    predicates = []
                    args = []
            if not predicates:
                predicates = ["-"] * markers.size
            index = markers.tolist().index(self.marker_index)
            predicates[index] = words[index]
            args.append(chunk([self.reverse_label_vocab[l] for l in labels], conll=True))

        if predicates:
            line = ''
            for index, predicate in enumerate(predicates):
                line += '{} {}\n'.format(predicate, " ".join([prop[index] for prop in args]))
            output_file.write(line + '\n')

        output_file.flush()
        output_file.seek(0)

    def _predict(self, iterator):
        predictions = []
        for batch in iterator.epoch():
            preds = [self._decode(pred, stop, convert=True) for (pred, stop) in zip(self._logits(batch), batch[LENGTH_KEY])]
            predictions.extend(preds)
        return predictions


def main(_):
    configure_logger(FLAGS.log)
    srl_trainer = DeepSrlTrainer(FLAGS, SequenceInstanceProcessor)
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
    parser.add_argument('--output', required=False, type=str, help='Output file to store test predictions')
    parser.add_argument('--vocab', required=True, type=str, help='Path to directory containing vocabulary files.')
    parser.add_argument('--script', required=True, type=str, help='Path to evaluation script.')
    parser.add_argument('--log', default='srl_trainer.log', type=str, help='Path to output log.')
    parser.add_argument('--config', required=True, type=str, help='Path to json configuration file.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
