import argparse

import features
from features import LABEL_KEY
from srl_reader import Conll2005Reader, Conll2012Reader
from srl_utils import serialize

MARKER_KEY = 'markers'


class SrlFeatureExtractor(features.SequenceInstanceProcessor):
    def __init__(self, feats):
        super(SrlFeatureExtractor, self).__init__(feats)

    def read_instances(self, sentences, train=False):
        """
        Read SRL instances from a list of SRL annotated sentences.
        :param sentences: SRL sentences
        :param train: train vocabularies during instance extraction (fixed if False)
        :return: SRL instances
        """
        if train:
            self._init_vocabularies()
        results = []
        for sentence, predicates in sentences:
            for key, labels in predicates.iteritems():
                sentence[LABEL_KEY] = labels
                sentence[MARKER_KEY] = [index == key and '1' or '0' for index in range(0, len(labels))]
                results.append(self.extract(sentence))
        return results


def main(flags):
    reader = Conll2005Reader() if flags.dataset == 'conll05' else Conll2012Reader()
    feats = features.get_features_from_config(flags.config)
    feature_extractor = SrlFeatureExtractor(feats=feats)
    train = True
    if flags.mode != 'new':
        feature_extractor.load(flags.vocab)
        if flags.mode == 'load':
            feature_extractor.test()
            train = False
    instances = feature_extractor.read_instances(reader.read_files(flags.input, flags.ext), train=train)
    serialize(instances, flags.output)
    if train:
        feature_extractor.save(flags.vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='new', choices=['new', 'update', 'load'],
                        help="Create new vocabularies during extraction, update an existing, or only load a previous vocab.")
    parser.add_argument('--input', required=True, type=str, help='CoNLL-formatted input file path.')
    parser.add_argument('--output', required=True, type=str, help='Path to save pickled input.')
    parser.add_argument('--config', required=True, type=str, help='Path to configuration json.')
    parser.add_argument('--ext', default='conll', type=str, help='Input file extension.')
    parser.add_argument('--vocab', required=True, type=str, help='Vocab directory path.')
    parser.add_argument('--dataset', default='conll05', choices=['conll05', 'conll2012'], type=str,
                        help='Dataset (conll05 or conll2012).')
    main(parser.parse_args())
