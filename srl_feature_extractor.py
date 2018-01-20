import argparse

import features
from constants import LENGTH_KEY, LABEL_KEY
from srl_reader import ConllPhraseReader
from srl_utils import serialize

MARKER_KEY = 'markers'
PHRASES = 'phrases'


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
        for sentence in sentences:
            results.append(self.extract(sentence, sentence[LABEL_KEY]))
        return results


class PhraseSrlFeatureExtractor(SrlFeatureExtractor):
    def __init__(self, feats):
        super(PhraseSrlFeatureExtractor, self).__init__(feats)

    def extract(self, sequence, labels=None):
        instance = {}
        for feature in self.features:
            instance[feature.name] = feature.extractor.extract(sequence)
            if feature.base_feature:
                instance[LENGTH_KEY] = instance[feature.name].size
        if labels:
            instance[LABEL_KEY] = self.extractors[LABEL_KEY]
        return instance


def main(flags):
    # reader = Conll2005Reader() if flags.dataset == 'conll05' else Conll2012Reader()
    reader = ConllPhraseReader()

    feats = features.get_features_from_config(flags.config)
    # feature_extractor = SrlFeatureExtractor(feats=feats)
    feature_extractor = PhraseSrlFeatureExtractor(feats=feats)

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
