import argparse

import features
from constants import LABEL_KEY
from srl_reader import Conll2003Reader
from srl_utils import serialize


class NerFeatureExtractor(features.SequenceInstanceProcessor):
    def __init__(self, feats):
        super(NerFeatureExtractor, self).__init__(feats)

    def read_instances(self, sentences, train=False):
        """
        Read NER instances from a list of NER annotated sentences
        :param sentences: NER sentences
        :param train: train vocabularies during instance extraction (fixed if False)
        :return: NER instances
        """
        if train:
            self._init_vocabularies()
        results = []
        for sentence in sentences:
            results.append(self.extract(sentence, sentence[LABEL_KEY]))
        return results


def main(flags):
    reader = Conll2003Reader()
    feats = features.get_features_from_config(flags.config)
    feature_extractor = NerFeatureExtractor(feats=feats)
    train = True
    if flags.mode != 'new':
        feature_extractor.load(flags.vocab)
        if flags.mode == 'load':
            feature_extractor.test()
            train = False
    data = reader.read_files(flags.input, flags.ext)
    print('Processing {} sentences from {}'.format(len(data), flags.input))
    instances = feature_extractor.read_instances(data, train=train)
    print('Saving {} processed sentences to {}'.format(len(instances), flags.output))
    serialize(instances, flags.output)
    if train:
        print('Saving updated feature vocabularies to {}'.format(flags.vocab))
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
    main(parser.parse_args())
