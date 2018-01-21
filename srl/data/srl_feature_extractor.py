import argparse

from features import SequenceInstanceProcessor, get_features_from_config
from srl.common.constants import LABEL_KEY
from srl.common.srl_utils import serialize
from srl_reader import Conll2005Reader, Conll2012Reader, ConllPhraseReader


class SrlFeatureExtractor(SequenceInstanceProcessor):
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


def main(flags):
    if flags.dataset == 'conll2012':
        reader = Conll2012Reader()
    elif flags.dataset == 'phrase':
        reader = ConllPhraseReader()
    else:
        reader = Conll2005Reader()

    feats = get_features_from_config(flags.config)
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
    parser.add_argument('--dataset', default='conll05', choices=['conll05', 'conll2012', 'phrase'], type=str,
                        help='Dataset (conll05, conll2012, or phrase).')
    main(parser.parse_args())
