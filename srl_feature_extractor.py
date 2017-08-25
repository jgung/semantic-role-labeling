import argparse

import numpy as np

import features
from srl_reader import Conll2005Reader
from srl_utils import deserialize
from srl_utils import initialize_vectors
from srl_utils import read_vectors
from srl_utils import serialize
from features import LABEL_KEY, LENGTH_KEY

MARKER_KEY = 'markers'


class SrlFeatureExtractor(object):
    def __init__(self, feats):
        """
        SRL feature extractor.
        :param feats: list of features
        """
        super(SrlFeatureExtractor, self).__init__()
        self.features = feats
        self.extractors = {feat.name: feat.extractor for feat in self.features}
        self.extractors[LABEL_KEY] = features.KeyFeatureExtractor(LABEL_KEY)
        self.resources = {}

    def extract(self, sentence):
        instance = {}
        for name, extractor in self.extractors.items():
            if extractor.list_feature:
                feats = [np.array(value, dtype=np.int32) for value in extractor.extract(sentence)]
            else:
                feats = np.array(extractor.extract(sentence), dtype=np.int32)
            instance[name] = feats
        instance[LENGTH_KEY] = instance[LABEL_KEY].shape[0]
        return instance

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

    def train(self, train=True):
        """
        Leave feature vocabularies fixed when extracting features.
        :param train: if False, update feature vocabularies
        """
        for extractor in self.extractors.values():
            extractor.train = train

    def test(self):
        """
        When extracting features, update feature vocabularies.
        """
        self.train(False)

    def save(self, out_path):
        """
        Serialize feature vocabularies and training resources such as vectors.
        :param out_path:  path to vocabulary directory
        """
        for feature in self.features:
            serialize(feature.extractor.indices, out_path, feature.name)
        serialize(self.extractors[LABEL_KEY].indices, out_path, LABEL_KEY)
        for feature in self.features:
            if feature.initializer:
                vectors, dim = self.resources[feature.name]
                feature.embedding = initialize_vectors(vector_map=vectors, vocabulary=feature.extractor.indices, dim=dim)
                feature.dim = dim
                serialize(feature.embedding, out_path, feature.initializer['pkl_path'])

    def load(self, in_path):
        """
        Load serialized vocabularies and training resources.
        :param in_path: path to vocabulary directory
        """
        for feature in self.features:
            feature.extractor.indices = deserialize(in_path, feature.name)
            if feature.initializer:
                feature.embedding = deserialize(in_path, feature.initializer['pkl_path'])
                feature.dim = feature.embedding.shape[1]
        self.extractors[LABEL_KEY].indices = deserialize(in_path, LABEL_KEY)

    def _init_vocabularies(self):
        for extractor in self.extractors.values():
            extractor.train = True
        for feature in self.features:
            if feature.initializer:
                path = feature.initializer['initializer_path']
                vectors, dim = read_vectors(path, unk_word=features.UNKNOWN_WORD, pad_word=features.PAD_WORD)
                self.resources[feature.name] = vectors, dim
                feature.extractor.initialize_indices(vectors.keys())
                feature.extractor.train = False


def main(flags):
    reader = Conll2005Reader()
    feats = features.get_features_from_config(flags.config)
    feature_extractor = SrlFeatureExtractor(feats=feats)
    train = True
    if flags.mode is not 'new':
        feature_extractor.load(flags.vocab)
        if flags.mode is 'load':
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
    main(parser.parse_args())
