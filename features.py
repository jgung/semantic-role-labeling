import math

import numpy as np
import tensorflow as tf

from srl_utils import read_json, serialize, initialize_vectors, deserialize, read_vectors
from constants import PAD_WORD, UNKNOWN_WORD, START_WORD, END_WORD
from constants import UNKNOWN_INDEX
from constants import LABEL_KEY, LENGTH_KEY


class Feature(object):
    def __init__(self, name, dim, extractor, rank, keep_prob=1, initializer=None, func=None, base_feature=False):
        """
        Generic feature.
        :param name: name of feature
        :param dim: dimensionality of feature embedding
        :param extractor: feature extractor
        :param rank: rank of feature tensor
        :param keep_prob: dropout keep probability
        :param initializer: numpy matrix to initialize feature embedding
        :param func: function applied to compute feature
        :param base_feature: true if this feature is used to determine the length of each instance
        """
        super(Feature, self).__init__()
        self.name = name
        self.dim = dim
        self.extractor = extractor
        self.keep_prob = keep_prob
        self.initializer = initializer
        self.function = func
        self.rank = rank
        self.base_feature = base_feature

        self.embedding = None
        self.left_padding = 0
        self.right_padding = 0
        self.max_length = -1

    def vocab_size(self):
        return len(self.extractor.indices)


def get_features_from_config(config):
    return [get_feature(feat) for feat in read_json(config)['features']]


def get_feature(feat_dict):
    def _get_extractor(extractor_dict, _rank):
        if extractor_dict is not None:
            extractor_name = extractor_dict['name']
            key = extractor_dict['key']
            if extractor_name == 'chars':
                if _rank == 4:
                    return NestedListFeatureExtractor(key=key, apply_func=lambda x: list(x))
                return ListFeatureExtractor(key=key, apply_func=lambda x: list(x))
            if extractor_name == 'key':
                return KeyFeatureExtractor(key=key)
            if extractor_name == 'lower':
                if _rank == 3:
                    return ListFeatureExtractor(key=key, apply_func=lambda x: [word.lower() for word in x])
                return KeyFeatureExtractor(key=key, apply_func=lambda x: x.lower())

        return IdentityExtractor()

    def _get_composition_function(func_dict, input_dim):
        if func_dict is not None:
            if func_dict['name'] == "ConvNet":
                return ConvNet(input_dim=input_dim, window_size=func_dict['window_size'], num_filters=func_dict['num_filters'],
                               max_length=func_dict['max_length'])
        return None

    name = feat_dict['name']
    dim = feat_dict['dim']
    rank = feat_dict.get('rank', 2)

    extractor = _get_extractor(feat_dict.get('extractor'), rank)
    keep_prob = feat_dict.get('keep_prob', 1)
    initializer = feat_dict.get('initializer')
    base_feat = feat_dict.get('base', False)

    func = _get_composition_function(feat_dict.get('function'), dim)
    feature = Feature(name=name, dim=dim, extractor=extractor, rank=rank,
                      keep_prob=keep_prob, initializer=initializer, func=func, base_feature=base_feat)
    feature.left_padding = feat_dict.get('left_padding', 0)
    feature.right_padding = feat_dict.get('right_padding', 0)
    return feature


class ConvNet(object):
    def __init__(self, input_dim, window_size, num_filters, max_length):
        super(ConvNet, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.num_filters = num_filters
        self.max_length = max_length

    def apply(self, feature):
        return self.get_cnn_step(inputs=feature, input_dim=self.input_dim, seq_len=self.max_length, char_filters=self.num_filters,
                                 window_size=self.window_size)

    @staticmethod
    def get_cnn_step(inputs, input_dim, seq_len, char_filters, window_size=2):
        shape = tf.shape(inputs)
        # flatten sequences for input
        inputs = tf.reshape(inputs, shape=[-1, shape[-2], shape[-1], 1])
        # convolution weights
        limit = math.sqrt(3.0 / char_filters)
        conv_filter = tf.get_variable("conv_w", [window_size, input_dim, 1, char_filters],
                                      initializer=tf.random_uniform_initializer(-limit, limit))
        conv_bias = tf.get_variable("conv_b", [char_filters], initializer=tf.zeros_initializer)
        # convolution ops
        conv = tf.nn.conv2d(input=inputs, filter=conv_filter, strides=[1, 1, 1, 1], padding="VALID")
        relu = tf.nn.relu(tf.nn.bias_add(value=conv, bias=conv_bias))
        pool = tf.nn.max_pool(value=relu, ksize=[1, seq_len - window_size + 1, 1, 1], strides=[1, 1, 1, 1],
                              padding="VALID")
        # unflatten
        char_conv = tf.reshape(pool, shape=[-1, shape[1], char_filters])
        return char_conv


class FeatureExtractor(object):
    def __init__(self, train=False, indices=None, unknown_word=UNKNOWN_WORD):
        super(FeatureExtractor, self).__init__()
        self.train = train
        self.indices = indices
        if self.indices is None:
            self.indices = {PAD_WORD: 0, UNKNOWN_WORD: 1, START_WORD: 2, END_WORD: 3}
        if unknown_word not in self.indices:
            # noinspection PyTypeChecker
            self.indices[unknown_word] = len(self.indices)
        self.unknown_word = unknown_word

    def initialize_indices(self, values):
        for value in values:
            if value not in self.indices:
                self.indices[value] = len(self.indices)

    def extract(self, sequence):
        """
        Extracts a feature for each value in the sequence, applying a function.
        :param sequence: feature extraction input
        :return: list of features
        """
        indices = [self._extract_single(self._apply(result)) for result in self._get_values(sequence)]
        return np.array(indices, np.int32)

    def _extract_single(self, feat):
        """
        Extract the index of a single feature, updating the index dictionary if training.
        :param feat: feature value
        :return: feature index
        """
        index = self.indices.get(feat)
        if index is None:
            if self.train:
                index = len(self.indices)
                self.indices[feat] = index
            else:
                index = UNKNOWN_INDEX
        return index

    def _apply(self, value):
        """
        Function applied to each token in a sequence.
        :param value: input token
        :return: transformed value
        """
        raise NotImplementedError()

    def _get_values(self, sequence):
        """
        :param sequence: dictionary of sequences for feature extraction
        :return: list of targets for feature extraction
        """
        raise NotImplementedError()


class IdentityExtractor(FeatureExtractor):
    def _apply(self, value):
        return value

    def _get_values(self, sequence):
        return sequence


class KeyFeatureExtractor(FeatureExtractor):
    def __init__(self, key, train=False, indices=None, unknown_word=UNKNOWN_WORD, apply_func=lambda x: x):
        super(KeyFeatureExtractor, self).__init__(train, indices, unknown_word)
        self.key = key
        self.apply_func = apply_func

    def _get_values(self, sequence):
        return sequence[self.key]

    def _apply(self, value):
        return self.apply_func(value)


class ListFeatureExtractor(KeyFeatureExtractor):
    def __init__(self, key, train=False, indices=None, unknown_word=UNKNOWN_WORD, apply_func=lambda x: x):
        super(ListFeatureExtractor, self).__init__(key, train, indices, unknown_word, apply_func)

    def extract(self, sequence):
        all_indices = []
        for value in self._get_values(sequence):
            indices = [self._extract_single(result) for result in self._apply(value)]
            all_indices.append(np.array(indices, np.int32))
        return all_indices


class NestedListFeatureExtractor(KeyFeatureExtractor):
    def __init__(self, key, train=False, indices=None, unknown_word=UNKNOWN_WORD, apply_func=lambda x: x):
        super(NestedListFeatureExtractor, self).__init__(key, train, indices, unknown_word, apply_func)

    def extract(self, sequence):
        all_indices = []
        for sub_list in self._get_values(sequence):
            sub_list_indices = []
            for value in sub_list:
                indices = [self._extract_single(result) for result in self._apply(value)]
                sub_list_indices.append(np.array(indices, np.int32))
            all_indices.append(sub_list_indices)
        return all_indices


class SequenceInstanceProcessor(object):
    def __init__(self, feats):
        """
        Instance processor -- performs feature extraction.
        :param feats: list of features
        """
        super(SequenceInstanceProcessor, self).__init__()
        self.features = feats
        self.extractors = {feat.name: feat.extractor for feat in self.features}
        self.extractors[LABEL_KEY] = KeyFeatureExtractor(LABEL_KEY, indices={}, unknown_word='O')
        self.resources = {}

    def extract(self, sequence, labels=None):
        instance = {}
        for feature in self.features:
            instance[feature.name] = feature.extractor.extract(sequence)
            if feature.base_feature:
                feat = instance[feature.name]
                instance[LENGTH_KEY] = isinstance(feat, list) and len(feat) or feat.size
        if labels:
            instance[LABEL_KEY] = self.extractors[LABEL_KEY].extract(sequence)
        return instance

    def read_instances(self, sentences, train=False):
        pass

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
                vectors, dim = read_vectors(path, unk_word=feature.extractor.unknown_word, pad_word=PAD_WORD)
                self.resources[feature.name] = vectors, dim
                feature.extractor.initialize_indices(vectors.keys())
                feature.extractor.train = False  # don't update vocabularies initialized from word embeddings
