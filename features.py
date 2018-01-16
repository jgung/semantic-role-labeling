import tensorflow as tf
import numpy as np
import math

from srl_utils import read_json, serialize, initialize_vectors, deserialize, read_vectors

PAD_WORD = "<PAD>"
UNKNOWN_WORD = "<UNK>"
START_WORD = "<BOS>"
END_WORD = "<EOS>"
PAD_INDEX = 0
UNKNOWN_INDEX = 1
START_INDEX = 2
END_INDEX = 3

LABEL_KEY = "labels"
LENGTH_KEY = "lengths"


class Feature(object):
    def __init__(self, name, dim, extractor, dropout=False, initializer=None, func=None):
        """
        Generic feature.
        :param name: name of feature
        :param dim: dimensionality of feature embedding
        :param extractor: feature extractor
        :param dropout: use dropout
        :param initializer: numpy matrix to initialize feature embedding
        :param func: function applied to compute feature
        """
        super(Feature, self).__init__()
        self.name = name
        self.dim = dim
        self.extractor = extractor
        self.dropout = dropout
        self.initializer = initializer
        self.function = func
        self.embedding = None

    def vocab_size(self):
        return len(self.extractor.indices)


def get_features_from_config(config):
    return [get_feature(feat) for feat in read_json(config)['features']]


def get_feature(feat_dict):
    def _get_extractor(extractor_dict):
        if extractor_dict is not None:
            extractor_name = extractor_dict['name']
            if extractor_name == 'key':
                return KeyFeatureExtractor(extractor_dict['key'])
            if extractor_name == 'lower':
                return LowerCaseExtractor(extractor_dict['key'])
            if extractor_name == 'chars':
                return CharacterFeatureFunction(extractor_dict['key'])
        return IdentityExtractor()

    def _get_composition_function(func_dict, input_dim):
        if func_dict is not None:
            if func_dict['name'] == "ConvNet":
                return ConvNet(input_dim=input_dim, window_size=func_dict['window_size'], num_filters=func_dict['num_filters'],
                               max_length=func_dict['max_length'])
        return None

    name = feat_dict['name']
    dim = feat_dict['dim']
    dropout = feat_dict.get('dropout', False)
    extractor = _get_extractor(feat_dict.get('extractor'))
    initializer = feat_dict.get('initializer')
    func = _get_composition_function(feat_dict.get('function'), dim)
    return Feature(name=name, dim=dim, extractor=extractor, dropout=dropout, initializer=initializer, func=func)


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
        self.list_feature = False
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
        indices = []
        for value in self._get_values(sequence):
            result = self._apply(value)
            index = self.indices.get(result)
            if index is None:
                if self.train:
                    index = len(self.indices)
                    self.indices[result] = index
                else:
                    index = UNKNOWN_INDEX
            indices.append(index)
        return indices

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


class ListFeatureExtractor(FeatureExtractor):
    def __init__(self, train=False, indices=None, unknown_word=UNKNOWN_WORD):
        super(ListFeatureExtractor, self).__init__(train, indices, unknown_word)
        self.list_feature = True

    def initialize_indices(self, values):
        super(ListFeatureExtractor, self).initialize_indices(values)

    def extract(self, sequence):
        all_indices = []
        for value in self._get_values(sequence):
            results = self._apply(value)
            indices = []
            for result in results:
                index = self.indices.get(result)
                if index is None and self.train:
                    if self.train:
                        index = len(self.indices)
                        self.indices[result] = index
                    else:
                        index = UNKNOWN_INDEX
                indices.append(index)
            all_indices.append(indices)
        return all_indices

    def _get_values(self, sequence):
        return []

    def _apply(self, value):
        return []


class CharacterFeatureFunction(ListFeatureExtractor):
    def __init__(self, key, train=False, indices=None, unknown_word=UNKNOWN_WORD):
        super(CharacterFeatureFunction, self).__init__(train, indices, unknown_word)
        self.key = key

    def _apply(self, value):
        return list(value)

    def _get_values(self, sequence):
        return sequence[self.key]


class KeyFeatureExtractor(IdentityExtractor):
    def __init__(self, key, train=False, indices=None, unknown_word=UNKNOWN_WORD):
        super(KeyFeatureExtractor, self).__init__(train, indices, unknown_word)
        self.key = key

    def _get_values(self, sequence):
        return sequence[self.key]

    def _apply(self, value):
        return value


class LowerCaseExtractor(KeyFeatureExtractor):
    def _apply(self, value):
        return super(LowerCaseExtractor, self)._apply(value).lower()


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
                feature.extractor.train = False
