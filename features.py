import tensorflow as tf

from srl_utils import read_json

PAD_WORD = "<PAD>"
UNKNOWN_WORD = "<UNK>"
PAD_INDEX = 0
UNKNOWN_INDEX = 1
LABEL_KEY = "labels"
LENGTH_KEY = "lengths"


class Feature(object):
    def __init__(self, name, dim, extractor, initializer=None, func=None):
        """
        Generic feature.
        :param name: name of feature
        :param dim: dimensionality of feature embedding
        :param extractor: feature extractor
        :param initializer: numpy matrix to initialize feature embedding
        :param func: function applied to compute feature
        """
        super(Feature, self).__init__()
        self.name = name
        self.dim = dim
        self.extractor = extractor
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
    extractor = _get_extractor(feat_dict.get('extractor'))
    initializer = feat_dict.get('initializer')
    func = _get_composition_function(feat_dict.get('function'), dim)
    return Feature(name=name, dim=dim, extractor=extractor, initializer=initializer, func=func)


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
        conv_filter = tf.get_variable("conv_w", [window_size, input_dim, 1, char_filters],
                                      initializer=tf.random_normal_initializer(0, 0.01))
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
    def __init__(self, train=False, indices=None):
        super(FeatureExtractor, self).__init__()
        self.train = train
        self.indices = indices
        if not self.indices:
            self.indices = {PAD_WORD: 0, UNKNOWN_WORD: 1}
        self.list_feature = False

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
            if not index:
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
    def __init__(self, train=False, indices=None):
        super(ListFeatureExtractor, self).__init__(train, indices)
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
                if not index and self.train:
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
    def __init__(self, key, train=False, indices=None):
        super(CharacterFeatureFunction, self).__init__(train, indices)
        self.key = key

    def _apply(self, value):
        return list(value)

    def _get_values(self, sequence):
        return sequence[self.key]


class KeyFeatureExtractor(IdentityExtractor):
    def __init__(self, key, train=False, indices=None):
        super(KeyFeatureExtractor, self).__init__(train, indices)
        self.key = key

    def _get_values(self, sequence):
        return sequence[self.key]

    def _apply(self, value):
        return value


class LowerCaseExtractor(KeyFeatureExtractor):
    def _apply(self, value):
        return super(LowerCaseExtractor, self)._apply(value).lower()
