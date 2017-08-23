class Feature(object):
    def __init__(self, name, dim, vocab_size, initializer=None, subword=False, max_len=None, func=None):
        """
        Generic feature.
        :param name: name of feature
        :param dim: dimensionality of feature embedding
        :param vocab_size: size of vocabulary (# possible feature values)
        :param initializer: numpy matrix to initialize feature embedding
        :param subword: indicates a subword feature (multiple values for a single token) (such as character unigrams or bigrams)
        :param func: function applied to compute feature
        """
        super(Feature, self).__init__()
        self.name = name
        self.dim = dim
        self.vocab_size = vocab_size
        self.initializer = initializer
        self.subword = subword
        self.max_len = max_len
        if subword and not max_len:
            raise ValueError('Subword feature must include max_len.')
        self.function = func or Identity()


class Identity(object):
    def __init__(self):
        super(Identity, self).__init__()

    def apply(self, feature):
        return feature


