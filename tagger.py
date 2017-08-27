import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple
# noinspection PyProtectedMember
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope
# noinspection PyProtectedMember
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.util import nest
from tensorflow.contrib.crf import crf_log_likelihood
from features import LENGTH_KEY

KEEP_PROB_KEY = "keep_prob"


class DBLSTMTagger(object):
    def __init__(self, features, num_layers, state_dim, num_classes, transition_params=None, crf=True):
        super(DBLSTMTagger, self).__init__()
        self.features = features

        self.num_layers = num_layers
        self.state_dim = state_dim
        self.num_classes = num_classes
        self.crf = crf
        self.transition_params = transition_params

        self._embedding_placeholder = {}
        self._embedding_init = {}

        self.scores = None
        self.loss = None
        self.train_step = None
        self.saver = None

        self.feed_dict = {}
        self.dropout_keep_prob = self._add_placeholder(KEEP_PROB_KEY, tf.float32)
        self.sequence_lengths = self._add_placeholder(LENGTH_KEY, tf.int32, [None])

    def transition_matrix(self):
        if self.crf:
            return self.transition_params.eval()
        return self.transition_params

    def _add_placeholder(self, name, dtype, shape=None):
        placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)
        self.feed_dict[name] = placeholder
        return placeholder

    def initialize_embeddings(self, sess):
        for feature in self.features:
            if feature.embedding is not None:
                placeholder = self._embedding_placeholder[feature.name]
                init = self._embedding_init[feature.name]
                sess.run(init, feed_dict={placeholder: feature.embedding})

    def embedding_layer(self):
        with tf.name_scope('embedding_layer'):
            inputs = []
            for feature in self.features:
                embedding_matrix = tf.get_variable(name='{}_embedding_matrix'.format(feature.name),
                                                   shape=[feature.vocab_size(), feature.dim],
                                                   initializer=tf.random_normal_initializer(0, 0.01))
                if feature.embedding is not None:
                    self._embedding_placeholder[feature.name] = tf.placeholder(tf.float32, [feature.vocab_size(), feature.dim])
                    self._embedding_init[feature.name] = embedding_matrix.assign(self._embedding_placeholder[feature.name])
                shape = [None, None, None] if feature.function else [None, None]
                indices = self._add_placeholder(name=feature.name, dtype=tf.int32, shape=shape)
                embedding = tf.nn.embedding_lookup(params=embedding_matrix, ids=indices, name='{}_embedding'.format(feature.name))
                result = embedding if not feature.function else feature.function.apply(embedding)
                inputs.append(result)
            return tf.concat(inputs, 2, name="concatenated_inputs")

    def _dblstm_cell(self):
        return DropoutWrapper(HighwayLSTMCell(self.state_dim, initializer=tf.orthogonal_initializer()),
                              variational_recurrent=True, dtype=tf.float32, output_keep_prob=self.dropout_keep_prob)

    def inference_layer(self, inputs):
        with tf.name_scope('deep_bidirectional_rnn'):
            rnn_outputs, _ = deep_bidirectional_dynamic_rnn(
                [self._dblstm_cell() for _ in range(self.num_layers)],
                inputs, sequence_length=self.sequence_lengths, dtype=tf.float32)

        with tf.name_scope('linear_projection'):
            softmax_weights = tf.get_variable('softmax_W', [self.state_dim, self.num_classes],
                                              initializer=tf.random_normal_initializer(0, 0.01))
            softmax_bias = tf.get_variable('softmax_b', [self.num_classes], initializer=tf.constant_initializer(0.0))

            time_steps = tf.shape(rnn_outputs)[1]
            rnn_outputs = tf.reshape(rnn_outputs, [-1, self.state_dim],
                                     name="flatten_rnn_outputs_for_linear_projection")
            logits = tf.nn.xw_plus_b(x=rnn_outputs, weights=softmax_weights, biases=softmax_bias,
                                     name="softmax_projection")
            self.scores = tf.reshape(logits, [-1, time_steps, self.num_classes], name="unflatten_logits")

    def add_train_ops(self):
        with tf.name_scope('loss_ops'):
            labels = self._add_placeholder("labels", tf.int32, [None, None])
            if self.crf:
                log_likelihood, self.transition_params = crf_log_likelihood(self.scores, labels, self.sequence_lengths)
                losses = -log_likelihood
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=labels)
                mask = tf.sequence_mask(self.sequence_lengths, name="padding_mask")
                losses = tf.boolean_mask(losses, mask, name="remove_padding")
            with tf.name_scope('total'):
                self.loss = tf.reduce_mean(losses)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
            self.train_step = optimizer.apply_gradients(zip(gradients, variables))

    def test(self):
        inputs = self.embedding_layer()
        self.inference_layer(inputs)
        self.saver = tf.train.Saver()

    def train(self):
        inputs = self.embedding_layer()
        self.inference_layer(inputs)
        self.add_train_ops()
        self.saver = tf.train.Saver()


def deep_bidirectional_dynamic_rnn(cells, inputs, sequence_length, dtype=None):
    def _reverse(input_, seq_lengths, seq_dim=1, batch_dimension=0):
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dimension)

    outputs, state = None, None
    with vs.variable_scope("dblstm"):
        for i, cell in enumerate(cells):
            if i % 2 == 1:
                with vs.variable_scope("dblstm-bw-%s" % (i / 2)) as bw_scope:
                    inputs_reverse = _reverse(inputs, seq_lengths=sequence_length)
                    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs_reverse,
                                                       sequence_length=sequence_length,
                                                       initial_state=None, dtype=dtype, scope=bw_scope)
                    outputs = _reverse(outputs, seq_lengths=sequence_length)
            else:
                with vs.variable_scope("dblstm-fw-%s" % (i / 2)) as fw_scope:
                    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs,
                                                       sequence_length=sequence_length,
                                                       initial_state=None, dtype=dtype,
                                                       scope=fw_scope)
            inputs = outputs
    return outputs, state


class HighwayLSTMCell(LSTMCell):
    def __init__(self, num_units, input_size=None, initializer=None, forget_bias=1.0,
                 activation=tf.nn.tanh, reuse=None):
        super(HighwayLSTMCell, self).__init__(num_units=num_units, input_size=input_size, cell_clip=None,
                                              initializer=initializer, forget_bias=forget_bias, activation=activation,
                                              reuse=reuse)

    def __call__(self, inputs, state, scope=None):
        (c_prev, m_prev) = state

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with _checked_scope(self, scope or "highway_lstm_cell", initializer=self._initializer, reuse=self._reuse):
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate, r = transform_gate
            with vs.variable_scope('hidden_weights'):
                hidden_matrix = linear_block_initialization(m_prev, 5 * [self._num_units], bias=False)
            ih, jh, fh, oh, rh = array_ops.split(value=hidden_matrix, num_or_size_splits=5, axis=1)
            with vs.variable_scope('input_weights'):
                input_matrix = linear_block_initialization(inputs, 6 * [self._num_units], bias=True)
            ix, jx, fx, ox, rx, hx = array_ops.split(value=input_matrix, num_or_size_splits=6, axis=1)

            i = sigmoid(ih + ix)
            o = sigmoid(oh + ox)
            f = sigmoid(fh + fx + self._forget_bias)
            j = self._activation(jh + jx)
            c = f * c_prev + i * j
            t = sigmoid(rh + rx)
            _m = o * self._activation(c)
            m = t * _m + (1 - t) * hx

        new_state = (LSTMStateTuple(c, m))
        return m, new_state


def linear_block_initialization(args, output_sizes, bias, bias_start=0.0):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = tf.concat([vs.get_variable("weights_{}".format(i), [total_arg_size, size],
                                             dtype=dtype) for i, size in enumerate(output_sizes)], axis=1)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = vs.get_variable(
                "biases", [sum(output_sizes)],
                dtype=dtype,
                initializer=tf.constant_initializer(bias_start, dtype=dtype))
        return tf.nn.bias_add(res, biases)
