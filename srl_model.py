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

CHAR_FILTERS = 32


class DBLSTMTagger(object):
    def __init__(self, vocab_size, char_vocab_size, emb_dim, num_layers, marker_dim, char_dim, state_dim, num_classes,
                 char_len, char_conv=False):
        super(DBLSTMTagger, self).__init__()
        self.vocab_size = vocab_size
        self.char_vocab_size = char_vocab_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.marker_emb_dim = marker_dim
        self.char_emb_dim = char_dim
        self.char_conv = char_conv
        self.char_len = char_len

        self.state_dim = state_dim
        self.num_classes = num_classes

        self._embedding_placeholder = None
        self._embedding_init = None

        self.scores = None
        self.loss = None
        self.train_step = None
        self.saver = None

        self.feed_dict = {}
        self.dropout_keep_prob = self._add_placeholder("keep_prob", tf.float32)
        self.sequence_lengths = self._add_placeholder("lengths", tf.int32, [None])

    def _add_placeholder(self, name, dtype, shape=None):
        placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)
        self.feed_dict[name] = placeholder
        return placeholder

    def initialize_embeddings(self, sess, vectors):
        sess.run(self._embedding_init, feed_dict={self._embedding_placeholder: vectors})

    def embedding_layer(self):
        with tf.name_scope('embedding_layer'):
            word_embedding_matrix = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.emb_dim]),
                                                trainable=True, name="word_embedding_matrix")
            self._embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.emb_dim])
            self._embedding_init = word_embedding_matrix.assign(self._embedding_placeholder)

            word_indices = self._add_placeholder("words", tf.int32, [None, None])  # [batch_size, seq_len]
            word_embedding = tf.nn.embedding_lookup(word_embedding_matrix, word_indices, name="word_embedding")

            predicate_indices = self._add_placeholder("markers", tf.int32, [None, None])  # [batch_size, seq_len]
            predicate_embedding = tf.nn.embedding_lookup(
                tf.get_variable('predicate_embedding_matrix', [2, self.marker_emb_dim],
                                initializer=tf.random_normal_initializer(0, 0.01)), predicate_indices,
                name="predicate_marker_embedding")

            inputs = [word_embedding, predicate_embedding]
            if self.char_conv:
                char_indices = self._add_placeholder("chars", tf.int32, [None, None, None])
                char_embeddings = tf.nn.embedding_lookup(
                    tf.get_variable(name="char_embeddings", dtype=tf.float32,
                                    shape=[self.char_vocab_size, self.char_emb_dim],
                                    initializer=tf.random_normal_initializer(0, 0.01)),
                    char_indices, name="char_embedding")
                char_conv = get_cnn_step(inputs=char_embeddings, input_dim=self.char_emb_dim, seq_len=self.char_len)
                inputs.append(char_conv)

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
        with tf.name_scope('cross_entropy'):
            labels = self._add_placeholder("labels", tf.int32, [None, None])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=labels)
            mask = tf.sequence_mask(self.sequence_lengths, name="padding_mask")
            losses = tf.boolean_mask(losses, mask, name="remove_padding")
            with tf.name_scope('total'):
                self.loss = tf.reduce_mean(losses)

        with tf.name_scope('train'):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=1, epsilon=1e-6)
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


def get_cnn_step(inputs, input_dim, seq_len, window_size=2, num_filters=CHAR_FILTERS):
    shape = tf.shape(inputs)
    # flatten sequences for input
    inputs = tf.reshape(inputs, shape=[-1, shape[-2], shape[-1], 1])
    # convolution weights
    conv_filter = tf.get_variable("conv_w", [window_size, input_dim, 1, num_filters],
                                  initializer=tf.random_normal_initializer(0, 0.01))
    conv_bias = tf.get_variable("conv_b", [num_filters], initializer=tf.zeros_initializer)
    # convolution ops
    conv = tf.nn.conv2d(input=inputs, filter=conv_filter, strides=[1, 1, 1, 1], padding="VALID")
    relu = tf.nn.relu(tf.nn.bias_add(value=conv, bias=conv_bias))
    pool = tf.nn.max_pool(value=relu, ksize=[1, seq_len - window_size + 1, 1, 1], strides=[1, 1, 1, 1],
                          padding="VALID")
    # unflatten
    char_conv = tf.reshape(pool, shape=[-1, shape[1], CHAR_FILTERS])
    return char_conv


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
