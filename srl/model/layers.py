import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import base_layer, math_ops, init_ops, nn_ops, clip_ops, LayerRNNCell
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.python.util import nest


def deep_bidirectional_dynamic_rnn(cells, inputs, sequence_length):
    def _reverse(_input, seq_lengths):
        return array_ops.reverse_sequence(input=_input, seq_lengths=seq_lengths, seq_axis=1, batch_axis=0)

    outputs, state = None, None
    with vs.variable_scope("dblstm"):
        for i, cell in enumerate(cells):
            if i % 2 == 1:
                with vs.variable_scope("bw-%s" % (i // 2)) as bw_scope:
                    inputs_reverse = _reverse(inputs, seq_lengths=sequence_length)
                    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs_reverse, sequence_length=sequence_length,
                                                       dtype=tf.float32, scope=bw_scope)
                    outputs = _reverse(outputs, seq_lengths=sequence_length)
            else:
                with vs.variable_scope("fw-%s" % (i // 2)) as fw_scope:
                    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=sequence_length,
                                                       dtype=tf.float32, scope=fw_scope)
            inputs = outputs
    return outputs, state


def orthogonal_initializer(num_splits):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if num_splits == 1:
            return tf.orthogonal_initializer(seed=None, dtype=dtype).__call__(shape, dtype, partition_info)
        shape = (shape[0], (shape[1] // num_splits))
        matrices = []
        for i in range(num_splits):
            matrices.append(tf.orthogonal_initializer(seed=None, dtype=dtype).__call__(shape, dtype, partition_info))
        return tf.concat(axis=1, values=matrices)

    return _initializer


class HighwayLSTMCell(LayerRNNCell):
    def __init__(self, num_units,
                 highway=True,
                 cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 forget_bias=1.0,
                 activation=None, reuse=None, name=None):
        """Initialize the parameters for an LSTM cell with simplified highway connections as described in
        'Deep Semantic Role Labeling: What works and what's next' (He et al. 2017).

        Args:
          num_units: int, The number of units in the LSTM cell.
          highway: (optional) Python boolean describing whether to include highway connections
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices. Uses an orthonormal initializer if none is provided.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
            provided, then the projected values are clipped elementwise to within
            `[-proj_clip, proj_clip]`.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
        """
        super(HighwayLSTMCell, self).__init__(_reuse=reuse, name=name)
        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units

        self._highway = highway
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

        if num_proj:
            self._state_size = (LSTMStateTuple(num_units, num_proj))
            self._output_size = num_proj
        else:
            self._state_size = (LSTMStateTuple(num_units, num_units))
            self._output_size = num_units

        # initialized in self.build
        self._input_kernel = None
        self._hidden_kernel = None
        self._bias = None
        self._proj_kernel = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units if self._num_proj is None else self._num_proj

        tf.logging.info("Initializing (%d, %d) LSTM layer, default init (orthonormal): %s, highway: %s",
                        input_depth, h_depth, not self._initializer, self._highway)

        num_splits = self._highway and 6 or 4
        self._input_kernel = self.add_variable(
            "input_kernel",
            shape=[input_depth, num_splits * self._num_units],
            initializer=self._initializer if self._initializer else orthogonal_initializer(num_splits))
        self._bias = self.add_variable(
            "bias",
            shape=[num_splits * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))
        num_splits = self._highway and 5 or 4
        self._hidden_kernel = self.add_variable(
            "hidden_kernel",
            shape=[h_depth, num_splits * self._num_units],
            initializer=self._initializer if self._initializer else orthogonal_initializer(num_splits))
        if self._num_proj is not None:
            self._proj_kernel = self.add_variable(
                "projection/%s" % "kernel",
                shape=[self._num_units, self._num_proj],
                initializer=self._initializer if self._initializer else orthogonal_initializer(1))

        self.built = True

    # noinspection PyMethodOverriding
    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid

        (c_prev, m_prev) = state

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate, r = transform gate
        input_matrix = math_ops.matmul(inputs, self._input_kernel)
        input_matrix = nn_ops.bias_add(input_matrix, self._bias)

        hidden_matrix = math_ops.matmul(m_prev, self._hidden_kernel)

        if self._highway:
            ih, jh, fh, oh, rh = array_ops.split(value=hidden_matrix, num_or_size_splits=5, axis=1)
            ix, jx, fx, ox, rx, hx = array_ops.split(value=input_matrix, num_or_size_splits=6, axis=1)

            i = sigmoid(ih + ix)
            o = sigmoid(oh + ox)
            f = sigmoid(fh + fx + self._forget_bias)
            j = self._activation(jh + jx)
            c = f * c_prev + i * j
            if self._cell_clip is not None:
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

            t = sigmoid(rh + rx)
            _m = o * self._activation(c)
            m = t * _m + (1 - t) * hx

        else:
            ix, jx, fx, ox = array_ops.split(value=input_matrix, num_or_size_splits=4, axis=1)
            ih, jh, fh, oh = array_ops.split(value=hidden_matrix, num_or_size_splits=4, axis=1)

            i = sigmoid(ix + ih)
            o = sigmoid(ox + oh)
            f = sigmoid(fx + fh + self._forget_bias)
            c = i * self._activation(jx + jh) + f * c_prev

            if self._cell_clip is not None:
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

            m = o * self._activation(c)

        if self._num_proj is not None:
            m = math_ops.matmul(m, self._proj_kernel)
            if self._proj_clip is not None:
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)

        new_state = (LSTMStateTuple(c, m))
        return m, new_state


def linear_block_initialization(args, output_sizes, bias):
    if not nest.is_sequence(args):
        args = [args]
    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = tf.concat([vs.get_variable("weights_{}".format(i), [total_arg_size, size], dtype=dtype)
                             for i, size in enumerate(output_sizes)], axis=1)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = vs.get_variable("biases", [sum(output_sizes)], dtype=dtype, initializer=tf.zeros_initializer)
        return tf.nn.bias_add(res, biases)
