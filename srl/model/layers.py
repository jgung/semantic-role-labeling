import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.python.util import nest


def deep_bidirectional_dynamic_rnn(cells, inputs, sequence_length):
    def _reverse(_input, seq_lengths):
        return array_ops.reverse_sequence(input=_input, seq_lengths=seq_lengths, seq_dim=1, batch_dim=0)

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


class HighwayLSTMCell(LSTMCell):
    def __init__(self, num_units, highway=True, initializer=None):
        """
        Initialize an LSTM cell with highway connections as described in "Deep Semantic Role Labeling: What works and what's next"
        (He et al. 2017).
        :param highway: enable highway connections
        :param num_units: number of LSTM units in this cell
        :param initializer: weight initializer
        """
        super(HighwayLSTMCell, self).__init__(num_units=num_units, initializer=initializer)
        self.highway = highway

    def call(self, inputs, state):
        (c_prev, m_prev) = state

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with vs.variable_scope("highway_lstm_cell", initializer=self._initializer, reuse=self._reuse):
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate, r = transform_gate
            num_weights = self.highway and 5 or 4
            with vs.variable_scope('hidden_weights'):
                hidden_matrix = linear_block_initialization(m_prev, num_weights * [self._num_units], bias=False)

            num_weights = self.highway and 6 or 4
            with vs.variable_scope('input_weights'):
                input_matrix = linear_block_initialization(inputs, num_weights * [self._num_units], bias=True)

            if self.highway:
                ih, jh, fh, oh, rh = array_ops.split(value=hidden_matrix, num_or_size_splits=5, axis=1)
                ix, jx, fx, ox, rx, hx = array_ops.split(value=input_matrix, num_or_size_splits=6, axis=1)

                i = sigmoid(ih + ix)
                o = sigmoid(oh + ox)
                f = sigmoid(fh + fx + self._forget_bias)
                j = self._activation(jh + jx)
                c = f * c_prev + i * j
                t = sigmoid(rh + rx)
                _m = o * self._activation(c)
                m = t * _m + (1 - t) * hx
            else:
                ih, jh, fh, oh = array_ops.split(value=hidden_matrix, num_or_size_splits=4, axis=1)
                ix, jx, fx, ox = array_ops.split(value=input_matrix, num_or_size_splits=4, axis=1)

                i = sigmoid(ih + ix)
                o = sigmoid(oh + ox)
                f = sigmoid(fh + fx + self._forget_bias)
                c = i * self._activation(jh + jx) + f * c_prev
                m = o * self._activation(c)

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
