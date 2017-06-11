import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple
# noinspection PyProtectedMember
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope
# noinspection PyProtectedMember
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid

NUM_BIDIRECTIONAL_LAYERS = 4


def build_graph(vocab_size,
                emb_dim,
                marker_emb_dim,
                state_dim,
                num_classes):
    # Placeholders
    seq_lens = tf.placeholder(tf.int32, [None], name="sequence_lengths")
    keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    # Embedding layer
    with tf.name_scope('embedding_layer'):
        word_indices = tf.placeholder(tf.int32, [None, None], "word_indices")  # [batch_size, num_steps]
        predicate_indices = tf.placeholder(tf.int32, [None, None], name="predicate_markers")  # [batch_size, num_steps]
        word_embedding = tf.nn.embedding_lookup(
            tf.get_variable('word_embedding_matrix', [vocab_size, emb_dim]), word_indices, name="word_embedding")

        predicate_embedding = tf.nn.embedding_lookup(
            tf.get_variable('predicate_embedding_matrix', [2, marker_emb_dim]), predicate_indices,
            name="predicate_marker_embedding")
        # concatenate on embedding dim
        inputs = tf.concat([word_embedding, predicate_embedding], 2, name="concatenated_inputs")

    def dblstm_cell():
        return DropoutWrapper(HighwayLSTMCell(state_dim, initializer=tf.orthogonal_initializer()),
                              variational_recurrent=True, dtype=tf.float32, output_keep_prob=keep_prob)

    with tf.name_scope('deep_bidirectional_rnn'):
        rnn_outputs, _ = deep_bidirectional_dynamic_rnn([dblstm_cell() for _ in range(NUM_BIDIRECTIONAL_LAYERS)],
                                                        inputs, sequence_length=seq_lens, dtype=tf.float32)

    with tf.name_scope('linear_projection'):
        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable('W', [state_dim, num_classes],
                                        initializer=tf.random_normal_initializer(0, 0.01))
            softmax_bias = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

        time_steps = tf.shape(rnn_outputs)[1]
        rnn_outputs = tf.reshape(rnn_outputs, [-1, state_dim], name="flatten_rnn_outputs_for_linear_projection")
        logits = tf.nn.xw_plus_b(x=rnn_outputs, weights=softmax_w, biases=softmax_bias, name="softmax_projection")
        logits = tf.reshape(logits, [-1, time_steps, num_classes], name="unflatten_logits")

    with tf.name_scope('cross_entropy'):
        labels = tf.placeholder(tf.int32, [None, None], name="labels")
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.sequence_mask(seq_lens, name="padding_mask")
        losses = tf.boolean_mask(losses, mask, name="remove_padding")
        with tf.name_scope('total'):
            loss = tf.reduce_mean(losses)

    with tf.name_scope('train'):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1, epsilon=1e-6)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        train_step = optimizer.apply_gradients(zip(gradients, variables))

    preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32, "predictions")

    return {
        'words': word_indices,
        'markers': predicate_indices,
        'lengths': seq_lens,
        'labels': labels,
        'loss': loss,
        'train': train_step,
        'predictions': preds,
        'keep_prob': keep_prob,
        'logits': logits
    }


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
                hidden_matrix = _linear(m_prev, 5 * self._num_units, bias=False)
            ih, jh, fh, oh, rh = array_ops.split(value=hidden_matrix, num_or_size_splits=5, axis=1)
            with vs.variable_scope('input_weights'):
                input_matrix = _linear(inputs, 6 * self._num_units, bias=True)
            ix, jx, fx, ox, rx, hx = array_ops.split(value=input_matrix, num_or_size_splits=6, axis=1)

            i, j, f, o, r = ih + ix, jh + jx, fh + fx, oh + ox, rh + rx

            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j))
            r = sigmoid(r)
            m = r * sigmoid(o) * self._activation(c) + (1 - r) * hx

        new_state = (LSTMStateTuple(c, m))
        return m, new_state
