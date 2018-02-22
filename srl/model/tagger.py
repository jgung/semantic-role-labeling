import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.python.ops.rnn_cell import DropoutWrapper, LSTMCell

from srl.common.constants import KEEP_PROB_KEY, LABEL_KEY, LENGTH_KEY
from srl.model.layers import HighwayLSTMCell, deep_bidirectional_dynamic_rnn


class DBLSTMTagger(object):
    def __init__(self, features, num_layers, state_dim, num_classes, transition_params=None, crf=True, dblstm=True,
                 orthonormal_init=True, recurrent_dropout=True, highway=True):
        super(DBLSTMTagger, self).__init__()
        self.features = features

        self.num_layers = num_layers
        self.state_dim = state_dim
        self.num_classes = num_classes
        self.transition_params = transition_params
        self.crf = crf
        self.dblstm = dblstm
        self.orthonormal_init = orthonormal_init
        self.recurrent_dropout = recurrent_dropout
        self.highway = highway

        self._embedding_placeholder = {}
        self._embedding_init = {}

        self.scores = None
        self.loss = None
        self.train_step = None
        self.saver = None

        self.dropout_keys = []
        self.feed_dict = {}
        self.dropout_keep_prob = self._add_placeholder(KEEP_PROB_KEY, tf.float32, dropout=True)
        self.sequence_lengths = self._add_placeholder(LENGTH_KEY, tf.int32, [None])
        self.global_step = tf.Variable(0, trainable=False)
        self.global_step_increment = self.global_step.assign_add(1)

    def transition_matrix(self):
        if self.crf:
            return self.transition_params.eval()
        return self.transition_params

    def _add_placeholder(self, name, dtype, shape=None, dropout=False):
        placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)
        self.feed_dict[name] = placeholder
        if dropout:
            self.dropout_keys.append(name)
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
                shape = [None] * feature.rank
                indices = self._add_placeholder(name=feature.name, dtype=tf.int32, shape=shape)
                embedding = tf.nn.embedding_lookup(params=embedding_matrix, ids=indices, name='{}_embedding'.format(feature.name))

                result = embedding
                if feature.rank == 3:
                    with tf.variable_scope('{}_ops'.format(feature.name)):
                        result = feature.function.apply(embedding)

                # if feature.rank == 4:
                #     time_major_embedding = tf.transpose(embedding, (1, 0, 2, 3, 4))
                #     with tf.variable_scope("{}_ops".format(feature.name), reuse=tf.AUTO_REUSE):
                #         results = tf.map_fn(lambda x: feature.function.apply(x), time_major_embedding)
                #     result = tf.transpose(tf.stack(results), (1, 0, 2, 3))
                #     with tf.variable_scope("{}_combine".format(feature.name)):
                #         result = feature.function.apply(result)

                if feature.keep_prob < 1:
                    keep_prob_placeholder = self._add_placeholder(feature.name + KEEP_PROB_KEY, tf.float32, dropout=True)
                    result = tf.nn.dropout(result, keep_prob=keep_prob_placeholder)
                inputs.append(result)
            return tf.concat(inputs, 2, name="concatenated_inputs")

    def _dblstm_cell(self):
        if self.orthonormal_init:
            cell = HighwayLSTMCell(self.state_dim, highway=self.highway, initializer=tf.orthogonal_initializer(),
                                   separate_init=True)
        else:
            cell = HighwayLSTMCell(self.state_dim, highway=self.highway, separate_init=False)
        return DropoutWrapper(cell, variational_recurrent=self.recurrent_dropout, dtype=tf.float32,
                              output_keep_prob=self.dropout_keep_prob)

    def inference_layer(self, inputs):
        if self.dblstm:
            with tf.name_scope('deep_bidirectional_rnn'):
                rnn_outputs, _ = deep_bidirectional_dynamic_rnn([self._dblstm_cell() for _ in range(self.num_layers)], inputs,
                                                                sequence_length=self.sequence_lengths)
            state_dim = self.state_dim
        else:
            cell_fw = DropoutWrapper(LSTMCell(num_units=self.state_dim), variational_recurrent=True,
                                     state_keep_prob=self.dropout_keep_prob,
                                     output_keep_prob=self.dropout_keep_prob, dtype=tf.float32)
            cell_bw = DropoutWrapper(LSTMCell(num_units=self.state_dim), variational_recurrent=True,
                                     state_keep_prob=self.dropout_keep_prob,
                                     output_keep_prob=self.dropout_keep_prob, dtype=tf.float32)

            with tf.name_scope('bidirectional_rnn'):
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=self.sequence_lengths,
                                                                 dtype=tf.float32)
                rnn_outputs = tf.concat(rnn_outputs, 2)
                state_dim = self.state_dim * 2

        with tf.name_scope('linear_projection'):
            softmax_weights = tf.get_variable('softmax_W', [state_dim, self.num_classes],
                                              initializer=tf.random_normal_initializer(0, 0.01))
            softmax_bias = tf.get_variable('softmax_b', [self.num_classes], initializer=tf.zeros_initializer)

            time_steps = tf.shape(rnn_outputs)[1]
            rnn_outputs = tf.reshape(rnn_outputs, [-1, state_dim], name="flatten_rnn_outputs_for_linear_projection")
            logits = tf.nn.xw_plus_b(x=rnn_outputs, weights=softmax_weights, biases=softmax_bias, name="softmax_projection")
            self.scores = tf.reshape(logits, [-1, time_steps, self.num_classes], name="unflatten_logits")

        if self.crf:
            self.transition_params = tf.get_variable("transitions", [self.num_classes, self.num_classes])

    def add_train_ops(self):
        with tf.name_scope('loss_ops'):
            labels = self._add_placeholder(LABEL_KEY, tf.int32, [None, None])
            if self.crf:
                log_likelihood, self.transition_params = crf_log_likelihood(self.scores, labels, self.sequence_lengths,
                                                                            self.transition_params)
                losses = -log_likelihood
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=labels)
                mask = tf.sequence_mask(self.sequence_lengths, name="padding_mask")
                losses = tf.boolean_mask(losses, mask, name="remove_padding")
            with tf.name_scope('total'):
                self.loss = tf.reduce_mean(losses)

        with tf.name_scope('train'):
            self.train_step = self.training_op()

    def training_op(self):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1, epsilon=1e-6)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        return optimizer.apply_gradients(zip(gradients, variables))

    def test(self):
        inputs = self.embedding_layer()
        self.inference_layer(inputs)
        self.saver = tf.train.Saver(max_to_keep=2)

    def train(self):
        self.test()
        self.add_train_ops()
        self.saver = tf.train.Saver(max_to_keep=2)
