import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

PREDICATE_EMBEDDING_DIM = 10


def build_graph(vocab_size,
                emb_dim,
                state_dim,
                num_classes):
    # Placeholders
    word_indices = tf.placeholder(tf.int32, [None, None])  # [batch_size, num_steps]
    predicate_indices = tf.placeholder(tf.int32, [None, None])  # [batch_size, num_steps]

    seq_lens = tf.placeholder(tf.int32, [None])
    labels = tf.placeholder(tf.int32, [None, None])
    # keep_prob = tf.constant(1.0)

    # Embedding layer
    word_embedding = tf.nn.embedding_lookup(
        tf.get_variable('word_embedding_matrix', [vocab_size, emb_dim]), word_indices)

    predicate_embedding = tf.nn.embedding_lookup(
        tf.get_variable('predicate_embedding_matrix', [2, PREDICATE_EMBEDDING_DIM]), predicate_indices)

    inputs = tf.concat([word_embedding, predicate_embedding], 2, name="inputs")  # concatenate on embedding dim

    # RNN
    state_dim += PREDICATE_EMBEDDING_DIM
    cell = GRUCell(state_dim)
    # init_state = tf.get_variable('init_state', [1, state_dim], initializer=tf.constant_initializer(0.0))
    # init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=seq_lens, dtype=tf.float32)
    # Add dropout, as the model otherwise quickly overfits
    # rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

    # Softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_dim, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    ntime_steps = tf.shape(rnn_outputs)[1]
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_dim])

    logits = tf.matmul(rnn_outputs, W) + b
    logits = tf.reshape(logits, [-1, ntime_steps, num_classes])

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    # shape = (batch, sentence, nclasses)
    mask = tf.sequence_mask(seq_lens)
    # apply mask
    losses = tf.boolean_mask(losses, mask)
    loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

    return {
        'words': word_indices,
        'markers': predicate_indices,
        'lengths': seq_lens,
        'labels': labels,
        'loss': loss,
        'train': train_step,
        'predictions': preds,
    }
