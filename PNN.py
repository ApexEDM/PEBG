import tensorflow as tf


def pnn1(inputs, embed_size, hidden_dim, keep_prob):
    num_inputs = len(inputs)
    num_pairs = int(num_inputs * (num_inputs-1) / 2)

    xw = tf.concat(inputs, 1)
    xw3d = tf.reshape(xw, [-1, num_inputs, embed_size])  # [batch_size, 3, embedding_size]
    
    row = []
    col = []
    for i in range(num_inputs-1):
        for j in range(i+1, num_inputs):
            row.append(i)
            col.append(j)      # row = [0 0 1]  col = [1 2 2]

    # batch * pair * k
    p = tf.transpose(
        # pair * batch * k
        tf.gather(
            # num * batch * k
            tf.transpose(
                xw3d, [1, 0, 2]),
            row),
        [1, 0, 2])
    # batch * pair * k
    q = tf.transpose(
        tf.gather(
            tf.transpose(
                xw3d, [1, 0, 2]),
            col),
        [1, 0, 2])
    p = tf.reshape(p, [-1, num_pairs, embed_size])
    q = tf.reshape(q, [-1, num_pairs, embed_size])
    ip = tf.reshape(tf.reduce_sum(p * q, [-1]), [-1, num_pairs])
    l = tf.concat([xw, ip], 1)

    h = tf.layers.dense(l, units=hidden_dim, activation=tf.nn.relu)
    h = tf.nn.dropout(h, keep_prob=keep_prob)
    p = tf.reshape(tf.layers.dense(h, units=1), [-1])
    return h, p