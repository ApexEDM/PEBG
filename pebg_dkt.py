import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from sklearn import metrics
from scipy import sparse
from tqdm import tqdm
import time, math


def train_test_split(data, split=0.8):
    n_samples = data[0].shape[0]
    split_point = int(n_samples*split)
    train_data, test_data = [], []
    for d in data:
        train_data.append(d[:split_point])
        test_data.append(d[split_point:])
    return train_data, test_data


# data
data_folder = "assist09"
data = np.load(os.path.join(data_folder, data_folder+'.npz'))
y, skill, problem, real_len = data['y'], data['skill'], data['problem'], data['real_len']
skill_num, pro_num = data['skill_num'], data['problem_num']
print('problem number %d, skill number %d' % (pro_num, skill_num))

# divide train test set
train_data, test_data = train_test_split([y, skill, problem, real_len])   # [y, skill, pro, real_len]
train_y, train_skill, train_problem, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
test_y, test_skill, test_problem, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]

# embed data, used for initialize
embed_data = np.load(os.path.join(data_folder, 'embedding_200.npz'))
_, _, pre_pro_embed = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
print(pre_pro_embed.shape, pre_pro_embed.dtype)


# hyper-params
epochs = 200
bs = 128
embed_dim = pre_pro_embed.shape[1]
hidden_dim = 128
lr = 0.001
use_pretrain = True
train_embed = False
train_flag = True


# build tensorflow graph 
# tf_skill = tf.placeholder(tf.int32, [None, None], name='tf_skill')           # [bs, max_len]
tf_y = tf.placeholder(tf.float32, [None, None], name='tf_y')
tf_pro = tf.placeholder(tf.int32, [None, None], name='tf_pro')
tf_real_seq_len = tf.placeholder(tf.int32, [None], name='tf_real_seq_len')   # [bs]

tf_batch_size = tf.shape(tf_y)[0]
tf_max_len = tf.shape(tf_y)[1]


with tf.variable_scope('pro_skill_embed', reuse=tf.AUTO_REUSE):
    if use_pretrain:
        pro_embed_init = tf.constant_initializer(pre_pro_embed)
        # skill_embed_init = tf.constant_initializer(pre_skill_embed)
        print("use pretrain embedding matrix")
    else:    
        pro_embed_init = tf.truncated_normal_initializer(stddev=0.1)
        # skill_embed_init = tf.truncated_normal_initializer(stddev=0.1)
        print("use random init embedding matrix")
    
    pro_embedding_matrix = tf.get_variable('pro_embed_matrix', [pro_num, embed_dim], 
                                initializer=pro_embed_init, trainable=train_embed)
    # skill_embedding_matrix = tf.get_variable('skill_embed_matrix', [skill_num, embed_dim], 
    #                             initializer=skill_embed_init, trainable=train_flag)
    # deal with complement symbol
    zero_tensor = tf.zeros([1, embed_dim], dtype=tf.float32)
    pro_embedding_matrix = tf.concat([zero_tensor, pro_embedding_matrix], axis=0)
    # skill_embedding_matrix = tf.concat([zero_tensor, skill_embedding_matrix], axis=0)
    

# skill, problem embedding
# all_skill_embed = tf.nn.embedding_lookup(skill_embedding_matrix, tf_skill)  # [bs, max_len, embed_dim]
all_pro_embed = tf.nn.embedding_lookup(pro_embedding_matrix, tf_pro)  # [bs, max_len, embed_dim]
# skill_embed = all_skill_embed[:, :-1, :]
pro_embed = all_pro_embed[:, :-1, :]
# next_skill_embed = all_skill_embed[:, 1:, :]
next_pro_embed = all_pro_embed[:, 1:, :]         # [bs, max_len-1, embed_dim]

inputs_embed = pro_embed               # [bs, max_len-1, embed_dim]
next_inputs_embed = next_pro_embed
inputs_embed_dim = embed_dim
print("problem level prediction")

# concat zero on the skill embedding
def concat_zero(x):
    xx = x[:-1]
    yy = x[-1]
    zero_tensor = tf.zeros([embed_dim], dtype=tf.float32)
    o = tf.cond(tf.greater(yy, 0.), lambda:tf.concat([xx, zero_tensor], axis=0), lambda:tf.concat([zero_tensor, xx], axis=0))
    return o

inputs_y = tf.concat([tf.reshape(inputs_embed, [-1, inputs_embed_dim]), tf.reshape(tf_y[:, :-1], [-1, 1])], axis=1)
inputs_y_embedding = tf.map_fn(concat_zero, inputs_y) 
rnn_inputs = tf.reshape(inputs_y_embedding, [-1, tf_max_len-1, inputs_embed_dim+embed_dim])

# lstm
with tf.variable_scope('LstmNet', reuse=tf.AUTO_REUSE):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    init_state = lstm_cell.zero_state(tf_batch_size, tf.float32)
    outputs, last_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=rnn_inputs,
                                            sequence_length=tf_real_seq_len,
                                            initial_state=init_state,
                                            dtype=tf.float32)
    outputs_reshape = tf.reshape(outputs, [-1, hidden_dim])   # [bs*(max_len-1), hidden_dim]
    outputs_reshape = tf.concat([outputs_reshape, tf.reshape(next_inputs_embed, [-1, inputs_embed_dim])], axis=1)

    rnn_w = tf.get_variable('softmax_w', [hidden_dim+inputs_embed_dim, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    rnn_b = tf.get_variable('softmax_b', [1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    logits = tf.nn.xw_plus_b(outputs_reshape, rnn_w, rnn_b)  # [bs*(max_len-1), 1]    

# ignore the answer -1
tf_targets_flatten = tf.reshape(tf_y[:, 1:], [-1])
index = tf.where(tf.not_equal(tf_targets_flatten, tf.constant(-1, dtype=tf.float32)))
filtered_targets = tf.squeeze(tf.gather(tf_targets_flatten, index), axis=1)

# lstm-outputs
logits = tf.reshape(logits, [-1])
filtered_logits = tf.squeeze(tf.gather(logits, index), axis=1)
final_logits = filtered_logits

# cross entropy, optimize
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=filtered_targets,
                                            logits=final_logits))
filtered_predict_prob = tf.nn.sigmoid(final_logits)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)
saver = tf.train.Saver()

if train_flag:
    # begin train
    train_steps = int(math.ceil(train_skill.shape[0]/float(bs)))
    test_steps = int(math.ceil(test_skill.shape[0]/float(bs)))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  
        
        # random_pro_embed = sess.run(pro_embedding_matrix)
        # np.savez(os.path.join(data_folder, 'random_embed.npz'), 
        #             random_skill_embed=random_skill_embed, random_pro_embed=random_pro_embed)
        
        best_auc = best_acc = 0
        for i in tqdm(range(epochs)):
            train_loss = 0

            for j in range(train_steps):
                batch_y = train_y[j*bs:(j+1)*bs, :]
                batch_pro = train_problem[j*bs:(j+1)*bs, :]
                # batch_skill = train_skill[j*bs:(j+1)*bs, :]
                batch_real_len = train_real_len[j*bs:(j+1)*bs] - 1
                feed_dict = {tf_pro: batch_pro, 
                            tf_y:batch_y, tf_real_seq_len:batch_real_len}
                _, batch_loss = sess.run([train_op, loss], feed_dict=feed_dict)
                train_loss += batch_loss
            train_loss /= batch_loss

            test_preds, test_targets, test_loss = [], [], 0
            for j in range(test_steps):
                test_y_ = test_y[j*bs:(j+1)*bs, :]
                # test_skill_ = test_skill[j*bs:(j+1)*bs, :]
                test_pro_ = test_problem[j*bs:(j+1)*bs, :]
                test_real_len_ = test_real_len[j*bs:(j+1)*bs] - 1
                feed_dict = {tf_y:test_y_, 
                            tf_pro: test_pro_, tf_real_seq_len:test_real_len_}
                filtered_targets_, filtered_preds_ = sess.run([filtered_targets, filtered_predict_prob], feed_dict=feed_dict)
                test_preds.append(filtered_preds_)
                test_targets.append(filtered_targets_)
                
            test_preds = np.concatenate(test_preds, axis=0)
            test_targets = np.concatenate(test_targets, axis=0)

            test_auc = metrics.roc_auc_score(test_targets, test_preds)
            test_preds[test_preds>0.5] = 1.
            test_preds[test_preds<=0.5] = 0.
            test_acc = metrics.accuracy_score(test_targets, test_preds)
            
            records = 'Epoch %d/%d, train loss:%3.5f, test auc:%f, test acc:%3.5f' % \
                            (i+1, epochs, train_loss, test_auc, test_acc)    
            print(records)

            # if best_auc < test_auc:
            #     best_auc = test_auc
            #     saver.save(sess, os.path.join(data_folder, 'bekt_dkt_model/dkt_pretrain.ckpt'))
else:
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(data_folder, 'bekt_dkt_model/dkt_pretrain.ckpt'))
        
        #pro_embed, skill_embed 
        pro_embed_trained = sess.run(pro_embedding_matrix)
        pro_embed_trained = pro_embed_trained[1:]
        
        np.savez(os.path.join(data_folder, 'bekt_dkt_model/pro_embed_bekt_dkt.npz'), pro_final_repre=pro_embed_trained)
   