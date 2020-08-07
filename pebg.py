"""
    bipartite graph node embedding --> item embedding, skill embedding
       (item is question.)

    plus: item difficutly features

    three different feature use PNN to fuse, and with the help of auxilary target
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
import math
from scipy import sparse
from PNN import pnn1


# load data

data_folder = 'ednet'
if data_folder == 'assist09':
    con_sym = '-'
elif data_folder == 'ednet':
    con_sym = ';'
elif data_folder == 'assist12':
    con_sym = '$$$'
else:
    print('no such dataset!')
    exit()

saved_model_folder = os.path.join(data_folder, 'pebg_model')
if not os.path.exists(saved_model_folder):
    os.mkdir(saved_model_folder)


pro_skill_coo = sparse.load_npz(os.path.join(data_folder, 'pro_skill_sparse.npz'))
skill_skill_coo = sparse.load_npz(os.path.join(data_folder, 'skill_skill_sparse.npz'))
pro_pro_coo = sparse.load_npz(os.path.join(data_folder, 'pro_pro_sparse.npz'))
[pro_num, skill_num] = pro_skill_coo.shape
print('problem number %d, skill number %d' % (pro_num, skill_num))
print('pro-skill edge %d, pro-pro edge %d, skill-skill edge %d' % (pro_skill_coo.nnz, pro_pro_coo.nnz, skill_skill_coo.nnz))

pro_skill_dense = pro_skill_coo.toarray()
pro_pro_dense = pro_pro_coo.toarray()
skill_skill_dense = skill_skill_coo.toarray()

pro_feat = np.load(os.path.join(data_folder, 'pro_feat.npz'))['pro_feat']    # [pro_diff_feat, auxiliary_target]
print('problem feature shape', pro_feat.shape)
print(pro_feat[:,0].min(),pro_feat[:,0].max())
print(pro_feat[:,1].min(),pro_feat[:,1].max())

diff_feat_dim = pro_feat.shape[1]-1
embed_dim = 64      # node embedding dim in bipartite
hidden_dim = 128    # hidden dim in PNN
keep_prob = 0.5
lr = 0.001
bs = 256
epochs = 200
model_flag = 0

tf_pro = tf.placeholder(tf.int32, [None])
tf_diff_feat = tf.placeholder(tf.float32, [None, diff_feat_dim])
tf_pro_skill_targets = tf.placeholder(tf.float32, [None, skill_num], name='tf_pro_skill')
tf_pro_pro_targets = tf.placeholder(tf.float32, [None, pro_num], name='tf_pro_pro')
tf_skill_skill_targets = tf.placeholder(tf.float32, [skill_num, skill_num], name='tf_skill_skill')
tf_auxiliary_targets = tf.placeholder(tf.float32, [None], name='tf_auxilary_targets')
tf_keep_prob = tf.placeholder(tf.float32, [1], name='tf_keep_prob')


with tf.variable_scope('pro_skill_embed', reuse=tf.AUTO_REUSE):
    pro_embedding_matrix = tf.get_variable('pro_embed_matrix', [pro_num, embed_dim], 
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
    skill_embedding_matrix = tf.get_variable('skill_embed_matrix', [skill_num, embed_dim], 
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
    diff_embedding_matrix = tf.get_variable('diff_embed_matrix', [diff_feat_dim, embed_dim], 
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))

pro_embed = tf.nn.embedding_lookup(pro_embedding_matrix, tf_pro)         # [bs, embed_dim]
diff_feat_embed = tf.matmul(tf_diff_feat, diff_embedding_matrix)        # [bs, embed_dim] 

# pro-skill
pro_skill_logits = tf.reshape(tf.matmul(pro_embed, tf.transpose(skill_embedding_matrix)), [-1])
tf_pro_skill_targets_reshape = tf.reshape(tf_pro_skill_targets, [-1])
cross_entropy_pro_skill = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_pro_skill_targets_reshape,
                                        logits=pro_skill_logits))

# pro-pro
pro_pro_logits = tf.reshape(tf.matmul(pro_embed, tf.transpose(pro_embedding_matrix)), [-1])
tf_pro_pro_targets_reshape = tf.reshape(tf_pro_pro_targets, [-1])
cross_entropy_pro_pro = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_pro_pro_targets_reshape,
                                        logits=pro_pro_logits))

# skill-skill
skill_skill_logits = tf.reshape(tf.matmul(skill_embedding_matrix, tf.transpose(skill_embedding_matrix)), [-1])
tf_skill_skill_targets_reshape = tf.reshape(tf_skill_skill_targets, [-1])
cross_entropy_skill_skill = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_skill_skill_targets_reshape,
                                logits=skill_skill_logits))

# feature fuse
skill_embed = tf.matmul(tf_pro_skill_targets, skill_embedding_matrix) / tf.reduce_sum(tf_pro_skill_targets, axis=1, keepdims=True)
pro_final_embed, p = pnn1([pro_embed, skill_embed, diff_feat_embed], embed_dim, hidden_dim, tf_keep_prob[0])
mse = tf.reduce_mean(tf.square(p-tf_auxiliary_targets))

# optimizer
loss = mse + cross_entropy_pro_skill + cross_entropy_pro_pro + cross_entropy_skill_skill
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

print('finish building graph')
#---------------


# begin train
saver = tf.train.Saver()
train_steps = int(math.ceil(pro_num/float(bs)))
with tf.Session() as sess:

    if model_flag == 0:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, os.path.join(saved_model_folder, 'pebg_%d.ckpt'%model_flag))
        
    for i in range(model_flag, epochs):
        train_loss = 0
        for m in range(train_steps):
            b, e = m*bs, min((m+1)*bs, pro_num)
            
            batch_pro = np.arange(b, e).astype(np.int32)
            batch_pro_skill_targets = pro_skill_dense[b:e, :]
            batch_pro_pro_targets = pro_pro_dense[b:e, :]
            batch_diff_feat = pro_feat[b:e, :-1]
            batch_auxiliary_targets = pro_feat[b:e, -1]

            feed_dict = {tf_pro:batch_pro,
                        tf_diff_feat:batch_diff_feat,
                        tf_auxiliary_targets:batch_auxiliary_targets,
                        tf_pro_skill_targets:batch_pro_skill_targets,
                        tf_pro_pro_targets:batch_pro_pro_targets,
                        tf_skill_skill_targets:skill_skill_dense,
                        tf_keep_prob:[keep_prob]}

            _, loss_ = sess.run([train_op, loss], feed_dict=feed_dict)
            train_loss += loss_
        
        train_loss /= train_steps
        print("epoch %d, loss %.4f" % (i, train_loss))
        
        if i+1 in [50, 100, 200, 500, 1000, 1500, 2000]: 
            saver.save(sess, os.path.join(saved_model_folder, 'pebg_%d.ckpt'%(i+1)))

    print('finish training')
    #---------------

    # store pretrained pro skill embedding 
    pro_repre, skill_repre = sess.run([pro_embedding_matrix, skill_embedding_matrix])
    print(pro_repre.shape, skill_repre.shape)
  
    feed_dict = {tf_pro:np.arange(pro_num).astype(np.int32),
                tf_diff_feat:pro_feat[:, :-1],
                tf_auxiliary_targets:pro_feat[:, -1],
                tf_pro_skill_targets:pro_skill_dense,
                tf_pro_pro_targets:pro_pro_dense,
                tf_skill_skill_targets:skill_skill_dense,
                tf_keep_prob:[1.]}
    pro_final_repre = sess.run(pro_final_embed, feed_dict=feed_dict)
    print(pro_final_repre.shape)

    with open(os.path.join(data_folder, 'skill_id_dict.txt'), 'r') as f:
        skill_id_dict = eval(f.read()) 
    join_skill_num = len(skill_id_dict)
    print('original skill number %d, joint skill number %d' % (skill_num, join_skill_num))

    skill_repre_new = np.zeros([join_skill_num, skill_repre.shape[1]])
    skill_repre_new[:skill_num, :] = skill_repre
    for s in skill_id_dict.keys():
        if con_sym in str(s):
            tmp_skill_id = skill_id_dict[s]
            tmp_skills = [skill_id_dict[ele] for ele in s.split(con_sym)]
            skill_repre_new[tmp_skill_id, :] = np.mean(skill_repre[tmp_skills], axis=0)

    np.savez(os.path.join(data_folder, 'embedding_%d.npz'%epochs), 
                pro_repre=pro_repre, skill_repre=skill_repre_new, pro_final_repre=pro_final_repre)
