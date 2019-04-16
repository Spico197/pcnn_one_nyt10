import tensorflow as tf
import numpy as np
from config import config
from data_loader_opennre import json_file_data_loader as data_loader

def word_position_embedding(word, word_vec_mat, pos1, pos2):
    with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
        with tf.name_scope('word_embedding'):
            word_vec_weights = tf.get_variable('word_vec_weights', initializer=word_vec_mat, dtype=tf.float32)
            word_vec_weights = tf.concat([word_vec_mat, 
                                        tf.get_variable('unk_vec', initializer=tf.contrib.layers.xavier_initializer(), shape=(1, config.WORD_EMBEDDING_DIM), dtype=tf.float32),
                                        tf.constant(0, shape=(1, config.WORD_EMBEDDING_DIM), dtype=tf.float32)], 0)
            wd_embedding = tf.nn.embedding_lookup(word_vec_weights, word)
        with tf.name_scope('pos_embedding'):
            pos1_weights = tf.get_variable('pos1_weights', initializer=tf.contrib.layers.xavier_initializer(), shape=(config.MAX_SENTENCE_LENGTH*2, config.POS_EMBEDDING_DIM), dtype=tf.float32)
            pos2_weights = tf.get_variable('pos2_weights', initializer=tf.contrib.layers.xavier_initializer(), shape=(config.MAX_SENTENCE_LENGTH*2, config.POS_EMBEDDING_DIM), dtype=tf.float32)
            pos1_embedding = tf.nn.embedding_lookup(pos1_weights, pos1)
            pos2_embedding = tf.nn.embedding_lookup(pos2_weights, pos2)
            pos_embedding = tf.concat([pos1_embedding, pos2_embedding], -1, name='pos_embedding_concat')
        result = tf.concat([wd_embedding, pos_embedding], -1, name='word_pos_embedding_concat')
        return result


def pcnn_encoder(embedding, pos1, pos2):
    with tf.variable_scope('pcnn', reuse=tf.AUTO_REUSE):
        # after cnn: (config.HIDDEN_SIZE, config.MAX_SENTENCE_LENGTH)
        with tf.name_scope('conv1d'):
            after_cnn = tf.layers.conv1d(inputs=embedding, 
                                        filters=config.HIDDEN_SIZE,
                                        kernel_size=config.KERNEL_SIZE,
                                        strides=config.STRIDE_SIZE,
                                        padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        # embedding.shape -> (?, 120, 60)
        # pos1.shape -> (?, 120)
        # pos2.shape -> (?, 120)
        # after_cnn.shape -> (?, 120, 230)
        # after_cnn[0, 0, 0:23].shape -> (23)
        # print(after_cnn[0, 1:120, 0].shape)
        # print(tf.reduce_max(after_cnn[0, 1:120, 0], -1))
        # return tf.reduce_max(after_cnn[0, 1:120, 0], -1)
        with tf.name_scope('piecewise_pooling'):
            after_pooling_list = []
            for batch_num in range(config.BATCH_SIZE):
                seg_max = []
                head_pos = config.MAX_SENTENCE_LENGTH - pos1[batch_num, 0]
                tail_pos = config.MAX_SENTENCE_LENGTH - pos2[batch_num, 0]
                for i in range(config.HIDDEN_SIZE):
                    # seg1 = after_cnn[batch_num, 0:head_pos, i]
                    # seg2 = after_cnn[batch_num, head_pos:tail_pos, i]
                    # seg3 = after_cnn[batch_num, tail_pos:config.MAX_SENTENCE_LENGTH, i]
                    seg1, seg2, seg3 = tf.split(after_cnn, [head_pos, tail_pos-head_pos, config.MAX_SENTENCE_LENGTH-tail_pos], -2)
                    seg_max.append([tf.reduce_max(seg1),
                                    tf.reduce_max(seg2),
                                    tf.reduce_max(seg3)])
                after_pooling_list.append(seg_max)
                after_pooling = tf.reshape(tf.stack(after_pooling_list), (-1, config.HIDDEN_SIZE*3))
        result = tf.nn.relu(after_pooling, name='activation')
        return result


def pcnn_encoder_fast(embedding, mask, dropout_rate=0.5):
    with tf.variable_scope('pcnn_fast', reuse=tf.AUTO_REUSE):
        # after cnn: (config.HIDDEN_SIZE, config.MAX_SENTENCE_LENGTH)
            # after_cnn = tf.layers.conv1d(inputs=embedding, 
            #                             filters=config.HIDDEN_SIZE,
            #                             kernel_size=config.KERNEL_SIZE,
            #                             strides=config.STRIDE_SIZE,
            #                             padding='same',
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        filters = tf.get_variable('filters', shape=(config.KERNEL_SIZE, 
                                                    config.WORD_EMBEDDING_DIM+2*config.POS_EMBEDDING_DIM,
                                                    config.HIDDEN_SIZE), 
                                  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('conv_filters', filters)
        after_cnn = tf.nn.conv1d(embedding,
                                 filters=filters,
                                 stride=1,
                                 padding='SAME',
                                 data_format='NWC')
        # reference from OpenNRE
        with tf.name_scope('piecewise_pooling_fast'):
            mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            mask = tf.nn.embedding_lookup(mask_embedding, mask)
            fast_encode = tf.reduce_max(tf.expand_dims(mask * 100, 2) + tf.expand_dims(after_cnn, 3), axis=1) - 100
        
        activate = tf.nn.relu(tf.reshape(fast_encode, [-1, config.HIDDEN_SIZE * 3]), name='activation')
        return tf.nn.dropout(activate, rate=dropout_rate, name='dropout')


def logit_output(inputs, rel_tot, add_softmax=True):
    with tf.variable_scope('logit-addSoftmax-{}'.format(add_softmax), reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable('w1', shape=(rel_tot, inputs.shape[1]), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('b', shape=(1, rel_tot), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        o = tf.matmul(inputs, tf.transpose(w1)) + bias
        if add_softmax:
            o = tf.nn.softmax(o)
        return o


def mil(inputs, scopes, rel_tot, bag_labels, is_training=True):
    # reference from OpenNRE (https://github.com/thunlp/OpenNRE)
    # inputs -> (batch_size, 3*hidden_size)
    if is_training:
        bag_repre = []
        for i in range(scopes.shape[0]):
            bag_mat = inputs[scopes[i][0]:scopes[i][1]] # (scope[i] length, 3*hidden_size)
            ins_logits = logit_output(bag_mat, rel_tot, add_softmax=True)
            j = tf.argmax(ins_logits[:, bag_labels[i]], output_type=tf.int32)
            bag_repre.append(bag_mat[j])
        bag_repre = tf.stack(bag_repre)
        pred_bag_logit = logit_output(bag_repre, rel_tot, add_softmax=False)
        return pred_bag_logit
    else:
        bag_logit = []
        for i in range(scope.shape[0]):
            bag_hidden_mat = x[scope[i][0]:scope[i][1]]
            instance_logit = logit_output(bag_hidden_mat, rel_tot, add_softmax=True) # (n', hidden_size) -> (n', rel_tot)
            bag_logit.append(tf.reduce_max(instance_logit, 0))
            bag_repre.append(bag_hidden_mat[0]) # fake max repre
        bag_logit = tf.stack(bag_logit)
        return bag_logit


def softmax_cross_entropy(x, bag_label, rel_tot, weights_table=None):
    with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
        if weights_table is None:
            weights = 1.0
        else:
            weights = tf.nn.embedding_lookup(weights_table, bag_label)
        label_onehot = tf.one_hot(indices=bag_label, depth=rel_tot, dtype=tf.int32)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=x, weights=weights)
        return loss


def get_weights_table(rel_tot, bag_labels):
    with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
        print("Calculating weights_table...")
        _weights_table = np.zeros((rel_tot), dtype=np.float32)
        for i in range(len(bag_labels)):
            _weights_table[bag_labels[i]] += 1.0 
        _weights_table = 1 / (_weights_table ** 0.05 + 1e-6)
        weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False, initializer=_weights_table)
        print("Finish calculating")
    return weights_table


train_data_loader = data_loader('./data/nyt/train.json', './data/nyt/word_vec.json', 
                                './data/nyt/rel2id.json', mode=data_loader.MODE_RELFACT_BAG,
                                shuffle=True, max_length=config.MAX_SENTENCE_LENGTH,
                                case_sensitive=False, batch_size=config.BATCH_SIZE)
word = tf.placeholder(tf.int32, shape=(None, config.MAX_SENTENCE_LENGTH), name='word')
pos1 = tf.placeholder(tf.int32, shape=(None, config.MAX_SENTENCE_LENGTH), name='pos1')
pos2 = tf.placeholder(tf.int32, shape=(None, config.MAX_SENTENCE_LENGTH), name='pos2')
bag_labels = tf.placeholder(tf.int32, shape=(config.BATCH_SIZE), name='bag_labels')
scope = tf.placeholder(tf.int32, shape=(config.BATCH_SIZE, 2), name='scope')
mask = tf.placeholder(tf.int32, shape=(None, config.MAX_SENTENCE_LENGTH), name='mask')


rel_tot = train_data_loader.rel_tot
word_vec = train_data_loader.word_vec_mat

with tf.name_scope('word_pos_embedding'):
    after_embedding = word_position_embedding(word, word_vec, pos1, pos2)
with tf.name_scope('pcnn'):
    after_pcnn = pcnn_encoder_fast(after_embedding, mask)
with tf.name_scope('mil'):
    bag_logit = mil(after_pcnn, scope, rel_tot, bag_labels)
    tf.summary.histogram('bag_logit', bag_logit)
with tf.name_scope('softmax_cross_entropy'):
    loss = softmax_cross_entropy(bag_logit, bag_labels, rel_tot, weights_table=get_weights_table(rel_tot, train_data_loader.data_rel))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train_step'):
    train_step = tf.train.AdadeltaOptimizer(**config.OPTIMIZER_PARAM).minimize(loss)
with tf.name_scope('accuracy'):
    with tf.name_scope('tot_acc'):
        tot_correct_count = tf.equal(tf.argmax(bag_logit), tf.argmax(bag_labels))
        tot_acc = tf.reduce_mean(tf.cast(tot_correct_count, tf.float32))
        tf.summary.scalar('tot_accuracy', tot_acc)
    with tf.name_scope('not_na_acc'):
        not_na_count_boolean = tf.logical_and(tf.equal(tf.argmax(bag_logit), tf.argmax(bag_labels)), tf.not_equal(tf.argmax(bag_labels), 0))
        not_na_acc = tf.reduce_mean(tf.cast(not_na_count_boolean, tf.int32))
        tf.summary.scalar('not_na_acc', not_na_acc)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./summary/pcnn_one_train_demo/1', sess.graph)
    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    print('Start training')
    for epoch in range(1, config.TRAIN_EPOCHS+1, 1):
        step = 1
        while True:
            try:
                batch_data = train_data_loader.next_batch(config.BATCH_SIZE)
                feed = {
                    word: batch_data['word'],
                    pos1: batch_data['pos1'],
                    pos2: batch_data['pos2'],
                    bag_labels: batch_data['rel'],
                    scope: batch_data['scope'],
                    mask: batch_data['mask']
                }
                tot_accuracy = sess.run(tot_acc, feed_dict=feed)
                not_na_accuracy = sess.run(not_na_acc, feed_dict=feed)
                print(' epoch: %2d, step: %8d, tot_acc: %.8f, not-NA_acc:%.8f\r' % \
                        (epoch, step, tot_accuracy, not_na_accuracy), end='')
                s = sess.run(merged_summary, feed_dict=feed)
                writer.add_summary(s, step)
                sess.run(train_step, feed_dict=feed)
            except StopIteration:
                break
            finally:
                step += 1
        saver.save(sess, './checkpoint/')
