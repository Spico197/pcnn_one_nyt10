import numpy as np
from keras.layers import Dropout, Softmax
from keras import backend as K


def __logit__(x, rel_tot, var_scope=None):
    relation_matrix = np.random.rand(rel_tot, x.shape[1]).astype(np.float32)
    bias = np.random.rand(rel_tot).astype(np.float32)

    relation_matrix = K.variable(relation_matrix, dtype='float32')
    bias = K.variable(bias, dtype='float32')
    logit = K.dot(x, K.transpose(relation_matrix)) + bias
    return logit

def bag_one(x, scope, bag_label, rel_tot, name=None, keep_prob=1.0):
    bag_repre = []
    for i in range(scope.shape[1]):
        # bag_hidden_mat = x[scope[i][0]:scope[i][1]]
        # TODO: x indices should be a int, but I have tried K.cast and still not solve the problem
        bag_hidden_mat = x[scope[i][0]:scope[i][1]]
        instance_logit = Softmax(axis=-1)(__logit__(bag_hidden_mat, rel_tot)) # (n', hidden_size) -> (n', rel_tot)
        j = K.argmax(instance_logit[:, bag_label[i]], output_type=np.int32)
        bag_repre.append(bag_hidden_mat[j])
    bag_repre = K.stack(bag_repre)
    bag_repre = Dropout(1 - keep_prob)(bag_repre)
    return __logit__(bag_repre, rel_tot), bag_repre
