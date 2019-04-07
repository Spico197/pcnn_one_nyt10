import tensorflow as tf
from keras.layers import Embedding
from keras.losses import categorical_crossentropy
from keras import backend as K

def softmax_cross_entropy(x, label, rel_tot, weights_table=None, var_scope=None):
    with tf.variable_scope(var_scope or "loss", reuse=tf.AUTO_REUSE):
        if weights_table is None:
            weights = 1.0
        else:
            weights = Embedding(input_dim=label.shape[1], 
                                output_dim=weights_table.shape[1], 
                                weights=[weights_table])(label)
        label_onehot = K.one_hot(indices=label, num_classes=rel_tot)
        pred_output_label = K.one_hot(output(x))
        return loss, output(x)

def output(x):
    return K.argmax(x, axis=-1)
