import tensorflow as tf
from keras.layers import Conv1D, GlobalMaxPool1D, Activation, Dropout

def cnn(x, hidden_size=230, kernel_size=3, stride_size=1, activation='relu', var_scope=None, keep_prob=1.0):
    conv = Conv1D(filters=hidden_size, kernel_size=kernel_size,
                  strides=stride_size, padding='same', 
                  kernel_initializer='glorot_normal')(x)
    pooling = GlobalMaxPool1D()(conv)
    activation = Activation('relu')(pooling)
    dropout = Dropout(1 - keep_prob)(activation)
    return dropout
