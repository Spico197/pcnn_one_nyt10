import numpy as np

from keras import backend as K
from keras.layers import Embedding, Input, concatenate


def word_embedding(word, word_vec_mat, name=None, word_embedding_dim=50, add_unk_and_blank=True):
    if add_unk_and_blank:
        word_vec_mat = np.concatenate([word_vec_mat, np.zeros((2, word_embedding_dim), dtype=np.float32)], axis=0)
    word_embedding = Embedding(input_length=K.int_shape(word)[0], input_dim=word_vec_mat.shape[0],
                               output_dim=word_embedding_dim, name=name, weights=[word_vec_mat])(word)
    return word_embedding

def pos_embedding(pos1, pos2, name=None, pos_embedding_dim=5, max_length=120):
    # print('\n\n\npos1 shape: ', K.int_shape(pos1))
    # assert 1 == 2
    pos1_embedding = Embedding(input_dim=max_length*2, input_length=K.int_shape(pos1)[0],
                               output_dim=pos_embedding_dim)(pos1)
    pos2_embedding = Embedding(input_dim=max_length*2, input_length=K.int_shape(pos2)[0],
                               output_dim=pos_embedding_dim)(pos2)
    return pos1_embedding, pos2_embedding

def word_position_embedding(word, word_vec_mat, pos1, pos2, word_embedding_dim=50, pos_embedding_dim=5, max_length=120, add_unk_and_blank=True):
    w_embedding = word_embedding(word, word_vec_mat, name='word_embedding', word_embedding_dim=word_embedding_dim, add_unk_and_blank=add_unk_and_blank)
    pos1_embedding, pos2_embedding = pos_embedding(pos1, pos2, name='positon_embedding', pos_embedding_dim=pos_embedding_dim, max_length=max_length)
    return concatenate([w_embedding, pos1_embedding, pos2_embedding])
