import os
import numpy as np

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adadelta

from network.embedding import word_position_embedding
from network.encoder import cnn
from network.selector import bag_one
from network.classfier import output


class CNN_ONE(object):
    def __init__(self, train_data_helper, test_data_helper):
        self.train_data_helper = train_data_helper
        self.test_data_helper = test_data_helper
        self.word_vec_mat = self.train_data_helper.word_vec
        self.rel_tot = len(self.train_data_helper.rel2id)
        self.result_path = './result'
        self.model = None
    
    def _get_weight_table(self):
        print("Calculating weights_table...")
        _weights_table = np.zeros((self.rel_tot), dtype=np.float32)
        for i in range(len(self.train_data_helper.bag_label)):
            _weights_table[self.train_data_helper.bag_label[i]] += 1.0 
        _weights_table = 1 / (_weights_table ** 0.05)
        print("Finish calculating")
        return _weights_table
        
    def compile(self):
        word_input = Input(shape=(self.train_data_helper.batch_size,))
        pos1_input = Input(shape=(self.train_data_helper.batch_size,))
        pos2_input = Input(shape=(self.train_data_helper.batch_size,))
        scope_input = Input(shape=(self.train_data_helper.batch_size, 2))
        bag_label_input = Input(shape=(self.train_data_helper.batch_size,))

        after_embedding = word_position_embedding(word_input, self.word_vec_mat, pos1_input, pos2_input)
        after_encoding = cnn(after_embedding)
        train_logit, train_repre = bag_one(after_encoding, scope_input, bag_label_input, self.rel_tot)
        label_output = output(train_logit)

        self.model = Model(inputs=[word_input, pos1_input, pos2_input, scope_input, bag_label_input], 
                           outputs=label_output)
        self.model.compile(optimizer=Adadelta(lr=1e-4),
                           loss='categorical_crossentropy',
                           loss_weights=self._get_weight_table(),
                           metrics=['accuracy'])

    def train(self, epochs=1):
        for iter_step in range(epochs):
            batch_data = self.train_data_helper.next_batch()
            self.model.fit([batch_data['bag_rel'], 
                            batch_data['bag_rel'],  
                            batch_data['bag_rel'], 
                            batch_data['bag_rel'], 
                            batch_data['bag_rel']], batch_data['bag_rel'],
                      epochs=1)
        print('model storing')
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        self.model.save(os.path.join(self.result_path, 'keras_cnn_one.model'))
        print('storing finished')

    def summary(self):
        return self.model.summary()
