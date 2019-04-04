import os
import pickle as pkl
import numpy as np
import json
from datetime import datetime

# from send_email import send_email

from tqdm import tqdm


def time_now():
    return datetime.now().strftime(r'%Y-%m-%d %H:%M:%S')


class DataHelper(object):
    # MODE_INSTANCE = 0       # normal multi-label classification
    # MODE_ENTITY_PAIRS = 1   # multi-instance learning by using multiple entity pairs
    # MODE_REL_FACTS = 2      # multi-instance learning by using multiple relaiton facts
    MODE_ENTITY_PAIRS_REL = 3

    def __init__(self, data_name, data_filepath, 
                 word_vec_filepath, rel2id_filepath, 
                 max_sentence_length=120, batch_size=100,
                 shuffle=True, mode=3):
                 
        if mode != self.MODE_ENTITY_PAIRS_REL:
            raise NotImplementedError('Not support mode for now')
        self.data_name = data_name
        self.max_sentence_length = max_sentence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.output_path = "./middle_data/"

        self.idx = 0

        self._data_raw = []
        self.word_vec_raw = {}
        self.rel2id = {}
        self.vocab = []
        self.word2id = {}
        self.word_vec = []
        self.bags = []
        self.bag_scope = []
        self.bag_label = []

        """Data loading"""
        if not self._middle_data_exists():
            print('[%s] loading %s data' % (time_now(), self.data_name))
            with open(data_filepath, 'r') as f:
                self._data_raw = json.load(f)
            print('[%s] dealing with train data case problems' % time_now())
            for i in tqdm(range(len(self._data_raw)), ncols=75):
                # self._data_raw[i]['relation'] = self._data_raw[i]['relation'].lower()
                self._data_raw[i]['sentence'] = self._data_raw[i]['sentence'].lower()
                self._data_raw[i]['head']['word'] = self._data_raw[i]['head']['word'].lower()
                self._data_raw[i]['tail']['word'] = self._data_raw[i]['tail']['word'].lower()
            
            print('[%s] loading rel2id' % time_now())        
            with open(rel2id_filepath, 'r') as f:
                self.rel2id = json.load(f)

            print('[%s] loading word vector data' % time_now())        
            if word_vec_filepath.endswith('.json'):
                with open(word_vec_filepath, 'r') as f:
                    self.word_vec_raw = json.load(f)
                for i in range(len(self.word_vec_raw)):
                    self.word_vec_raw[i]['word'] = self.word_vec_raw[i]['word'].lower()
                    self.vocab.append(self.word_vec_raw[i]['word'])
                self.vocab_size = len(self.word_vec_raw)
                self.word_vec_dim = len(self.word_vec_raw[0]['vec'])
                
                self.word_vec = np.zeros((self.vocab_size, self.word_vec_dim), dtype=np.float32)
                for ind, word in tqdm(enumerate(self.vocab), ncols=75):
                    self.word2id[word] = ind
                    self.word_vec[ind, :] = self.word_vec_raw[ind]['vec']
                self.word2id['UNK'] = self.vocab_size
                self.word2id['BLANK'] = self.vocab_size + 1

                print('[%s] sorting data' % (time_now()))
                self._data_raw.sort(key=lambda x: x['head']['word'] + '#' + x['tail']['word'] + '#' + x['relation'])

                print('[%s] building data' % time_now())
                self.number_of_instances = len(self._data_raw)
                self.sentence_word = np.zeros((self.number_of_instances, self.max_sentence_length), dtype=np.int64)
                self.sentence_pos1 = np.zeros((self.number_of_instances, self.max_sentence_length), dtype=np.int64)
                self.sentence_pos2 = np.zeros((self.number_of_instances, self.max_sentence_length), dtype=np.int64)
                self.sentence_label = np.zeros((self.number_of_instances), dtype=np.int64)
                self.sentence_length = np.zeros((self.number_of_instances), dtype=np.int64)

                for ind, data in tqdm(enumerate(self._data_raw), ncols=75):
                    self.sentence_word[ind, :] = self._get_sentence_sequence(data['sentence'])
                    self.sentence_label[ind] = self.rel2id[data['relation']]
                    self.sentence_length[ind] = len(data['sentence'].split())
                    self.sentence_pos1[ind, :], self.sentence_pos2[ind, :] = \
                        self._get_sentence_position_embedding(data['sentence'], 
                                                            data['head']['word'], data['tail']['word'])
                    bag = "%s#%s#%s" % (data['head']['word'], data['tail']['word'], data['relation'])
                    
                    if len(self.bags) == 0 or self.bags[len(self.bag_scope) - 1][1] != bag:
                        self.bags.append(bag)
                        self.bag_scope.append([ind, ind])
                    else:
                        self.bag_scope[len(self.bag_scope) - 1][1] = ind
                for scope in self.bag_scope:
                    self.bag_label.append(self.sentence_label[scope[0]])
                
                self.bag_label = np.array(self.bag_label, dtype=np.int64)
                self.bag_scope = np.array(self.bag_scope, dtype=np.int64)
                
                self.indices = list(range(len(self.bag_scope)))
                if self.shuffle:
                    import random
                    random.shuffle(self.indices)
                self.indices = np.array(self.indices)

                del self._data_raw
                del self.word_vec_raw
                del self.rel2id
                del self.vocab
                del self.word2id

            elif word_vec_filepath.endswith('.pkl'):
                # with open(word_vec_filepath, 'rb') as f:
                #     self.word_vec_raw = pkl.load(f)
                raise NotImplementedError('Temporarily not support pkl format')
            else:
                raise ValueError('Not support word2vec format (suffix): %s' % word_vec_filepath.split('.')[-1])
            
            print("[%s] storing data" % time_now())
            if not os.path.exists(self.output_path):
                os.mkdir(self.output_path)
            np.save(os.path.join(self.output_path, self.data_name + '_word_vec.npy'), self.word_vec)
            np.save(os.path.join(self.output_path, self.data_name + '_word.npy'), self.sentence_word)
            np.save(os.path.join(self.output_path, self.data_name + '_pos1.npy'), self.sentence_pos1)
            np.save(os.path.join(self.output_path, self.data_name + '_pos2.npy'), self.sentence_pos2)
            np.save(os.path.join(self.output_path, self.data_name + '_bag_label.npy'), self.bag_label)
            np.save(os.path.join(self.output_path, self.data_name + '_bag_scope.npy'), self.bag_scope)
            np.save(os.path.join(self.output_path, self.data_name + '_instance_label.npy'), self.sentence_label)
            np.save(os.path.join(self.output_path, self.data_name + '_instance_length.npy'), self.sentence_length)
            np.save(os.path.join(self.output_path, self.data_name + '_indices.npy'), self.indices)
            print("Finish storing")
        else:
            print('[%s] find middle data, loading...' % time_now())
            self._load_middle_data()
        
        print('word embedding: %d words and %d dimensions' % (self.word_vec.shape[0], self.word_vec.shape[1]))
        print('number of bags: %d, number of instances: %d' % (len(self.bag_label), len(self.sentence_label)))
        print('[%s] init finished' % time_now())

    def _get_sentence_position_embedding(self, sentence, head, tail):
        pos1, pos2 = self._get_pos(sentence, head, tail)
        pos1_embedding = []
        pos2_embedding = []
        for pos in range(self.max_sentence_length):
            pos1_embedding.append(pos - pos1 + self.max_sentence_length)
            pos2_embedding.append(pos - pos2 + self.max_sentence_length)
        return pos1_embedding, pos2_embedding

    def _get_sentence_sequence(self, sentence):
        words = sentence.split()
        sentence_sequence = np.zeros(self.max_sentence_length, dtype=np.int32)
        for ind, word in enumerate(words):
            if ind >= self.max_sentence_length:
                break
            if word in self.word2id:
                sentence_sequence[ind] = self.word2id[word]
            else:
                sentence_sequence[ind] = self.word2id['UNK']
        for i in range(len(words), self.max_sentence_length):
            sentence_sequence[i] = self.word2id['BLANK']

        return sentence_sequence

    def _get_pos(self, sentence, head, tail):
        # sentence.split() will split by spaces in default
        # so no worry if there is extra spaces
        sen_pos1 = sentence.find(head)
        sen_pos2 = sentence.find(tail)
        head_pos = tail_pos = 0
        for ind in range(sen_pos1):
            if sentence[ind] == ' ':
                head_pos += 1
        for ind in range(sen_pos2):
            if sentence[ind] == ' ':
                tail_pos += 1
        return head_pos, tail_pos
    
    def _load_middle_data(self):
        self.word_vec = np.load(os.path.join(self.output_path, self.data_name + '_word_vec.npy'))
        self.sentence_word = np.load(os.path.join(self.output_path, self.data_name + '_word.npy'))
        self.sentence_pos1 = np.load(os.path.join(self.output_path, self.data_name + '_pos1.npy'))
        self.sentence_pos2 = np.load(os.path.join(self.output_path, self.data_name + '_pos2.npy'))
        self.bag_label = np.load(os.path.join(self.output_path, self.data_name + '_bag_label.npy'))
        self.bag_scope = np.load(os.path.join(self.output_path, self.data_name + '_bag_scope.npy'))
        self.sentence_label = np.load(os.path.join(self.output_path, self.data_name + '_instance_label.npy'))
        self.sentence_length = np.load(os.path.join(self.output_path, self.data_name + '_instance_length.npy'))
        self.indices = np.load(os.path.join(self.output_path, self.data_name + '_indices.npy'))

    def _middle_data_exists(self):
        if not os.path.exists(self.output_path) or \
            not os.path.exists(os.path.join(self.output_path, self.data_name + '_word_vec.npy')) or \
            not os.path.exists(os.path.join(self.output_path, self.data_name + '_word.npy')) or \
            not os.path.exists(os.path.join(self.output_path, self.data_name + '_pos1.npy')) or \
            not os.path.exists(os.path.join(self.output_path, self.data_name + '_pos2.npy')) or \
            not os.path.exists(os.path.join(self.output_path, self.data_name + '_bag_label.npy')) or \
            not os.path.exists(os.path.join(self.output_path, self.data_name + '_bag_scope.npy')) or \
            not os.path.exists(os.path.join(self.output_path, self.data_name + '_instance_label.npy')) or \
            not os.path.exists(os.path.join(self.output_path, self.data_name + '_instance_length.npy')) or \
            not os.path.exists(os.path.join(self.output_path, self.data_name + '_indices.npy')):
            return False
        else:
            return True

    def next_batch(self):
        self.new_idx = self.idx + self.batch_size
        if self.idx >= len(self.bag_label):
            raise StopIteration
        if self.new_idx > len(self.indices):
            self.new_idx = len(self.indices)
        batch_indices = self.indices[self.idx, self.new_idx]
        
        _word = []; _pos1 = []; _pos2 = []; _bag_rel = []; _ins_rel = []; _length = []; _bag_scope = []
        cur_pos = 0
        for ind in range(self.idx, self.new_idx):
            _word.append(self.sentence_word[ self.bag_scope[ self.indices[ind] ][0] : self.bag_scope[self.indices[ind]][1] ])
            _pos1.append(self.sentence_pos1[self.bag_scope[self.indices[ind]][0]:self.bag_scope[self.indices[ind]][1]])
            _pos2.append(self.sentence_pos2[self.bag_scope[self.indices[ind]][0]:self.bag_scope[self.indices[ind]][1]])
            _bag_rel.append(self.bag_label[self.bag_scope[self.indices[ind]][0]])
            _ins_rel.append(self.sentence_label[self.bag_scope[self.indices[ind]][0]:self.bag_scope[self.indices[ind]][1]])
            _length.append(self.sentence_length[self.bag_scope[self.indices[ind]][0]:self.bag_scope[self.indices[ind]][1]])
            bag_size = self.bag_scope[self.indices[ind]][1] - self.bag_scope[self.indices[ind]][0]
            _scope.append([cur_pos, cur_pos + bag_size])
            cur_pos = cur_pos + bag_size
        for ind in range(batch_size - (self.new_idx - self.idx)):   # padding
            _word.append(np.zeros((1, self.sentence_word.shape[-1]), dtype=np.int32))
            _pos1.append(np.zeros((1, self.sentence_pos1.shape[-1]), dtype=np.int32))
            _pos2.append(np.zeros((1, self.sentence_pos2.shape[-1]), dtype=np.int32))
            _bag_rel.append(0)
            _ins_rel.append(np.zeros((1), dtype=np.int32))
            _length.append(np.zeros((1), dtype=np.int32))
            _scope.append([cur_pos, cur_pos + 1])
            cur_pos += 1

        batch_data = {}
        batch_data['word'] = np.concatenate(_word)
        batch_data['pos1'] = np.concatenate(_pos1)
        batch_data['pos2'] = np.concatenate(_pos2)
        batch_data['bag_rel'] = np.stack(_bag_rel)
        batch_data['ins_rel'] = np.concatenate(_ins_rel)
        batch_data['length'] = np.concatenate(_length)
        batch_data['scope'] = np.stack(_scope)

        self.idx = self.new_idx

        return batch_data

if __name__ == "__main__":
    test = DataHelper('test', './data/nyt/test.json', './data/nyt/word_vec.json', './data/nyt/rel2id.json')
    import sys
    cnt = 1
    process = test.bag_label.shape[0]*test.batch_size
    while True:
        try:
            print("%10d/%10d" % (cnt, process), end='\r')
            cnt += 1
        except StopIteration:
            break
