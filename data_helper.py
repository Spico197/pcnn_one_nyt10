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
    MODE_ENTITY_PAIRS = 1   # multi-instance learning by using multiple entity pairs
    # MODE_REL_FACTS = 2      # multi-instance learning by using multiple relaiton facts

    def __init__(self, data_name, data_filepath, 
                 word_vec_filepath, rel2id_filepath, 
                 max_sentence_length=120, batch_size=100,
                 shuffle=True, mode=self.MODE_ENTITY_PAIRS):
                 
        if mode != self.MODE_ENTITY_PAIRS:
            raise NotImplementedError('Not support mode for now')
        self.data_name = data_name
        self.max_sentence_length = max_sentence_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.idx = 0

        self._data_raw = []
        self.word_vec_raw = {}
        self.rel2id = {}
        self.vocab = []
        self.word2id = {}
        self.word_vec = []
        self.bags = []

        """Data loading"""
        if not self._middle_data_exists():
            print('[%s] loading train data' % time_now())
            with open(data_file_path, 'r') as f:
                self._data_raw = json.load(f)
            print('[%s] dealing with train data case problems' % time_now())
            for i in tqdm(range(len(self._data_raw))):
                self._data_raw[i]['relation'] = self._data_raw[i]['relation'].lower()
                self._data_raw[i]['head']['word'] = self._data_raw[i]['head']['word'].lower()
                self._data_raw[i]['tail']['word'] = self._data_raw[i]['tail']['word'].lower()
            
            print('[%s] loading rel2id' % time_now())        
            with open(rel2id_filepath, 'r') as f:
                self.rel2id = json.load(f)

            print('[%s] loading word vector data' % time_now())        
            if word_vec_filepath.endswith('.json'):
                with open(word_vec_filepath, 'r') as f:
                    self.word_vec_raw = json.load(word_vec_filepath)
                self.vocab = [word.lower() for word in self.word_vec_raw.keys()]
                self.vocab_size = len(self.word_vec_raw)
                self.word_vec_dim = len(self.word_vec_raw[0]['vec'])
                
                print('word embedding: %d words and %d dimensions' % (self.vocab_length, self.word_vec_dim))
                self.word_vec = np.zeros((self.vocab_size, self.word_vec_dim), dtype=np.float32)
                for ind, word in tqdm(enumerate(self.vocab)):
                    self.word2id[word] = ind
                    self.word_vec[ind, :] = self.word_vec_raw[ind]['vec']

            print('[%s] sorting data' % (time_now()))
            self._data_raw.sort(key=lambda x: x['head']['word'] + '#' + x['tail']['word'] + '#' + x['relation'])

            print('[%s] building data')
            self.number_of_instances = len(self._data_raw)
            self.sentence_word = np.zeros((self.number_of_instances, self.max_sentence_length), dtype=np.int64)
            self.sentence_pos1 = np.zeros((self.number_of_instances, self.max_sentence_length), dtype=np.int64)
            self.sentence_pos2 = np.zeros((self.number_of_instances, self.max_sentence_length), dtype=np.int64)
            self.sentence_label = np.zeros((self.number_of_instances), dtype=np.int64)
            self.sentence_len = np.zeros((self.number_of_instances), dtype=np.int64)

            for ind, data in tqdm(enumerate(self._data_raw)):
                self.sentence_word[ind, :] = self._get_sentence_sequence(data['sentence'])
                self.sentence_label[ind] = self.rel2id(data['relation'])
                self.sentence_len[ind] = len(data['sentence'].split())
                self.sentence_pos1[ind, :], self.sentence_pos2[ind, :] = \
                    self._get_sentence_position_embedding(data['sentence'], 
                                                          data['head']['word'], data['tail']['word'])


            elif word_vec_filepath.endswith('.pkl'):
                # with open(word_vec_filepath, 'rb') as f:
                #     self.word_vec_raw = pkl.load(f)
                raise NotImplementedError('Temporarily not support pkl format')
            else:
                raise ValueError('Not support word2vec format (suffix): %s' % word_vec_filepath.split('.')[-1])
            
            print("[%s] storing data" % time_now())
            if not os.path.exists('./middle_data/'):
                os.mkdir('./middle_data/')
            
        else:
            self._load_middle_data()

    def _get_sentence_position_embedding(self, sentence, head, tail):
        pos1, pos2 = self._get_pos(sentence, head, tail)
        pos1_embedding = []
        pos2_embedding = []
        for pos in range(len(sentence.split())):
            pos1_embedding.append(pos - pos1)
            pos2_embedding.append(pos - pos2)
        return pos1_embedding, pos2_embedding

    def _get_sentence_sequence(self, sentence):
        words = sentence.split()
        sentence_sequence = np.zeros((1, self.max_sentence_length), dtype=np.int32)
        for ind, word in enumerate(words):
            if ind >= self.max_sentence_length:
                break
            sentence_sequence[1, ind] = self.word2id(word)
        return sentence_sequence

    def _get_pos(self, sentence, head, tail):
        # sentence.split() will split by spaces in default
        # so no worry if there is extra spaces
        words = sentence.split() 
        head_pos = words.index(head)
        tail_pos = words.index(tail)
        return head_pos, tail_pos
    
    def _load_middle_data(self):
        pass