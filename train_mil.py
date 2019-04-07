from data_helper import DataHelper
from models.cnn_one import CNN_ONE


train_data_helper = DataHelper('train', './data/nyt/train.json', './data/nyt/word_vec.json', './data/nyt/rel2id.json')
test_data_helper = DataHelper('test', './data/nyt/test.json', './data/nyt/word_vec.json', './data/nyt/rel2id.json')

model = CNN_ONE(train_data_helper, test_data_helper)
model.compile()
model.summary()
# model.train(1)
