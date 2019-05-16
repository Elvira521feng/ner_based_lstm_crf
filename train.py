import argparse
import os
import tensorflow as tf
import numpy as np

from fastNLP.io.dataset_loader import Conll2003Loader
from data_processor import sentences2ids, reload_mapping, random_embedding, tag2label
from my_model import BiLSTM_CRF
from utils.utils import str2bool, pre_embeddings
from utils.constant import VOCAB_PATH

# from my_model import BiLSTM_CRF


# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory


# hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for NER task')
parser.add_argument('--train_data', type=str, default='data/train.txt', help='train modeldata source')
parser.add_argument('--valid_data', type=str, default='data/valid.txt', help='valid modeldata source')
parser.add_argument('--test_data', type=str, default='data/test.txt', help='test modeldata source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='glove', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training modeldata before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1541061396', help='model for test and demo')
args = parser.parse_args()

# get word embeddings
vocab_path = VOCAB_PATH
word2id = reload_mapping(vocab_path)
embeddings = None
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    pre_embeddings_path = './data/glove.6B.100d.txt'
    embeddings = pre_embeddings(word2id, pre_embeddings_path)

# print("args:", args)
train_path = args.train_data
validation_path = args.valid_data
test_path = args.test_data

# 读入数据
loader = Conll2003Loader()
train_data = loader.load(train_path)
test_data = loader.load(test_path)
validation_data = loader.load(validation_path)  # ['tokens','pos', 'chunks', 'ner']


item_to_id = reload_mapping(vocab_path)

# 得到sent2ids
train_sent_ids = sentences2ids(train_data, item_to_id)
test_sent_ids = sentences2ids(test_data, item_to_id)
validation_sent_ids = sentences2ids(validation_data, item_to_id)

paths = "./model"

model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)




