#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import sys
import time
import tensorflow as tf

from fastNLP.io.dataset_loader import Conll2003Loader
from data_processor import reload_mapping, random_embedding, tag2label, dataset2list
from my_model import BiLSTM_CRF
from utils.utils import pre_embeddings, get_logger
from utils.constant import *

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

sys.stdout.flush()

# hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for NER task')
parser.add_argument('--train_data', type=str, default='data/train.txt', help='train modeldata source')
parser.add_argument('--valid_data', type=str, default='data/valid.txt', help='valid modeldata source')
parser.add_argument('--test_data', type=str, default='data/test.txt', help='test modeldata source')
parser.add_argument('--vocabulary', type=str, default='./word2id.pkl', help='vocabulary')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=60, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='glove', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=100, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle training modeldata before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
args = parser.parse_args()

# get word embeddings
vocab_path = args.vocabulary
word2id = reload_mapping(vocab_path)

embeddings = None
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    pre_embeddings_path = './glove.6B.100d.txt'
    embeddings = pre_embeddings(word2id, pre_embeddings_path)

train_path = args.train_data
validation_path = args.valid_data
test_path = args.test_data

# load data
loader = Conll2003Loader()
train_data = loader.load(train_path)
train_data = dataset2list(train_data)
test_data = loader.load(test_path)
test_data = dataset2list(test_data)
valid_data = loader.load(validation_path)  # ['tokens','pos', 'chunks', 'ner']
valid_data = dataset2list(valid_data)

# paths setting
paths = {}
timestamp = str(int(time.time()))
summary_path = os.path.join(EVALUATION_DIR, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(MODEL_DIR, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
paths['model_path'] = model_path
result_path = os.path.join(EVALUATION_DIR, "results")
paths['result_path'] = result_path
paths['test_result_path'] = os.path.join(EVALUATION_DIR,'test_result/')
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

if args.mode == "train":
    print("Start training!!")
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=valid_data)
    print("End of the training!")
elif args.mode == "test":
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    test_size = len(test_data)
    print("test data: {}".format(test_size))
    label_list, seq_len_list = model.test(test_data, "predict")
    model.evaluate(label_list, seq_len_list, test_data)
else:
    print("no this mode!")