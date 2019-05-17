import argparse
import os
import sys
import time

import tensorflow as tf

from fastNLP.io.dataset_loader import Conll2003Loader
from data_processor import reload_mapping, random_embedding, tag2label, dataset2list
from my_model import BiLSTM_CRF
from utils.utils import pre_embeddings, get_logger
from utils.constant import VOCAB_PATH

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory
print("环境参数设置完成！")

sys.stdout.flush()

# hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for NER task')
parser.add_argument('--train_data', type=str, default='data/train.txt', help='train modeldata source')
parser.add_argument('--valid_data', type=str, default='data/valid.txt', help='valid modeldata source')
parser.add_argument('--test_data', type=str, default='data/test.txt', help='test modeldata source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=20, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=bool, default=False, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='glove', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle training modeldata before each epoch')
parser.add_argument('--demo_model', type=str, default='1541061396', help='model for test and demo')
args = parser.parse_args()
print("添加超参完成！")

# get word embeddings
vocab_path = VOCAB_PATH
word2id = reload_mapping(vocab_path)
print("加载vocabulary！")
embeddings = None
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    pre_embeddings_path = './glove.6B.100d.txt'
    embeddings = pre_embeddings(word2id, pre_embeddings_path)
print("获得预训练embedding！")

# print("args:", args)
train_path = args.train_data
validation_path = args.valid_data
test_path = args.test_data

# 读入数据
loader = Conll2003Loader()
train_data = loader.load(train_path)
train_data = dataset2list(train_data)
test_data = loader.load(test_path)
test_data = dataset2list(test_data)
valid_data = loader.load(validation_path)  # ['tokens','pos', 'chunks', 'ner']
valid_data = dataset2list(valid_data)
print("读入数据完成！")

# item_to_id = reload_mapping(vocab_path)
# print("加载vocabulary完成！")

# # 得到sent2ids
# train_sent_ids = sentences2ids(train_data, item_to_id)
# test_sent_ids = sentences2ids(test_data, item_to_id)
# validation_sent_ids = sentences2ids(valid_data, item_to_id)

# paths = "./model"

# paths setting
paths = {}
timestamp = str(int(time.time()))
output_path = './model/'
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
paths['test_result_path'] = os.path.join('.','test_result/')
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))
print("设置路径完成！")

model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
print("初始化模型完成！")
model.build_graph()
print("构建计算图！！")
print("train data: {}".format(len(train_data)))
print("开始训练模型！")
model.train(train=train_data, dev=valid_data)
print("模型训练完成！")


# ckpt_file = tf.train.latest_checkpoint(model_path)
#     print(ckpt_file)
#     paths['model_path'] = ckpt_file
#     model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
#     model.build_graph()
#     saver = tf.train.Saver()
#     with tf.Session(config=config) as sess:
#         print('============= demo =============')
#         saver.restore(sess, ckpt_file)
#         while(1):
#             print('Please input your sentence:')
#             demo_sent = input()
#             if demo_sent == '' or demo_sent.isspace():
#                 print('See you next time!')
#                 break
#             else:
#                 demo_sent = list(demo_sent.strip())
#                 demo_data = [(demo_sent, ['O'] * len(demo_sent))]
#                 tag = model.demo_one(sess, demo_data)
#                 PER, LOC, ORG = get_entity(tag, demo_sent)
#                 print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))

