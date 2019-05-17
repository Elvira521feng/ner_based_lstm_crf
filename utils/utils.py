import argparse
import logging
import numpy as np


def pre_embeddings(word_index, pre_embeddings_path):
    embeddings_index = get_pre_embeddings(pre_embeddings_path)
    embeddings = np.zeros((len(word_index), 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        # print(word)
        # print(embedding_vector)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embeddings[i] = embedding_vector

    return embeddings


def get_pre_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


