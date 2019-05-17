import pickle
import random
import numpy as np
from fastNLP.io.dataset_loader import Conll2003Loader

from utils.constant import TOTAL_PATH, VOCAB_PATH


def dataset2list(data):
    new_data = []
    for tmp in data:
        sent_ = tmp['tokens']
        tag_ = tmp['ner']
        new_data.append((sent_, tag_))

    return new_data


def generate_tag2label():
    tag2label = {'O': 0}
    tags = ['ORG', 'MISC', 'PER', 'LOC']
    bilist = ['B-', 'I-']
    for i in range(len(tags)):
        for x in bilist:
            tag = x + tags[i]
            tag2label[tag] = len(tag2label)

    return tag2label


tag2label =generate_tag2label()


def create_dictionary(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dictionary = {}
    for items in item_list:
        for item in items:
            if item not in dictionary:
                dictionary[item] = 1
            else:
                dictionary[item] += 1
    return dictionary


def create_mapping(dictionary, vocab_path):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dictionary.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}

    with open(vocab_path, 'wb') as fw:
        pickle.dump(item_to_id, fw)

    return item_to_id, id_to_item


def reload_mapping(vocab_path):
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def sentences2ids(sents, vocab):
    """
    :param sents:
    :param vocab:
    :return:
    """
    sent_ids = []
    for sent in sents:
        word_ids = sentence2id(sent['tokens'], vocab)
        sent_ids.append(word_ids)

    return sent_ids


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def sentence2id(sent, vocab):
    ids = []
    for word in sent:
        ids.append(vocab[word])

    return ids


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for sent_, tag_ in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


if __name__ == '__main__':
    loader = Conll2003Loader()
    total_path = TOTAL_PATH
    total_data = loader.load(total_path)

    # 建词表
    word_list = []
    for sent in total_data:
        word_list.append(sent['tokens'])
    word_dictionary = create_dictionary(word_list)
    print('the count unique word:', len(word_dictionary))
    # print(word_dictionary)

    vocab_path = VOCAB_PATH
    item_to_id, id_to_item = create_mapping(word_dictionary, vocab_path)
    # print(item_to_id)