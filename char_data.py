# -*- coding:utf-8 -*-

import sys, pickle, os, random
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

# tags, BIO
# tag2label = {"O": 0,
#              "B-PER": 1, "I-PER": 2,
#              "B-LOC": 3, "I-LOC": 4,
#              "B-ORG": 5, "I-ORG": 6
#              }
tag2label = {"O": 0,
             "B-T": 1, "I-T": 2
             }


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    # with open("BiLSTM_CRF\\"+corpus_path, encoding='utf-8') as fr:
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            # print("one", word)
            word = '<NUM>'
        # 大小写字母65-90，97-122
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            # print("two", word)
            word = '<ENG>'
        if word not in word2id:
            # print("three", word)
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def sentence2embedding(sent, obj_char2vec, init_embedding):
    one_input = list()
    for ch in sent:
        try:
            vec = init_embedding[ch]
        except KeyError:
            vec = obj_char2vec.char2vec(ch)
            init_embedding[ch] = vec
        one_input.append(vec)
    return one_input


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    # with open("BiLSTM_CRF\\"+vocab_path, 'rb') as fr:
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    # print(word2id)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def bert_embedding(vocab, embedding_dim):
    word_vectors = KeyedVectors.load_word2vec_format("embeddings/bert.vector", binary=False)
    embedding_list = list()
    unfind_num = 0
    unfind_list = list()
    for hanz in vocab:
        vetcor = np.zeros([word_vectors.vector_size], np.float32)
        if hanz in word_vectors.vocab:
            vetcor += word_vectors[hanz]
        else:
            # print("embedding hanz {} not in word_vector".format(hanz))
            unfind_num += 1
            unfind_list.append(hanz)
            vetcor = np.random.uniform(-0.25, 0.25, embedding_dim)
        embedding_list.append(vetcor)
    embedding_mat = np.array(embedding_list)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """
    # 取句子长度，以及限制最长长度

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, obj_char2vec, obj_cut_char, init_embedding, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    # 一句话不会被打乱
    # 多句输入会被打乱顺序，一句话内部不会
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    sequence_length = 0
    for (sent_, tag_) in data:
        sequence_length = max(sequence_length, len(sent_))
    for (sent_, tag_) in data:
        sent_ = obj_cut_char.cut("".join(sent_))
        while len(sent_) < sequence_length:
            sent_.append("<BLANK>")
            tag_.append("O")
        # 找到一句话每个字one hot向量
        # sent_ = sentence2id(sent_, vocab)
        sent_ = sentence2embedding(sent_, obj_char2vec, init_embedding)
        # tag2label = { "O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6 }
        # 根据给定类别给序号，e.g. [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        label_ = [tag2label[tag] for tag in tag_]
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

