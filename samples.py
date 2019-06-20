# -*- coding: utf-8 -*-

import random
import _pickle as pickle

from pre_utils import *


def get_comments():
    df = pd.read_csv("../new_reviews/experience/cut_new_pcid4cid50228001.cvs", encoding="utf_8_sig")
    texts = map(str, df["comment_all"].values)
    size = len(df)
    del df
    print("has comments", size)
    return texts, size


def produce_sample(win_size, holder_dim, prob=1e-2, limit=2e5, show=False):
    texts, size = get_comments()
    char2vec = Char2Vec(restart_info=False)
    half_size = win_size // 2

    obj_cut_char = CutChar()
    obj_get_sample = GetSample(char2vec=char2vec, holder_dim=holder_dim, half_size=half_size)

    samples = list()
    for seq, text in enumerate(texts, start=1):
        cut_words = obj_cut_char.cut(text)
        start, end = half_size, len(cut_words) - half_size
        inx = start
        while inx < end:
            seed = random.uniform(0, 1)
            if seed < prob:
                record = obj_get_sample.get_sample(cut_words, inx)
                samples.append([record[2], record[3]])
                inx += 1
            else:
                inx += 1

        if 0 == seq % 1e4:
            print(f"completed {seq}/{size}")

        if 0 == len(samples) % 1e4:
            print(f"has samples {len(samples)}/{1e4}")

        if len(samples) >= limit:
            with open("data/samples.pkl", mode="wb") as fp:
                pickle.dump(samples, fp)
            break

    char2vec.dump_info()

    if show:
        for sample in samples:
            print(sample)

    return samples


def make_data(src="data/samples.pkl",
              train_file="data/train.pkl", dev_file="data/dev.pkl", test_file="data/test.pkl",
              train_share=0.8, dev_share=0.1, test_share=0.1, max_size=1e5):
    with open(src, mode="rb") as fp:
        samples = pickle.load(fp)

    random.shuffle(samples)
    if len(samples) > max_size:
        samples = samples[: max_size]

    p1 = int(len(samples) * train_share)
    p2 = p1 + int(len(samples) * dev_share)
    p3 = p2 + int(len(samples) * test_share)

    train = samples[: p1]
    dev = samples[p1: p2]
    test = samples[p2: p3]

    with open(train_file, mode="wb") as fp:
        pickle.dump(train, fp)

    with open(dev_file, mode="wb") as fp:
        pickle.dump(dev, fp)

    with open(test_file, mode="wb") as fp:
        pickle.dump(test, fp)


def load_data(train_file="data/train.pkl", dev_file="data/dev.pkl", test_file="data/test.pkl"):
    with open(train_file, mode="rb") as fp:
        train = pickle.load(fp)
    print("load train data", len(train))

    with open(dev_file, mode="rb") as fp:
        dev = pickle.load(fp)
    print("load dev data", len(dev))

    with open(test_file, mode="rb") as fp:
        test = pickle.load(fp)
    print("load test data", len(test))

    return train, dev, test


def make_1000():
    produce_sample(win_size, holder_dim, prob=1e-1, limit=1e3, show=True)
    make_data("data/test_data_1000.pkl")

    train, dev, test = load_data()

    print("\ntrain data")
    for line in train:
        print(line)

    print("\ndev data")
    for line in dev:
        print(line)

    print("\ntrain test")
    for line in test:
        print(line)


def make_10000():
    produce_sample(win_size, holder_dim, prob=1e-1, limit=1e4, show=False)
    make_data("data/samples.pkl")


if __name__ == '__main__':
    from CharEmbedding.settings import *

    # make_1000()
    # make_10000()

    train, dev, test = load_data()
    for line in test:
        print(line)
