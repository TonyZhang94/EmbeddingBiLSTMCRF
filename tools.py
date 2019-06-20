# -*- coding: utf-8 -*


import _pickle as pickle
import datetime


def read_word2id():
    # with open("data_path/word2id.pkl", mode="rb") as fp:
    #     data = pickle.load(fp)
    with open("data_path/bak/word2id.pkl", mode="rb") as fp:
        data = pickle.load(fp)

    print("type of word2id.pkl", type(data))
    print("word2id len", len(data))
    for c, line in data.items():
        # print(len(c))
        print(c, line)
        # exit(0)
        if c == '<NUM>':
            print("<NUM>", line)
        if c == '<ENG>':
            print("<ENG>", line)
        if c == '<UNK>':
            print("<UNK>", line)
    # rank = 0
    # for char in data:
    #     print(char)
    #     rank += 1
    #     if rank > 10:
    #         break


def read_test_data_line():
    test_data_line = "data_path/original/test1.txt"
    with open(test_data_line, mode="r", encoding="utf-8") as fp:
        num = 0
        lines = fp.readlines()
        for line in lines:
            if len(line) < 1:
                continue
            num += 1
    print("test1 len =", num)


if __name__ == '__main__':
    read_word2id()
    # read_test_data_line()
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #
    # print(timestamp)
    # print(type(timestamp))


