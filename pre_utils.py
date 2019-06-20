# -*- coding: utf-8 -*-

import _pickle as pickle
import pandas as pd


class Char2Vec(object):
    def __init__(self, restart_info=False):
        digits = [str(n) for n in range(10)] + ['A', 'B', 'C', 'D', 'E', 'F']
        self.digits2num = {digits[num]: num for num in range(16)}
        self.digits2num.update({'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15})
        self.special = {'<ENG>': 0xA000, '<SYM>': 0xA001, '<NUM>': 0xA002, '<UNK>': 0xA003, '<MASK>': 0xA004,
                        '<BLANK>': 0xA007}

        self.start = 0x4e00
        self.end = 0x9fff
        self.size = 21000

        self.__char2pos = None
        self.__char2vec = None
        # self.init_info()
        self.load_info(restart=restart_info)
        self.log_size = (len(self.__char2pos) + 1e3) // 1e3 * 1e3

    def init_info(self):
        self.__char2pos = dict()
        self.__char2vec = dict()
        self.dump_info()

    def load_info(self, restart=False):
        if restart:
            self.__char2pos = dict()
            self.__char2vec = dict()
            for k, v in self.special.items():
                pos = v - self.start
                self.__char2pos[k] = pos
                vec = [0] * self.size
                vec[pos] = 1
                self.__char2vec[k] = vec
        else:
            with open("char2info/char2pos.pkl", mode="rb") as fp:
                self.__char2pos = pickle.load(fp)
                for key, pos in self.special.items():
                    self.__char2pos[key] = pos - self.start

            with open("char2info/char2vec.pkl", mode="rb") as fp:
                self.__char2vec = pickle.load(fp)
                for key, pos in self.special.items():
                    vec = [0] * self.size
                    vec[pos - self.start] = 1
                    self.__char2vec[key] = vec

            print("load info size:", len(self.__char2pos))

    def dump_info(self):
        with open("char2info/char2pos.pkl", mode="wb") as fp:
            pickle.dump(self.__char2pos, fp)
        del self.__char2pos

        with open("char2info/char2vec.pkl", mode="wb") as fp:
            pickle.dump(self.__char2vec, fp)
        del self.__char2vec

    def char2vec(self, ch, onehot=True):
        try:
            pos = self.__char2pos[ch]
            vec = self.__char2vec[ch]
        except KeyError:
            if onehot:
                pos = self.char2rank(ch) - self.start
                vec = [0] * self.size
                vec[pos] = 1

                if ch in self.__char2pos or ch in self.__char2vec:
                    raise Exception(f"{ch} has been included in info")
                self.__char2pos[ch] = pos
                self.__char2vec[ch] = vec
            else:
                raise Exception(f"Error: {ch} not considered")
        # print(ch, pos)
        if len(self.__char2pos) >= self.log_size:
            print(f"has record char {self.log_size}")
            self.dump_info()
            self.load_info(restart=False)
            self.log_size += 1e3

        return vec

    def char2rank(self, ch):
        u_form = ch.encode("unicode_escape")
        u_form = str(u_form).replace("b'\\\\u", "").replace("'", "")
        rank = 0
        for x in list(u_form):
            rank = rank * 16 + self.digits2num[x]

        # if rank < self.start or self.end < rank:
        #     print(ch, "0x%x" % rank, "不在0x4e00~0x9FFF范围内")
        #     rank = 0xA000

        return rank


class CutChar(object):
    def __init__(self):
        self.symbols = self.load_symbols()

    @staticmethod
    def load_symbols():
        symbols = set()
        with open("char2info/symbols.txt", mode="r", encoding="utf-8") as fp:
            for line in fp.readlines():
                symbols.add(line.strip())

        symbols.add(" ")
        # print("symbols num:", len(symbols))
        return symbols

    @staticmethod
    def is_num(ch):
        if '0' <= ch <= '9':
            return True
        else:
            return False

    @staticmethod
    def is_en(ch):
        if "a" <= ch <= "z":
            return True
        if "A" <= ch <= "Z":
            return True
        return False

    def is_symbol(self, ch):
        if ch in self.symbols:
            return True
        else:
            return False

    def cut(self, text):
        res = list()
        inx = 0
        while inx < len(text):
            ch = text[inx]
            if u'\u4e00' <= ch <= u'\u9FFF':
                res.append(ch)
                # print(ch, ch)
                inx += 1
            elif self.is_num(ch):
                num = ch
                flag = False
                while inx + 1 < len(text):
                    inx += 1
                    ch = text[inx]
                    if self.is_num(ch):
                        num += ch
                    else:
                        flag = True
                        break
                # res.append(num)
                res.append('<NUM>')
                # print(num, '<NUM>')
                if not flag:
                    inx += 1
            elif self.is_en(ch):
                en = ch
                flag = False
                while inx + 1 < len(text):
                    inx += 1
                    ch = text[inx]
                    if self.is_en(ch):
                        en += ch
                    else:
                        flag = True
                        break
                # res.append(en)
                res.append('<ENG>')
                # print(en, '<ENG>')
                if not flag:
                    inx += 1
            elif self.is_symbol(ch):
                res.append('<SYM>')
                # print(ch, '<SYM>')
                inx += 1
            else:
                res.append('<UNK>')
                # print(ch, '<UNK>')
                inx += 1
        return res


class GetSample(object):
    def __init__(self, char2vec, holder_dim, half_size):
        self.char2vec = char2vec
        self.holder_dim = holder_dim
        self.half_size = half_size

    def get_sample_interrupt(self, cut_words, inx):
        # sample = list()
        # interrupt = False
        # for offset in range(self.half_size):
        #     ch = cut_words[inx - offset]
        #     if '<SYM>' == ch:
        #         interrupt = True
        #
        #     if not interrupt:
        #         vec = self.char2vec.char2vec(ch)
        #     else:
        #         vec = [0] * self.holder_dim
        pass

    def get_sample(self, cut_words, inx):
        inputs = list()
        short_text = list()
        for offset in range(self.half_size):
            ch = cut_words[inx - offset - 1]
            vec = self.char2vec.char2vec(ch)
            inputs.append(vec)
            short_text.append(ch)

        inputs.reverse()
        short_text.reverse()

        for offset in range(self.half_size):
            ch = cut_words[inx + offset + 1]
            vec = self.char2vec.char2vec(ch)
            inputs.append(vec)
            short_text.append(ch)

        ch = cut_words[inx]
        vec = self.char2vec.char2vec(ch)
        return [inputs, vec, short_text, ch]


def test1():
    objCutChar = CutChar()
    # objChar2Vec = Char2Vec(restart_info=True)
    objChar2Vec = Char2Vec(restart_info=False)

    text = "一鿿我哈哈；哈 哈123##english sb 123ｩｰab哈 123ｼ"
    cut_words = objCutChar.cut(text)
    for ch in cut_words:
        objChar2Vec.char2vec(ch)

    objChar2Vec.dump_info()


def test2():
    objCutChar = CutChar()
    # objChar2Vec = Char2Vec(restart_info=True)
    objChar2Vec = Char2Vec(restart_info=False)

    df = pd.read_csv("../new_reviews/experience/cut_new_pcid4cid50228001.cvs", encoding="utf_8_sig")
    texts = map(str, df["comment_all"].values)
    size = len(df)
    del df
    print("has comments", size)
    for seq, text in enumerate(texts, start=1):
        cut_words = objCutChar.cut(text)
        for ch in cut_words:
            objChar2Vec.char2vec(ch)
        if 0 == seq % 1e5:
            print(f"completed {seq}/{size}")

    objChar2Vec.dump_info()


def test3():
    objCutChar = CutChar()
    # objChar2Vec = Char2Vec(restart_info=True)
    objChar2Vec = Char2Vec(restart_info=False)


if __name__ == '__main__':
    # test1()
    # test2()
    test3()
