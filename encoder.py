# -*- coding: utf-8 -*-

import tensorflow as tf

from samples import *
from pre_utils import Char2Vec
from settings import *
# from BiLSTM_CRF.char_data import batch_yield


class EncoderModel(object):
    def __init__(self, **kwargs):
        self.holder_dim = kwargs["holder_dim"]
        # self.win_size = kwargs["win_size"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.batch_size = kwargs["batch_size"]

        self.init_embedding = kwargs["init_embedding"]

        self.checkpoints_path = kwargs["checkpoints_path"]

        self.char2vec = Char2Vec()

        self.build()

    def build(self):
        self.add_placeholders()
        self.encoder_layer_op()

    def add_placeholders(self):
        self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.holder_dim], name="inputs")
        # self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

    def encoder_layer_op(self):
        with tf.variable_scope("Encoder"):
            W = tf.get_variable(name="W",
                                shape=[self.holder_dim, self.hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.hidden_dim],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            origin_shape = tf.shape(self.inputs)
            sequence_length = origin_shape[1]

            inputs = tf.reshape(self.inputs, [-1, self.holder_dim])
            self.shape_reshape_inputs = tf.shape(inputs)

            output = tf.matmul(inputs, W) + b
            self.shape_output = tf.shape(output)

            self.embedding = tf.reshape(output, [-1, sequence_length, self.hidden_dim])
            self.shape_embedding = tf.shape(self.embedding)

    def get_feed_dict(self, data, num_batches):
        for cur_batch in range(num_batches):
            # inputs, labels = list(), list()
            inputs = list()
            # sequence_lengths = list()
            inx = cur_batch * self.batch_size
            end = min(inx + self.batch_size, len(data))
            while inx < end:
                one_input = list()
                for ch in data[inx]:
                    try:
                        vec = self.init_embedding[ch]
                    except KeyError:
                        vec = self.char2vec.char2vec(ch)
                        self.init_embedding[ch] = vec
                    one_input.append(vec)
                inputs.append(one_input)
                # sequence_lengths.append(len(data[inx]))
                # try:
                #     vec = self.init_embedding[data[inx][1]]
                # except KeyError:
                #     ch = data[inx][1]
                #     vec = self.char2vec.char2vec(ch)
                #     self.init_embedding[ch] = vec
                # labels.append(vec)
                inx += 1
            # yield {self.inputs: inputs, self.labels: labels}
            # yield {self.inputs: inputs, self.sequence_lengths: sequence_lengths}
            yield {self.inputs: inputs}
            del inputs
            # del labels

    def encoder(self, test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt_file = tf.train.latest_checkpoint(self.checkpoints_path)
            print(ckpt_file)
            saver.restore(sess, ckpt_file)

            num_batches = (len(test) + self.batch_size - 1) // self.batch_size
            for cur_batch, feed_dict in enumerate(self.get_feed_dict(test, num_batches), start=1):
                # embedding = sess.run([self.inputs], feed_dict=feed_dict)
                # shape_output, embedding, shape_embedding = sess.run(
                #     [self.shape_output, self.embedding, self.shape_embedding],
                #     feed_dict=feed_dict)
                # print(shape_output)
                # print(shape_embedding)
                embedding = sess.run([self.embedding], feed_dict=feed_dict)


if __name__ == '__main__':
    # train, dev, test = load_data()
    texts = ["这是一个测试！",
             "这是另一个测试！",
             "啦啦啦啦",
             "我很长，一大段的测试文本。重复一遍，我很长，我是一大段文本！！！"]
    length = 0
    for text in texts:
        length = max(length, len(text))

    obj_cut_char = CutChar()
    test = [obj_cut_char.cut(text) for text in texts]
    for inx in range(len(test)):
        while len(test[inx]) < length:
            test[inx].append("<BLANK>")

    for line in test:
        print(line)

    # sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

    with open("char2info/char2vec.pkl", mode="rb") as fp:
        init_embedding = pickle.load(fp)

    timestamp = "1559804726"
    model = EncoderModel(holder_dim=holder_dim,
                         hidden_dim=hidden_dim,
                         batch_size=batch_size,
                         init_embedding=init_embedding,
                         checkpoints_path=checkpoints_path.format(timestamp)
                         )
    model.encoder(test)
