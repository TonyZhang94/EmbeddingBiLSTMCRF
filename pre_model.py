# -*- coding: utf-8 -*-

import tensorflow as tf

from pre_utils import Char2Vec


class PredictModel(object):
    def __init__(self, **kwargs):
        self.holder_dim = kwargs["holder_dim"]
        self.win_size = kwargs["win_size"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.batch_size = kwargs["batch_size"]
        self.epoch_num = kwargs["epoch_num"]
        self.lr = kwargs["lr"]
        self.clip_grad = kwargs["clip_grad"]

        self.init_embedding = kwargs["init_embedding"]

        self.summaries_path = kwargs["summaries_path"]
        self.checkpoints_path = kwargs["checkpoints_path"]

        self.char2vec = Char2Vec()

        self.build()

    def build(self):
        self.add_placeholders()
        self.encoder_layer_op()
        self.decoder_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.train_step_op()

    def add_placeholders(self):
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.win_size, self.holder_dim], name="inputs")
        self.labels = tf.placeholder(tf.float32, shape=[None, self.holder_dim], name="labels")

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

            inputs = tf.reshape(self.inputs, [-1, self.holder_dim])
            self.shape_reshape_inputs = tf.shape(inputs)

            output = tf.matmul(inputs, W) + b
            self.shape_output1 = tf.shape(output)

            output = tf.reshape(output, [-1, self.win_size, self.hidden_dim])
            self.shape_output2 = tf.shape(output)

            self.hidden_layer = tf.reduce_sum(output, 1)
            self.shape_hidden_layer = tf.shape(self.hidden_layer)

    def decoder_layer_op(self):
        with tf.variable_scope("Decoder"):
            W = tf.get_variable(name="W",
                                shape=[self.hidden_dim, self.holder_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.holder_dim],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            self.output_layer = tf.matmul(self.hidden_layer, W) + b
            self.shape_output_layer = tf.shape(self.output_layer)

    def softmax_pred_op(self):
        self.logits_softmax = tf.nn.softmax(self.output_layer)

    def loss_op(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits_softmax)
        self.loss = tf.reduce_mean(self.loss)

    def train_step_op(self):
        with tf.variable_scope("Train"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            # optim = tf.train.AdamOptimizer(learning_rate=self.lr)
            # grads_and_vars = optim.compute_gradients(self.loss)
            # grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            # self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)

    def add_summary(self, sess):
        self.file_writer = tf.summary.FileWriter(self.summaries_path)
        self.file_writer.add_graph(sess.graph)
        tf.summary.scalar("loss", self.loss)
        self.merged = tf.summary.merge_all()

    def train(self, train, dev):
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            self.add_summary(sess)
            for epoch in range(self.epoch_num):
                print(f"\n=============== epoch {epoch+1} ===============")
                self.run_one_epoch(sess, train, dev, epoch, saver)

    def run_one_epoch(self, sess, train, dev, epoch, saver):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        step_num = 0
        for cur_batch, feed_dict in enumerate(self.get_feed_dict(train, num_batches), start=1):
            _, loss_train, summary, step_num = sess.run(
                [self.train_op, self.loss, self.merged, self.global_step],
                feed_dict=feed_dict)
            print(f"cur batch {cur_batch}/{num_batches}, global step {step_num}")
            self.file_writer.add_summary(summary, global_step=step_num)

        saver.save(sess, self.checkpoints_path, global_step=step_num)

    def get_feed_dict(self, data, num_batches):
        for cur_batch in range(num_batches):
            inputs, labels = list(), list()
            inx = cur_batch * self.batch_size
            end = min(inx + self.batch_size, len(data))
            while inx < end:
                one_input = list()
                for ch in data[inx][0]:
                    try:
                        vec = self.init_embedding[ch]
                    except KeyError:
                        vec = self.char2vec.char2vec(ch)
                        self.init_embedding[ch] = vec
                    one_input.append(vec)
                inputs.append(one_input)
                try:
                    vec = self.init_embedding[data[inx][1]]
                except KeyError:
                    ch = data[inx][1]
                    vec = self.char2vec.char2vec(ch)
                    self.init_embedding[ch] = vec
                labels.append(vec)
                inx += 1
            yield {self.inputs: inputs, self.labels: labels}
            del inputs
            del labels


class EncoderModel(object):
    def __init__(self):
        self.build()

    def build(self):
        pass


if __name__ == '__main__':
    x = tf.constant([[[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]],
                     [[3, 4, 3, 4, 3, 4], [3, 4, 3, 4, 3, 4], [3, 4, 3, 4, 3, 4]],
                     [[5, 6, 5, 6, 5, 6], [5, 6, 5, 6, 5, 6], [5, 6, 5, 6, 5, 6]],
                     [[7, 8, 7, 8, 7, 8], [7, 8, 7, 8, 7, 8], [7, 8, 7, 8, 7, 8]]], dtype=tf.float32)
    shape1 = tf.shape(x)
    output = tf.reduce_sum(x, 1)
    shape2 = tf.shape(output)

    softmax = tf.nn.softmax(output)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        shape1, output, shape2, softmax = sess.run([shape1, output, shape2, softmax])
        print(shape1)
        print(output)
        print(shape2)
        print(softmax)
