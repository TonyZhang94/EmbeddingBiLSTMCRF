import tensorflow as tf
import numpy as np
import os, argparse, time, random
from char_model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from char_data import read_corpus, read_dictionary, tag2label, random_embedding, bert_embedding
import datetime

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
# parser.add_argument('--hidden_dim', type=int, default=768, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random',
                    help='use pretrained char embedding or init it randomly')
# parser.add_argument('--pretrain_embedding', type=str, default='bert',
#                     help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
# parser.add_argument('--embedding_dim', type=int, default=768, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
# parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
parser.add_argument('--demo_model', type=str, default='1552023653', help='model for test and demo')
args = parser.parse_args()

# get char embeddings
# word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
word2id = read_dictionary(os.path.join('.', args.test_data, 'word2id.pkl'))
if args.pretrain_embedding == 'random':
    print("random init word2id.pkl")
    embeddings = random_embedding(word2id, args.embedding_dim)
elif args.pretrain_embedding == 'bert':
    print("use bert embedding word2id")
    embeddings = bert_embedding(word2id, args.embedding_dim)
    print("bert embedding num is", len(embeddings))
else:
    # try this
    print("use pretrain_embedding.npy embedding word2id")
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')

# read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    test_size = len(test_data)

# paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
# timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_path = os.path.join('.', args.train_data + "_save", timestamp)
if not os.path.exists(output_path):
    os.makedirs(output_path)

summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path):
    os.makedirs(summary_path)

model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path):
    os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix

result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path):
    os.makedirs(result_path)

log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path

get_logger(log_path).info(str(args))

# training model
if args.mode == 'train':
    from samples import *
    from pre_utils import Char2Vec
    from settings import *
    from encoder import *

    with open("char2info/char2vec.pkl", mode="rb") as fp:
        init_embedding = pickle.load(fp)

    char_timestamp = "1559804726"
    char_model = EncoderModel(holder_dim=holder_dim,
                         hidden_dim=hidden_dim,
                         batch_size=batch_size,
                         init_embedding=init_embedding,
                         checkpoints_path=checkpoints_path.format(char_timestamp)
                         )

    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, char_model=char_model, config=config)
    model.build_graph()

    # hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    # train model on the whole training data
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

# testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    """
    embeddings: get from func: random_embedding or load from pretrain_embedding.npy
    tag2label: tag2label = { "O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6 }
    word2id: word2id 里的 3905 个汉字
    paths:
    config: tensorflow 内置，Session中使用，配置一些硬件使用，对逻辑没有影响
    """
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    start = time.time()
    model.test(test_data)
    print("cost:", time.time()-start)

# demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while (1):
            print('输入待识别句子:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                # 邱实在实验室买戴尔的显示屏
                # ['邱', '实', '在', '实', '验', '室', '买', '戴', '尔', '的', '显', '示', '屏']
                demo_sent = list(demo_sent.strip())
                # [(
                # ['邱', '实', '在', '实', '验', '室', '买', '戴', '尔', '的', '显', '示', '屏'],
                # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
                # )]
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                # ['B-PER', 'I-PER', 0, 0, 0, 0, 0, 'B-PER', 'I-PER', 0, 0, 0, 0]
                tag = model.demo_one(sess, demo_data)
                # PER, LOC, ORG = get_entity(tag, demo_sent)
                targets = get_entity(tag, demo_sent)
                print('属性词: {}'.format(targets))


# 2019.03.07 15:45 start train
