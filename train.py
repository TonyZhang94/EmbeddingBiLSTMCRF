# -*- coding: utf-8 -*-

import time
import os

from samples import *
from pre_model import PredictModel
from settings import *


def run():
    train, dev, test = load_data()
    with open("char2info/char2vec.pkl", mode="rb") as fp:
        init_embedding = pickle.load(fp)

    timestamp = str(int(time.time()))
    output_path = result_dir + timestamp
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = PredictModel(holder_dim=holder_dim,
                         win_size=win_size,
                         hidden_dim=hidden_dim,
                         batch_size=batch_size,
                         epoch_num=epoch_num,
                         lr=lr,
                         clip_grad=clip_grad,
                         init_embedding=init_embedding,
                         summaries_path=summaries_path.format(timestamp),
                         checkpoints_path=checkpoints_path.format(timestamp)+"model"
                         )
    model.train(train, dev)
    # PredictModel.test(test)

    # model.dump_embedding("char2info/char2vec.pkl")


if __name__ == '__main__':
    run()
