# -*- coding: utf-8 -*
# import re

# import numpy as np
# import matplotlib.pyplot as plt
#
# """ 线条颜色
# b: blue
# g: green
# r: red
# c: cyanm: magenta
# y: yellow
# k: black
# w: white
# """
#
# """ 线条参数ls可取值为：
# -      实线(solid)
# --     短线(dashed)
# -.     短点相间线(dashdot)
# ：    虚点线(dotted)
# '', ' ', None
# """
#
# """ 节点参数
# marker 形状
# mec 颜色
# mew 折点线宽
# mfc w 空心 默认实心
# mfcalt
# ms 节点大小
# """
#
# import os
# import time
#
#
# fig_title = ""
#
# linetypes = list()
# linetypes.append("r-")
# linetypes.append("r:")
# linetypes.append("g-")
# linetypes.append("g:")
#
# # seq = "1521112368"  # one_hot
# # seq = "1551944747"  # train 1
# seq = "1552023653"  # bert + CTR
# path = "data_path_save/" + seq + "/results/result_metric_"
#
#
# # token_num = list()
# # phrases_all_num = list()
# # phrases_find_num = list()
# # phrases_correct_num = list()
# # phr_acc, phr_pre, phr_rec, phr_f1 = list(), list(), list(), list()
# # all_acc, all_pre, all_rec, all_f1 = list(), list(), list(), list()
# # loc_pre, loc_rec, loc_f1, loc_num = list(), list(), list(), list()
# # org_pre, org_rec, org_f1, org_num = list(), list(), list(), list()
# # per_pre, per_rec, per_f1, per_num = list(), list(), list(), list()
# data = dict()
#
# rank = 1
# pattern = re.compile(r"\d+\.*\d+")
# while True:
#     file = path + str(rank)
#     rank += 1
#     try:
#         with open(file, mode="r") as fp:
#             text = ",".join(fp.readlines())
#             res = pattern.findall(text)
#             res = [float(x) for x in res]
#             data.setdefault("token_num", list()).append(res[0])
#
#             num, find, correct = res[1], res[2], res[3]
#             TP = correct
#             FP = find - correct
#             FN = num - correct
#             try:
#                 pre = TP / (TP + FP)
#             except ZeroDivisionError:
#                 pre = 0
#             try:
#                 rec = TP / (TP + FN)
#             except ZeroDivisionError:
#                 rec = 0
#             try:
#                 f1 = (2 * pre * rec) / (pre + rec)
#             except ZeroDivisionError:
#                 f1 = 0
#             # phr_acc.append(res[0])
#             data.setdefault("phr_pre", list()).append(pre)
#             data.setdefault("phr_rec", list()).append(rec)
#             data.setdefault("phr_f1", list()).append(f1)
#
#             data.setdefault("all_acc", list()).append(res[4])
#             data.setdefault("all_pre", list()).append(res[5])
#             data.setdefault("all_rec", list()).append(res[6])
#             data.setdefault("all_f1", list()).append(res[7])
#
#             data.setdefault("loc_pre", list()).append(res[8])
#             data.setdefault("loc_rec", list()).append(res[9])
#             data.setdefault("loc_f1", list()).append(res[10])
#             data.setdefault("loc_num", list()).append(res[11])
#
#             data.setdefault("org_pre", list()).append(res[12])
#             data.setdefault("org_rec", list()).append(res[13])
#             data.setdefault("org_f1", list()).append(res[14])
#             data.setdefault("org_num", list()).append(res[15])
#
#             data.setdefault("per_pre", list()).append(res[16])
#             data.setdefault("per_rec", list()).append(res[17])
#             data.setdefault("per_f1", list()).append(res[18])
#             data.setdefault("per_num", list()).append(res[19])
#     except (FileNotFoundError, FileExistsError) as e:
#         break
#     except Exception as e:
#         print(e)
#         raise
#
#
# def build_all():
#     dst_path = "graph/{}/".format(seq)
#     try:
#         os.mkdir(dst_path)
#     except FileExistsError:
#         pass
#     except Exception as e:
#         print(e)
#         raise
#
#     for name, y in data.items():
#         x = list(range(1, rank-1))
#         print(name, y)
#         plt.plot(x, y, label=name)
#
#         plt.legend()
#         plt.title(fig_title)
#         plt.savefig(dst_path+name+".png")
#         plt.close()
#         # plt.show()
#
#         # for y, linetype, fname in zip(ys, linetypes, fnames):
#         #     plt.plot(x, y, linetype, label=fname.split("/")[1].replace(".txt", "").replace("_train", "").replace("zhibiao", "baseline"))
#
#
# if __name__ == '__main__':
#     build_all()
