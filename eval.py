import os


def conlleval(label_predict, label_path, metric_path):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    eval_perl = "./conlleval_rev.pl"
    with open(label_path, mode="w", encoding="utf-8") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                # char = char.encode("utf-8")
                char = char
                # 汉字，标准，预测
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    print("commond is: perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics
