# -*- coding: utf-8 -*-
import numpy as np
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from constant import *
from visualizer import *
from util import *

def chi_features(number, pos_comments, neg_comments, is_plot=False, is_save=False):
    '''获取信息量最高(前number个)的特征(卡方统计)'''
    pos_words = np.concatenate(pos_comments)  ##集合的集合展平成一个集合
    neg_words = np.concatenate(neg_comments)

    word_fd = FreqDist()  # 可统计所有词的词频
    cond_word_fd = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频

    for word in pos_words:
        word_fd[word] += 1
        cond_word_fd[pos][word] += 1

    for word in neg_words:
        word_fd[word] += 1
        cond_word_fd[neg][word] += 1

    pos_word_count = cond_word_fd[pos].N()  # 积极词的数量

    neg_word_count = cond_word_fd[neg].N()  # 消极词的数量

    total_word_count = pos_word_count + neg_word_count

    word_scores = {}  # 包括了每个词和这个词的信息量

    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd[pos][word], (freq, pos_word_count),
                                               total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量

        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd[neg][word], (freq, neg_word_count),
                                               total_word_count)  # 同理

        word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量

    # 把词按信息量倒序排序
    best_features = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[:number]


    if is_plot: plot_word_cloud(best_features)
    if is_save: dump_to_pickle(dir_path, 'chi_features', best_features)
    return dict(best_features)