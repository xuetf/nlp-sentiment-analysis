# -*- coding: utf-8 -*-
import jieba
import io
from feature_extraction import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


def cut_comments(comments, name=None, is_load_from_file=True):
    if is_load_from_file and is_exist_file(dir_path, name):
        return load_from_pickle(dir_path, name)
    cut_comments_list = []
    stop = [line.strip() for line in io.open(stop_path, 'r', encoding='utf-8').readlines()]  # 停用词
    for i in range(50): stop.append("x" * i) # 脱敏词x去掉
    for comment in comments:
        s = comment.split('\n')
        fenci = jieba.cut(s[0], cut_all=False)  # False默认值：精准模式
        valid_words = list(set(fenci) - set(stop))
        cut_comments_list.append(valid_words)

    if is_load_from_file: dump_to_pickle(dir_path, name, cut_comments_list)
    return cut_comments_list


def word2vec(comments, selected_features, tag = None, is_test_mode=False, tol=1):
    vec = []
    features = selected_features.keys()
    for comment in comments:
        a = {}
        for word in comment:
            if word in features:
                a[word] = 'True'
        if is_test_mode: # 测试模式，不允许贴标签
            vec.append(a)
        elif len(a) >= tol: #训练模式，贴标签。如果该条短信能够提取到有用特征大于阈值tol，则加入训练
            vec.append([a, tag])

    return vec

def construct_vsm_train_features(pos_comments, neg_comments, selected_features):
    '''构建训练数据的向量空间模型'''
    pos_features = word2vec(pos_comments, selected_features, pos)
    neg_features = word2vec(neg_comments, selected_features, neg)
    train_features = np.concatenate([pos_features, neg_features])
    return train_features


def fit_preprocess(train_data, n, is_need_cut, is_load_from_file):
    '''训练前预处理，包括特征提取和特征表示'''
    # cut or not
    if is_need_cut: train_data[comment_name] = cut_comments(train_data[comment_name], all_word_cut_name, is_load_from_file)
    pos_comments = train_data[train_data[label_name] == pos][comment_name].values
    neg_comments = train_data[train_data[label_name] == neg][comment_name].values
    # Feature Selection by Chi-square Test Method
    selected_features = chi_features(n, pos_comments, neg_comments)
    # Construct Train features
    train_features = construct_vsm_train_features(pos_comments, neg_comments, selected_features)
    return selected_features, train_features

def transform_features(data_features):
    X, y = list(zip(*data_features))
    X = DictVectorizer(sparse=True, dtype=float).fit_transform(X)
    y = LabelEncoder().fit_transform(y)
    return X, y






