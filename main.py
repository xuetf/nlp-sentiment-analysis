# -*- coding: utf-8 -*-
from comment_classcifier import *


def compare_models(data):
    '''对比模型, 使用不同模型进行交叉验证并绘制学习曲线'''
    models= {"logisticRegression": LogisticRegression(),
            "Perceptron":Perceptron(),
            "DecisionTree":DecisionTreeClassifier(),
            "GBDT":GradientBoostingClassifier(),
             "SVC":svm.LinearSVC(),
             "Multinomial Bayes":MultinomialNB()
            }
    for name in models:
         print (name, " begin...")
         cross_validate_score(data, k_fold=5, model=models[name], n=1000, sparse=True)

    _, data_features = fit_preprocess(train_data=data, is_need_cut=False, n=1000,
                                      is_load_from_file=False)
    X, y = transform_features(data_features)
    plot_compare_learning_curve(models, X, y)

def read_data():
    good = pd.read_csv(good_comment_path, sep='\n', header=None, names=['comment'])
    bad = pd.read_csv(bad_comment_path, sep='\n', header=None, names=['comment'])
    good[label_name] = pos
    bad[label_name] = neg
    return pd.concat([good, bad], axis=0, ignore_index=True).reset_index(drop=True)


'''
main方法，
1.首先进行模型对比，得到最优分类器之一LogisticRegression
2.再进一步考察LR学习过程， 包括LR参数选择，学习曲线绘制，PR曲线绘制，交叉验证，最终训练模型并得到测试集的预测结果
这里注释了中间过程，运行哪个就把注释去掉
'''
if __name__ == '__main__':
    # 加载数据并进行分词处理
    data = read_data()
    data[comment_name] = cut_comments(data[comment_name], is_load_from_file=False, name=all_word_cut_name)  # 统一切词

    model = MultinomialNB() # LogisticRegression()

    # 交叉验证绘制学习曲线
    learning_curve(data, model,scoring='f1', classifier_name="Multinomial Bayes")


    # 交叉验证，输出评价指标结果
    cross_validate_score(data, k_fold=5, model=NaiveBayesClassifier, n=2000, is_nltk_model=True) # 可以输出有用的特征
    # cross_validate_score(data, k_fold=5, model=model, n=2000, is_nltk_model=False)


    # 对比实验：不同模型交叉验证/学习曲线
    compare_models(data)


    # 绘制precision_recall曲线
    # precision_recall_curve(data, LogisticRegression(class_weight={pos:best_pos_class_weight, neg:1.0}))

    # 调参
    # adjust_parameter_validate_curve(data, LogisticRegression(), 'LogisticRegression Validation')


