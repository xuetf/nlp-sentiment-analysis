# encoding=utf-8
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc
from constant import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve


def plot_word_cloud(best_words):
    best_words = dict(best_words)
    # Generate a word cloud image
    cloud = WordCloud(font_path=font_path)
    wordcloud = cloud.generate_from_frequencies(best_words)
    # Display the generated image:
    # the matplotlib way:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 10), baseline=None, scoring='f1'):
    """
    Generate a simple plot of the test and traning learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    """

    X, y = shuffle(X, y) # important for logistic because of the parallel modeling
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv = cv, train_sizes=train_sizes, scoring=scoring, n_jobs=1) # neg=1的f1 score

    print ('cross f1 scores of different training size for training data:')
    print (train_scores)

    print ('f1 scores of different training size for test data:')
    print (test_scores)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")
    if baseline:
        plt.axhline(y=baseline, color='red', linewidth=5, label='Desired Performance')  # baseline
    plt.xlabel("Training examples")
    plt.ylabel("%s Score"  %scoring)
    plt.legend(loc="best")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()

def plot_validation_curve(estimator,title, X, y,
                          param_name, param_range, param_plot_range,
                          x_ticks=np.arange(0.5, 2.10, 0.1), y_ticks=np.arange(0.8, 0.95, 0.01), cv=None, scoring='f1'):
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1)  # 固定种子
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=kf, scoring=scoring, n_jobs=1)
    plt.figure()

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print (test_scores_mean)
    plt.title(title)
    plt.xlabel(param_name)
    if isinstance(scoring,str):
        plt.ylabel("%s Score" %scoring)
    else:
        plt.ylabel("F1 Score")


    plt.semilogx(param_plot_range, train_scores_mean, label="Training score",
                 color="r")
    plt.fill_between(param_plot_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    plt.semilogx(param_plot_range, test_scores_mean, label="Cross-validation score",
                 color="b")
    plt.fill_between(param_plot_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="b")

    plt.yticks(y_ticks)
    plt.xticks(x_ticks,rotation=90)
    plt.xlim((0, 2.0))

    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.legend(loc="best")
    plt.show()

    best =  param_plot_range[np.argmax(test_scores_mean)]
    print (best)
    return best

def plot_precision_recall_curve(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=1) # Use the first fold to draw the curve
    # Create a simple classifier
    classifier.fit(X_train, y_train)
    y_score = classifier.decision_function(X_test)
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, y_score)
    # print('Average precision-recall score: {0:0.2f}'.format(
    #     average_precision))
    precision, recall, _ = precision_recall_curve(y_test, y_score, pos_label=1)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve of neg comment')
    plt.show()


def plot_compare_learning_curve(estimators, X, y, title="Compare Learning curve of Different Model", ylim=(0.7, 1.1), cv=5,
                        train_sizes=np.linspace(.1, 1.0, 5), baseline=0.85, scoring='f1'):
    X, y = shuffle(X, y)  # important for logistic because of the parallel modeling
    plt.figure()
    colors = np.array(['r', 'b', 'k', 'g', 'm', 'pink', 'purple', 'orange'])
    i = 0
    for name in estimators:
        estimator = estimators[name]
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, n_jobs=1, random_state=42)  # neg=1的f1 score
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, test_scores_mean, 'o-', color=colors[i],
             label=name)
        i+=1

    if baseline:
        plt.axhline(y=baseline, color='red', linewidth=5, label='Desired Performance')  # baseline
    plt.xlabel("Training examples")
    plt.ylabel("Cross-Validation %s Score of Negative Comment" % scoring)
    plt.legend(loc="best")
    plt.grid("on")

    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()