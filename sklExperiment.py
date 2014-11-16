# -*- coding: utf-8 -*-
"""
An example script for training & testing a classifier using LIBLINEAR
Copyright Matthew Purver 2014
"""

from collections import deque
from itertools import chain
from unicodecsv import DictReader

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.svm.libsvm import cross_validation

START_SYMBOL = '__START__'
END_SYMBOL = '__END__'

def getTweets(csvfile='data.csv'):
    """
    Get some data from a CSV file source
    """
    with open(csvfile, 'rb') as f:
        reader = DictReader(f, encoding='utf-8')
        labelKey = 'label'
        textKey = 'text'
        # some files look a bit different ...
        if labelKey not in reader.fieldnames:
            labelKey = 'Rating'
            textKey = 'Tweet'
        for line in reader:
            if labelKey in line and line[labelKey] and line[labelKey].lstrip('-').isdigit():
                yield line[labelKey], line[textKey]

                
def getVectors(labelTextPairs, labels=None):
    """
    Turn iterable label-text pair sequences into sklearn feature vector and label arrays
    """
    y = []
    x = []
    y_values = {}
    for (label,text) in labelTextPairs:
        if labels and float(label) not in labels:
            continue
        y.append(float(label))
        x.append(START_SYMBOL + " " + text + " " + END_SYMBOL)
        try:
            y_values[float(label)] += 1
        except KeyError:
            y_values[float(label)] = 1
    print "read %s datapoints, with these labels: %s" % (len(y),y_values)
    vectorizer = CountVectorizer(min_df=1, 
                                 ngram_range=(1,3), 
                                 token_pattern=u'(?u)\\b\\w+\\b', 
                                 binary=False)
    X = vectorizer.fit_transform(x)
    #print vectorizer.get_feature_names()
    X = TfidfTransformer(use_idf=False).fit_transform(X)
    # X = MinMaxScaler().fit_transform(X)
    #print X.toarray()
    return X, y


def crossValidate(X, y, nfolds=10, cost=None):
    """
    Run a k-fold (default 10) cross-validation using a linear SVM
    """
    svm = LinearSVC(C=cost) if cost else LinearSVC()
    #svm.fit(X, y)
    acc = cross_val_score(svm, X, y, cv=5)
    print "Cross-validation accuracy %s: %s" % (np.mean(acc),acc)

    # compare to LibSVM
    acc = cross_val_score(SVC(kernel='rbf', C=cost) if cost else SVC(kernel='linear'), X, y, cv=5)
    print "Cross-validation accuracy %s: %s" % (np.mean(acc),acc)

    # # there is a low-level LibSVM x-val function too, is it faster?
    # acc = cross_validation(X.toarray(), np.array(y), 5)
    # print "Cross-validation accuracy %s" % (acc)

    return svm


if __name__ == "__main__":

    X, y = getVectors(getTweets(), labels=[-1,1])
    crossValidate(X, y, cost=3)
