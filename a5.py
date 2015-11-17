# a5.py - 
# by K. Brett Mulligan
# CSU - CS545 (Fall 2015)
# Assignment 5
# This assignment explores feature selection using the Arcene and leukemia datasets

from __future__ import division

import math
import time
import numpy as np
from matplotlib import pyplot as plt

from sklearn import cross_validation, svm, metrics, datasets
from sklearn import pipeline
from sklearn import preprocessing as pp
from sklearn import decomposition as decomp
from sklearn import feature_selection as fs

arcene_data_filename = 'arcene_tv.data'
arcene_labels_filename =  'arcene_tv.labels'
leukemia_filename = 'leu.both'

complete = False

def separate(X, y) :
    """ Returns indices of all positives examples, and indices of all negative examples in X according to labels y. """
    
    pos = []
    neg = []

    for i in range(X.shape[0]):         # iterate over all examples
        if y[i] > 0:
            pos.append(i)
        elif y[i] < 0:
            neg.append(i)
        else:
            print 'ERROR: Invalid label'

    return pos, neg

def golub_for_feature(X, y, i, pos=[], neg=[]) :
    """ Return the Golub score of feature i for labeled dataset X,y. """

    if (pos == [] or neg == []):
        pos, neg = separate(X, y)

    # print pos

    i_positives = np.array([X[j][i] for j in pos])
    i_negatives = np.array([X[j][i] for j in neg])

    # print X[0]

    i_positives = np.array([])
    for j in pos:
        i_positives = np.append(i_positives, X[j][i])
    
    # print i_positives

    # print 'Pos shape:', i_positives.shape
    # print 'Neg shape:', i_negatives.shape

    avg_pos = np.average(i_positives)
    avg_neg = np.average(i_negatives)

    sd_pos = np.std(i_positives)
    if sd_pos == 0:
        sd_pos = 1

    sd_neg = np.std(i_negatives)
    if sd_neg == 0:
        sd_neg = 1

    score = np.absolute(avg_pos - avg_neg) / (sd_pos + sd_neg)

    return score

def golub(X, y) :
    """ Return the Golub score of a labeled dataset. Returns vector, scores, of length 
        equal to number of features in X. """

    scores = np.array([])

    positives, negatives = separate(X, y)

    for i in range(X.shape[1]) :
        scores = np.append(scores, golub_for_feature(X, y, i, positives, negatives))

    return scores, scores

def load_arcene_data() :
    print 'Loading Arcene data...'
    X_arc = np.genfromtxt(arcene_data_filename)
    y_arc = np.genfromtxt(arcene_labels_filename)

    print '(Examples, features)'
    print X_arc.shape
    print len(y_arc), 'labels' 
    print '(Examples, features) in first example'
    print X_arc[0].shape

    return X_arc, y_arc

def load_leukemia_data() :
    print 'Loading leukemia data...'
    X_leu, y_leu = datasets.load_svmlight_file(leukemia_filename)

    X_leu = X_leu.toarray()

    print '(Examples, features)'
    print X_leu.shape
    print len(y_leu), 'labels' 
    print '(Examples, features) in first example'
    print X_leu[0].shape

    return X_leu, y_leu

def test_golub (X_arc, y_arc, X_leu, y_leu) :

    print '\nTesting separate...'
    pos_arc, neg_arc = separate(X_arc, y_arc)

    assert(len(X_arc) == len(pos_arc) + len(neg_arc))
    print len(X_arc), len(pos_arc), len(neg_arc)

    pos_leu, neg_leu = separate(X_leu, y_leu)

    assert(X_leu.shape[0] == len(pos_leu) + len(neg_leu))
    print X_leu.shape[0], len(pos_leu), len(neg_leu)


    print '\nTesting golub score...'

    print X_arc
    print 'Feature 0...'
    print X_arc.T[0], len(X_arc.T[0])

    X_pos_arc = np.array([X_arc[p] for p in pos_arc])
    X_neg_arc = np.array([X_arc[n] for n in neg_arc])

    print len(X_arc), len(X_pos_arc), len(X_neg_arc)

    assert(len(X_arc) == len(X_pos_arc) + len(X_neg_arc)) 

    print 'Means:', np.mean(X_arc.T[0]), np.mean(X_pos_arc.T[0]), np.mean(X_neg_arc.T[0])
    print 'StdDevs:', np.std(X_arc.T[0]), np.std(X_pos_arc.T[0]), np.std(X_neg_arc.T[0])
    test_golub = ( np.mean(X_pos_arc.T[0]) - np.mean(X_neg_arc.T[0]) ) / ( np.std(X_pos_arc.T[0]) + np.std(X_neg_arc.T[0]) )
    print 'Golub[0]:', test_golub

    assert(golub_for_feature(X_arc, y_arc, 0, pos_arc, neg_arc) == test_golub)
    print 'Golub passes test!'


    ############# COMPUTATIONAL EXPENSIVE --- COMMENT OUT UNTIL END #####################

    if complete == True:
        print '\nCalculating Golub scores...'
        golubs_arc = golub(X_arc, y_arc)[0]
        print ''
        print 'Golubs(Arcene):'
        print golubs_arc
        print len(golubs_arc)
        assert(len(golubs_arc) == len(X_arc.T))

        golubs_leu = golub(X_leu, y_leu)[0]
        print ''
        print 'Golubs(leukemia):'
        print golubs_leu
        print len(golubs_leu)
        assert(len(golubs_leu) == len(X_leu.T))

        print 'Golub scores calculated!'

def get_nonzeros (X,y):

    iterations = 1
    nonzeros = []
    for i in range(iterations):

        svm_l1 = svm.LinearSVC(penalty='l1', dual=False)
        classify = svm_l1.fit(X, y)

        # print classify.coef_
        # print 'Nonzeros:', classify.coef_[0].nonzero()[0]
        nonzeros.append(len(classify.coef_[0].nonzero()[0]))
        print 'Nonzero elements:', nonzeros[-1]

    return np.mean(nonzeros)


def test_svm (Xa, ya, Xl, yl) :

    folds = 5
    step_delta = 0.05


    print '\nTesting L1 SVM...'

    arc_nnz = get_nonzeros(Xa, ya)
    print 'ARCENE   - Average number of non-zero elements:', arc_nnz

    leu_nnz = get_nonzeros(Xl, yl)
    print 'LEUKEMIA - Average number of non-zero elements:', leu_nnz

    

    # setup cross validation folds
    cross_val = cross_validation.StratifiedKFold(ya, folds, shuffle=True)

    svm_l1 = svm.LinearSVC(penalty='l1', dual=False)
    svm_l2 = svm.LinearSVC(penalty='l2', dual=False)

    rfe = fs.RFECV(svm_l2, step=step_delta, cv=folds)

    # setup pipes and estimators
    estimators0 = [('svm1', svm_l1)]
    estimators1 = [('svm1', svm_l1), ('svm2', svm_l2)]
    estimators2 = [('svm2', svm_l2)]
    estimators3 = [('rfe', rfe)]

    pipes = [pipeline.Pipeline(estimators0), pipeline.Pipeline(estimators1), pipeline.Pipeline(estimators2), pipeline.Pipeline(estimators3)]

    if complete == True:
        print '\nARCENE RESULTS................'
        for pipe in pipes:

            # CHECK SCORES
            cv_scores = cross_validation.cross_val_score(pipe, Xa, ya, cv=folds)

            print 'Cross val scores for', folds, 'folds:'
            print np.mean(cv_scores), cv_scores 

        print '\nLEUKEMIA RESULTS................'
        for pipe in pipes:

            # CHECK SCORES
            cv_scores = cross_validation.cross_val_score(pipe, Xl, yl, cv=folds)

            print 'Cross val scores for', folds, 'folds:'
            print np.mean(cv_scores), cv_scores

        print 'Cross validation for SVMs complete!'

    return

def test_subsample_method(Xa, ya, Xl, yl, subs=50):

    Xs, ys = k_subsamples(Xa, ya, subs)

    classifiers = []
    coefs = []

    for X, y in zip(Xs, ys):
        classifiers.append(train_svm(X, y))
        coefs.append(classifiers[-1].coef_[0])


    print len(coefs)
    print len(coefs[0]), len(coefs[-1])
    print coefs[0], coefs[-1]

    scores = get_scores(coefs)

    print scores, len(scores)

    return scores


def k_subsamples(X, y, k):

    X_subs = []
    y_subs = []

    for i in range(k):
        newX, newy = rand_subsample(X, y)
        X_subs.append(newX)
        y_subs.append(newy)

    assert(len(X_subs) == k)
    assert(len(y_subs) == k)

    return X_subs, y_subs

def rand_subsample(X, y, frac=0.8):
    """ Returns random subsample X_sub, y_sub with only fraction 'frac' of the original examples."""

    assert(frac <= 1.0)

    total = len(X)

    keep = int(np.floor(total * frac))

    # print 'Total:', total
    # print 'Keep:', keep

    X_sub = X[:]
    y_sub = y[:]

    np.random.shuffle(X_sub)
    np.random.shuffle(y_sub)

    X_sub = X_sub[:keep]
    y_sub = y_sub[:keep]

    return X_sub, y_sub


def train_svm(X,y):

    svm_l1 = svm.LinearSVC(penalty='l1', dual=False)
    classifier = svm_l1.fit(X,y)
    
    return classifier

def get_scores(coefs):

    scores = np.zeros(len(coefs[0]))

    # print len(scores)

    for w in coefs:
        indices = np.nonzero(w)

        for i in indices:
            scores[i] += 1

    return scores



if __name__ == '__main__':
    print 'Testing...a5.py'

    # load Arcene data
    X_arc, y_arc = load_arcene_data()

    # load leukemia data
    X_leu, y_leu = load_leukemia_data()

    # test golub
    test_golub(X_arc, y_arc, X_leu, y_leu)


    # do L1 svm
    test_svm(X_arc, y_arc, X_leu, y_leu)



    subs = 25
    test_subsample_method(X_arc, y_arc, X_leu, y_leu, subs)

