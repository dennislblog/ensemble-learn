
import numpy as np
import sqlite3
from sklearn.model_selection import StratifiedKFold
from pickle import loads, dumps
from Script.metrics.evaluation import Evaluator
from .utils import *
from sklearn.utils import resample
from copy import deepcopy


class Bagging(object):
    """Description

        1. db_file used for local storage of model, CV score, vote(oracal) on pruning data
        2. n_folds internal CV for computing base model's accuracy on the balanced training data
        3. models are the base model from the classifier library (400 models)
    """

    def __init__(self, X=None, y=None, feature=None, fold_id=None, n_folds=5, classifiers=None, conn=None, verbose=False, random_state=None):
        self.X = X
        self.y = y
        self.feature = feature
        self._folds_ = list(StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state).split(X, y))
        self.classifiers = classifiers
        self.conn = conn
        self.id = fold_id

    def run(self, n_bags=100, algorithm=None, **extraArgs):
        ##############################################################################
        # 1) what sampling method applied to train(x)
        func = eval(algorithm) if not algorithm else lambda x, y: (x, y)
        N = len(self.X)
        T = len(self.feature)
        ##############################################################################
        # 2) for each bag, fit an algorithm
        pred_prob, pred_y = np.zeros((n_bags, N)), np.zeros((n_bags, T))
        acc = np.zeros((n_bags,))
        for i in range(n_bags):
            # i) first create a vector recording predprob of each instance (X,y)
            # ---------------------------------------------------------------------- #
            for _, test_index in self._folds_:
                # i) bootstrap training data
                train_index = resample(_)
                # ii) apply data sampling to training data of this fold
                X_train, y_train = func(
                    self.X[train_index, :], self.y[train_index], **extraArgs)
                self.classifiers[i].fit(X_train, y_train)
                # iii) predict testing data of this fold
                pred_prob[i][test_index] = self.classifiers[i].predict_proba(self.X[test_index])[
                    :, 1]
            # ---------------------------------------------------------------------- #
            # ii) evaluate predicted accuracy assuming classifier doesn't bias towards any class
            acc[i] = Evaluator().calculate_accuracy(self.y, pred_prob[i])
            self.classifiers[i].fit(*func(self.X, self.y, **extraArgs))
            pred_y[i] = self.classifiers[i].predict_proba(self.feature)[:, 1]
        with self.conn as conn:
            vals = ((self.id, i, memoryview(dumps(deepcopy(self.classifiers[i]))), algorithm, acc[i], memoryview(dumps(pred_prob[i])), memoryview(dumps(pred_y[i]))) for i in range(n_bags))

            query = """insert into model_info(fold_idx, model_idx, model,
                        sampling, score, cv_probs, test_probs) values
                        (?,?,?,?,?,?,?)"""
            conn.executemany(query, vals)
            conn.commit()

    def predict(self, X):
        pass
