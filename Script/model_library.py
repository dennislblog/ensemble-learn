# -*- coding: utf8
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD
# Modified by Tilii [reditrouncel -at- gmail -dot- com]
"""Utility module for building model library"""

from __future__ import print_function

import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression


# generic model builder
def build_models(model_class, param_grid):

    result = [model_class(**p) for p in ParameterGrid(param_grid)]
    print('Building {} {} models'.format(len(result), str(model_class).split('.')[-1][:-2]))

    return result


def build_LightGBMClassifiers(random_state=None):
    import multiprocessing as mp

    if mp.cpu_count() > 1:
        n_thread = int( mp.cpu_count() / 2 )
    else:
        n_thread = 1

    param_grid = {
        'silent'            : [True],
        'boosting_type'     : ['gbdt'],
        'objective'         : ['binary'],
        'random_state'      : [random_state],
        'min_child_weight'  : [2, 5],
        'n_estimators'      : [200, 300, 500],
        'learning_rate'     : [0.1, 0.05],
        'num_leaves'        : [20, 30, 40],
        'min_child_samples' : [10, 20, 30],
        'max_depth'         : [-1],
        'max_bin'           : [127],
        'n_jobs'            : [n_thread],
    }

    return build_models(LGBMClassifier, param_grid)



def build_LogisticRegressionClassifiers(random_state=None):
    import multiprocessing as mp

    if mp.cpu_count() > 1:
        n_thread = int( mp.cpu_count() / 2 )
    else:
        n_thread = 1

    param_grid = {
        'tol'           : np.logspace(-5, -3, num=3, base=10),
        'C'             : np.logspace(-2, 1, num=4, base=10),
        'solver':  ['newton-cg', 'lbfgs', 'sag'],
        'class_weight': [None, 'balanced'],
        'random_state': [random_state],
        'max_iter': [ 10000 ],
        'penalty': ['l2'],
        'n_jobs': [n_thread],
        'dual': [False],
        'fit_intercept' : [False, True],
    }

    return build_models(LogisticRegression, param_grid)


def build_MultiLayerPerceptron(random_state=None):
    # 1 and 8 are ruled out; momentunm = 0.1 ruled out; 
    param_grid = {
        'hidden_layer_sizes' : [(8,),(16,),(32,), (4, 2), (8, 2), (16,2)],
        'solver': ['sgd'],
        'momentum': [0.2, 0.5, 0.9],
        'learning_rate_init': [0.02, 0.01, 0.005],
        'activation': ["relu", "logistic", "tanh"]
    }
    return build_models(MLPClassifier, param_grid)


def build_randomForestClassifiers(random_state=None):
    param_grid = {
        'n_estimators': [20, 50],
        'criterion':  ['gini', 'entropy'],
#        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_features': [None],
#        'min_density': [0.25, 0.5, 0.75, 1.0],
#### Added by Tilii
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
###
        'random_state': [random_state],
        'max_depth': [2, 5, 10],
#        'max_depth': [1, 2, 5, 10],
    }

    return build_models(RandomForestClassifier, param_grid)


def build_gradientBoostingClassifiers(random_state=None):
    param_grid = {
        'n_estimators': [10, 20, 50, 100],
#        'subsample': np.linspace(0.2, 1.0, 5),
        'subsample': np.linspace(0.1, 1.0, 5),
#        'max_features': np.linspace(0.2, 1.0, 5),
        'max_features': np.linspace(0.1, 1.0, 5),
#        'max_depth': [1, 2, 5, 10],
        'max_depth': [2, 5, 10],
    }

    return build_models(GradientBoostingClassifier, param_grid)


def build_sgdClassifiers(random_state=None):
    param_grid = {
        'loss': ['log', 'modified_huber'],
#        'penalty': ['elasticnet'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
#        'learning_rate': ['constant', 'optimal'],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
#        'n_iter': [2, 5, 10],
#### Added by Tilii
        'max_iter': [10, 50],
###
        'eta0': [0.001, 0.01, 0.1],
        'l1_ratio': np.linspace(0.0, 1.0, 3),
    }

    return build_models(SGDClassifier, param_grid)


def build_decisionTreeClassifiers(random_state=None):
    rs = check_random_state(random_state)
    #max_features = 'None'; max_depth = 2; criterion: entropy
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_features': [None, 'auto', 'log2'],
        'max_depth': [2, 4, 6, None],
        'min_samples_split': [2, 5, 10, 20],
        # 'random_state': [rs.random_integers(100000) for i in range(3)],
    }

    return build_models(DecisionTreeClassifier, param_grid)


def build_extraTreesClassifiers(random_state=None):
    param_grid = {
        'n_estimators': [5, 10, 20, 100],
        'max_features': [None, 'auto', 'log2'],
        'max_depth': [2, 4, 6],
        'min_samples_split': [2, 5, 10],
        'random_state': [random_state],
    }

    return build_models(ExtraTreesClassifier, param_grid)


def build_svcs(random_state=None):

    Cs = np.logspace(-7, 2, 10)
    gammas = np.logspace(-6, 1, 8, base=2)
    coef0s = [-1.0, 0.0, 1.0]

    models = []

    # for C in Cs:
    #     models.append(SVC(kernel='linear', C=C, probability=True,
    #                       cache_size=1000))

    # for C in Cs:
    #     for coef0 in coef0s:
    #         models.append(SVC(kernel='sigmoid', C=C, coef0=coef0,
    #                           probability=True, cache_size=1000))

    for C in Cs:
        for gamma in gammas:
            models.append(SVC(kernel='rbf', C=C, gamma=gamma,
                              cache_size=1000, probability=True))

    param_grid = {
        'kernel': ['poly'],
        'C': Cs,
        # 'gamma': gammas,
        'degree': [2,3],
        # 'coef0': coef0s,
        'probability': [True],
        'cache_size': [1000],
    }

    for params in ParameterGrid(param_grid):
        models.append(SVC(**params))
    print('Building %d SVM models' %len(models))

    return models


def build_kernPipelines(random_state=None):
    print('Building Kernel Approximation Pipelines')

    param_grid = {
        'n_components': range(5, 105, 5),
        'gamma': np.logspace(-6, 2, 9, base=2)
    }

    models = []

    for params in ParameterGrid(param_grid):
        nys = Nystroem(**params)
        lr = LogisticRegression()
        models.append(Pipeline([('nys', nys), ('lr', lr)]))

    return models


def build_kNeighborsClassifier(N_instances, random_state=None):
    
    N = 25
    sep = N_instances//N
    start = np.random.randint(sep)
    lst = [(start + sep * i)%N_instances for i in range(N)]
    lst = [5 if ele == 0 else ele for ele in lst]

    ######### ! as it turns out local defined function is not pickable ############
    def func(distance):
        return 1 - distance
    param_grid = {
        'n_neighbors': lst,
        'algorithm': ['brute', 'auto'],
        'weights': ['uniform','distance']
    }

    return build_models(KNeighborsClassifier, param_grid)


models_dict = {
    'svc': build_svcs,
    # 'sgd': build_sgdClassifiers,
    # 'gbc': build_gradientBoostingClassifiers,
    'dtree': build_decisionTreeClassifiers,
    # 'forest': build_randomForestClassifiers,
    # 'lgb': build_LightGBMClassifiers,
    'logit': build_LogisticRegressionClassifiers,
    # 'extra': build_extraTreesClassifiers,
    'knn': build_kNeighborsClassifier,
    # 'mlp': build_MultiLayerPerceptron,
    # 'kernp': build_kernPipelines,
}


# def build_model_library(model_types=['dtree'], random_seed=None):
#     models = []
#     for m in model_types:
#         models.extend(models_dict[m](random_state=random_seed))
#     return models


def build_model_library(N_instances, random_seed=None):

    models = []
    for name, model in models_dict.items():
        if name == 'knn':
            models.extend(models_dict[name](N_instances, random_state=random_seed))
        else:
            models.extend(models_dict[name](random_state=random_seed))
    return models