import numpy as np
import sqlite3
from pickle import loads, dumps
from Script.metrics.evaluation import Evaluator
from .utils import *
from sklearn.metrics import confusion_matrix


class Selector(object):

    def __init__(self, y_prune=None, y_test=None, classifier=None, acc=None, prune=None, test=None, loop=21):
        """Desciprtion
                        1. access model_idx, probs over training data from db_file
                        2. apply select(heuristic=,balance=) to probs->votes and get optimal weight
                        3. write model_idx, weight into a new table heuristic+balance(imb and no-imb)
                        ---
                        Parameter
                        max_stop: maximum number of subensemble to evaluate
                                        if there is less than 21 models in the full ensemble, evaluate all subensembles
                        metrics: an Evaluator object
        """
        self.yprune = y_prune
        self.ytest = y_test
        self.classifier = classifier
        self.acc = acc
        self.prune = prune
        self.test = test
        self.loop = loop
        # try:
        # except ValueError:
        #     print("make sure table model_scores contains model_idx, score and probs")
        #     raise
        # except AssertionError:
        #     print("the target variable doesn't match database")
        #     raise
        # stmt = (", {} real default 0.0" *
        #         len(self._heuristic)).format(*self._heuristic)
        # createstmt1 = "create table if not exists ensemble (model_idx integer UNIQUE NOT NULL" + stmt + ");"
        # createstmt2 = "create table if not exists ensemble_imb (model_idx integer UNIQUE NOT NULL" + stmt + ");"
        # createstmt = createstmt1 + createstmt2
        # db.executescript(createstmt)
        # raise ValueError("ensemble weight table couldn't be built")

    # def reset(self, heuristic=None):

    def run(self, balance=False, heuristic='recall', metric='auc', weight=False, update=False):
        """Description
                1. votes is a ndarray(N_instance, N_classifier) where each element
                2. balance controls whether importance of majority/minority should be equal
                3. heuristic specifies which heuristic function used to drive the selection
                4. whether allow classifier weight to be updated while selection
           Return
                1. a sequence of cv_score
                2. a sequence of test_score
                3. pred_prob of test data
        """
        ########################################################################################
        """
            step 1: initate variables and select first candidate classifier
                - return cv_score(21, ) and test_score(21, )
                - optimal (optimal+1 = size of selected best ensemble) and pred_prob_test of optimal
        """
        ########################################################################################
        # --- initiate variables to return --- #
        cv_score, test_score = np.zeros((self.loop,)), np.zeros((self.loop,))
        N, M = self.prune.shape
        # --- transform prune,test data to -x,x weighted votes --- #
        if weight:
            clsWeight = self.acc.copy()
            # avoid inf error and neg weight
            neg_filter, inf_filter = clsWeight < 0.5, clsWeight == 1.0
            nom_filter = ~neg_filter & ~inf_filter
            clsWeight[neg_filter], clsWeight[inf_filter] = 0.0, np.log10(100)
            clsWeight[nom_filter] = np.log10(
                clsWeight[nom_filter] / (1 - clsWeight[nom_filter]))
        else:
            clsWeight = np.ones((M,), dtype='int64')
        # -------------------------------------------------------------------------- #
        # --- votes are transformed to allow multiplicative of weighted accuracy --- #
        prune_vote = clsWeight * (2 * self.prune - 1)
        test_vote = clsWeight * (2 * self.test - 1)
        # --- set up enfocer --- #
        if heuristic == 'udwa_enf':
            ratio = sum(self.yprune == 0) / sum(self.yprune == 1)
            enforcer = np.log10(ratio) if weight else int(ratio) - 1
        else:
            enforcer = 0.0
        # -------------------------------------------------------------------------- #
        ########################################################################################
        # --- sepcify the measurement --- #
        self.metric = Evaluator(metric)
        # --- select best candidate as the 1st one --- #
        optimal = np.argmax(self.acc)
        classifiers = self.classifier.copy()
        ensemble = [optimal]
        # --- generate cv_score and test_score of 1st selected classifier --- #
        cv = self.metric.calculate(self.yprune, get_probability(
            prune_vote[:, ensemble], enforcer))
        test = self.metric.calculate(
            self.ytest, get_probability(test_vote[:, ensemble], enforcer))
        cv_score[0], test_score[0] = cv, test
        # --- remove 1st selected classifier from the candidate list --- #
        classifiers.remove(optimal)
        ########################################################################################
        """
            step 2: update instance weight accordingly
                - (0,1) binary coding is transformed to (-1,1) so that we can add multiplicator log(odds)
                - instance weight is initialized to be the same
                - for direct search (including kappa), we have to evaluate based on the performance
                    of CurEns + Candidate, instance weight is not required. For efficiency purpose,
                    dynamically update a ledger of vote for pos vs neg as well as score
        """
        ######################################################################################## 
        cv_func = lambda x: self.metric.calculate(self.yprune, get_probability(
                prune_vote[:, x], enforcer))
        test_func = lambda x: self.metric.calculate(self.ytest, get_probability(
            test_vote[:, x], enforcer))
        ########################################################################################
        # if random select is specified
        if heuristic == 'random':
            classifier_perm = np.array([np.random.permutation(classifiers) for _ in range(10)]).reshape(10,len(classifiers))                
            classifier_perm = np.c_[ensemble*10, classifier_perm]
            for t in range(1, self.loop):
                cv = np.mean(list(map(cv_func, classifier_perm[:,:t+1])))
                test = np.mean(list(map(test_func, classifier_perm[:,:t+1])))
                cv_score[t], test_score[t] = cv, test
            optimal = np.argmax(cv_score)
            pred_prob = np.zeros((len(self.ytest),))
            return cv_score, test_score, optimal, pred_prob
        # if Hueristic search is UWA, UDWA, UDWA_imb based
        if heuristic in {'boost', 'uwa', 'dwa', 'udwa', 'udwa_enf'}:
            instWeight = np.array([1 / N] * N)
        # if direct search is applied, instance weight is not needed
        if heuristic in {'recall', 'auc', 'gm', 'sar', 'kappa'}:
            cntVote = np.array(
                list(map(lambda x: (-min(x, 0), max(x, 0), 0.0), prune_vote[:, optimal])))
        ########################################################################################
        """
            step 3: main block, iterate through t = 1 -> 20
        """
        ########################################################################################
        for t in range(1, self.loop):
            ###################################################################################
            if update:
                prune_vote = clsWeight * (2 * self.prune - 1)
            ###################################################################################
            # --- all heuristic share in common --- #
            param_dict = {
                'true_y': self.yprune,
                'select': ensemble,
                'candidates': classifiers,
                'votes': prune_vote,
                'clsWeight': clsWeight}
            # --- all heuristic share in common --- #
            if heuristic in {'boost', 'uwa', 'dwa', 'udwa', 'udwa_enf'}:
                param_dict.update(
                    {'insWeight': instWeight, 'balance': balance, 'heuristic': heuristic, 'enforcer': enforcer})
                best_candidate, instWeight, clsWeight = heuristic_select(
                    **param_dict)                
            else:
                param_dict.update({'cnt_votes': cntVote, 'prob': True})
                best_candidate, cntVote, clsWeight, best_score = eval('select_candidate_by_' + heuristic)(
                    **param_dict)
            # --- remove selected classifier from candidate list --- #
            ensemble += [best_candidate]
            classifiers.remove(best_candidate)
            # --- evaluate CurEns + selected Candidate cv and test score --- #
            cv, test = cv_func(ensemble), test_func(ensemble)
            cv_score[t], test_score[t] = cv, test
            # -------------------------------------------------------------------------------- #
        optimal = np.argmax(cv_score)
        pred_prob = get_probability(
            test_vote[:, ensemble[:optimal + 1]], enforcer)
        return cv_score, test_score, optimal, pred_prob

    # def score_with_ensemble(self, ensemble, metric=None):
    #     """Parameter
    #                     ensemble is a list of index indicating which models are selected
    #                     target, votes are sourced from self.db_file
    #                     metric is a list of {f1, auc, logloss, rmse, accuracy, recall}
    #     """
    #     if metric == None:
    #         _metric = self.metric
    #     else:
    #         _metric = Evaluator(metric)
    #     return _metric.calculate(self.target, (np.mean(self.vote[:, ensemble], axis=1)))

    # @staticmethod
    # def get_proba(X, ind):
    #     """ here X is a 2-d array with element from -x to +x"""
    #     return

    # def get_confusion_matrix_with_classifier(self, classifier):
    #     return confusion_matrix(self.target, self.vote[:, classifier])

    # def _check_params(self):
    #     """Parameter sanity checks"""
    #     if (not self.db_file):
    #         raise ValueError("db_file parameter is required")
    #     if (not os.path.exists(self.db_file)):
    #         raise ValueError("database file '%s' not exists!" % self.db_file)


class EnsembleSelect(object):

    """
    This is especially helpful in trakcing ensemble selection with sepcified heuristic

    Parameters
    ----------
    keys: sampling x balance x heuristic x metric x weight
    record: from 1 -> self.loop (21), remember cv_score and test_score of each selection
    Returns
    ----------

    """
    ##################################################################################

    def __init__(self, **args):
        K = args.pop('max')
        N = args.pop('size')
        self.id = args.pop('id')
        self.sample = args['sample']
        self.key = args
        self.cv, self.test = np.zeros((K,)), np.zeros((K,))
        self.pred_prob = np.zeros((N,))

    def __repr__(self):
        return str(self.key)

    def update_score(self, cv_score, test_score):
        # cv_score is an array
        self.cv += cv_score
        self.test += test_score

    # def add_score(self, cv, test, size):
    #     # cv/test/size is singleton
    #     self.select_size += size
    #     self.best_cv += cv
    #     self.best_test += test

    def divde_by(self, num):
        self.cv /= num
        self.test /= num


    def withoutkey(self, key):
        _dict_ = self.key
        return {i: _dict_[i] for i in _dict_ if i != key}