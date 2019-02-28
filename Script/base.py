import numpy as np
from .model_library import build_model_library
import sys
import sqlite3
import os
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from pickle import loads, dumps
from Script.generation.base import Bagging
from Script.pruning.selector import Selector, EnsembleSelect
from Script.metrics.evaluation import Evaluator
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# two object:
# 1) how to write data into database
# 2) how to ensemble


class Ensemble(object):
    """Class that represents a collection of classifiers.

    The Ensemble class serves as a wrapper for a list of classifiers,
    besides providing a simple way to calculate the output of all the
    classifiers in the ensemble.

    Attributes
    ----------
    `classifiers` : list
        Stores all classifiers in the ensemble.

    `yval` : array-like, shape = [indeterminated]
        Labels of the validation set.

    `knn`  : sklearn KNeighborsClassifier,
        Classifier used to find neighborhood.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier
    >>>
    >>> from brew.base import Ensemble
    >>>
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0],
                      [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>>
    >>> dt1 = DecisionTreeClassifier()
    >>> dt2 = DecisionTreeClassifier()
    >>>
    >>> dt1.fit(X, y)
    >>> dt2.fit(X, y)
    >>>
    >>> ens = Ensemble(classifiers=[dt1, dt2])

    """

    def __init__(self, features=None, target=None, N_bags=100, K_folds=5, verbose=False, random_state=None):
        """
        Init whole process

        Parameters
        ----------
        N, number of instances

        Returns
        ----------

        """
        self.X = features
        self.y = target
        self.N = len(target)
        self.n = N_bags
        self.k = K_folds
        self.seed = random_state
        self.classifiers = build_model_library(
            N_instances=int(len(target) * 0.64), random_seed=random_state)
        np.random.shuffle(self.classifiers)

    ###################################################################################################
    def init_db(self, db_file):
        if os.path.exists(db_file):
            raise ValueError("db_file '%s' already exists!" % db_file)
        db = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        sqlite3.register_adapter(bool, int)
        sqlite3.register_converter("bool", lambda v: bool(int(v)))
        createTablesScript = """
                create table if not exists model_info (
                    fold_idx       integer NOT NULL,
                    model_idx      integer NOT NULL,
                    model          blob NOT NULL,
                    sampling       text NOT NULL,
                    score          real NOT NULL,
                    cv_probs       blob NOT NULL,
                    test_probs     blob NOT NULL
                );

                create table if not exists data_info (
                    fold_idx       integer unique NOT NULL,
                    assmb_idx      blob NOT NULL,
                    test_idx       blob NOT NULL
                );

                create table if not exists heuristic_info (
                    heuristic      text NOT NULL,
                    evaluation     text NOT NULL,
                    sampling       text NOT NULL,
                    weight         bool NOT NULL,
                    balance        bool NOT NULL,
                    select_size    integer NOT NULL,
                    cv_score       real NOT NULL,
                    test_avg       real NOT NULL,
                    test_score     real NOT NULL,
                    object         blob NOT NULL
                );
        """
        with db:
            db.executescript(createTablesScript)
        db.close()
        self.db_file = db_file
    ###################################################################################################

    def gen_data_info(self, db_file=None, random_state=None):
        """
        Generate Table Data_Info

        Parameters
        ----------

        Returns
        ----------

        """
        ##################################################################################
        if not db_file or not os.path.exists(db_file):
            db_file = self.db_file
        if not random_state:
            random_state = self.seed
        folds = StratifiedKFold(
            n_splits=self.k, shuffle=True, random_state=random_state).split(self.X, self.y)
        db_conn = sqlite3.connect(db_file)
        curs = db_conn.cursor()
        insert_stmt = """insert into data_info
                         (fold_idx, assmb_idx, test_idx)
                         values (?,?,?)"""
        for i, (assmb_index, test_index) in enumerate(folds):
            vals = (i + 1, memoryview(dumps(assmb_index)),
                    memoryview(dumps(test_index)))
            curs.execute(insert_stmt, vals)
        db_conn.commit()
        db_conn.close()

    ###################################################################################################
    def gen_model_info(self, db_file=None, folds=5, verbose=False, random_state=None, algorithms=['no', 'smote', 'rus']):
        """
        Generate Table Model_Info

        Parameters
        ----------
        sample: list, specify which data-sampling technique is applied on training classifiers

        Returns
        ----------

        Reference
        ----------
        LENS: Learning ENSembles using Reinforcement Learning (described in [Stanescu and Pandey, 2017])
        """
        ##################################################################################
        if not db_file or not os.path.exists(db_file):
            db_file = self.db_file
        if not random_state:
            random_state = self.seed
        # ------------------------------------------------------------------------------ #
        # open a cursor for sharing database in the following calls
        db_conn = sqlite3.connect(db_file)
        # for each assemble-test fold, do the following
        for i in range(self.k):
            # 1) select assemble data from db to get (X,y) for train and select
            with db_conn:
                select_stmt = "select assmb_idx, test_idx from data_info where fold_idx = %d"
                for (id1, id2) in db_conn.execute(select_stmt % (i + 1)):
                    assmb_id, test_id = loads(id1), loads(id2)
            # 2) randomly select 100 classifiers to fit 100 bootstrapped bags of train.data
            selected_models = np.random.choice(
                self.classifiers, size=self.n, replace=False)
            # 3) for each of 100 randomly selected classifier, initiate a bagging instance
            param_dict = {
                'X': self.X[assmb_id],
                'y': self.y[assmb_id],
                'fold_id': i + 1,
                'n_folds': folds,
                'classifiers': selected_models,
                'conn': db_conn,
                'feature': self.X[test_id],
                'verbose': False,
                'random_state': random_state
            }
            _bagging_ = Bagging(**param_dict)
            # 4) for each sampling method, fill the corresponding table rows
            for algorithm in algorithms:
                _bagging_.run(n_bags=self.n, algorithm=algorithm)
        db_conn.close()

    ###################################################################################################
    _direct_ = ['recall', 'auc', 'gm', 'sar']
    _heuristic_ = ['kappa', 'boost', 'uwa', 'dwa',
                   'udwa', 'udwa_enf', 'recall', 'auc', 'gm', 'sar', 'random']
    _bool_ = [True, False]

    def gen_heuristic_info(self, db_file=None, verbose=False, random_state=None, evaluation=_direct_, heuristic=_heuristic_, weight=_bool_, balance=_bool_, max_stop=21):
        """
        Generate Table Heuristic_Info

        Parameters
        ----------

        Returns
        ----------


        """
        ##################################################################################
        if not db_file or not os.path.exists(db_file):
            db_file = self.db_file
        if not random_state:
            random_state = self.seed
        self.loop = max_stop if self.n > max_stop else self.n
        # ------------------------------------------------------------------------------ #
        # open a cursor for sharing database in the following calls
        db_conn = sqlite3.connect(db_file)
        # retrieve sampling info, return a tuple of unique sampling method
        sampling = [x[0] for x in db_conn.execute(
            'select distinct sampling from model_info')]
        # ------------------------------------------------------------------------------ #
        # define some variables
        # 1) ensobj keeps track of how training and test behaves as we select more models
        # 2) select_size, cv_score, test_score of each fold as a list
        K = len(evaluation) * len(heuristic) * \
            len(weight) * len(balance) * len(sampling)
        EnsObj = [None for _ in range(K)]
        select_size, best_cv, best_test = np.zeros((3, K, self.k))
        pred_prob = np.zeros((K, self.N, ))
        param_grid = {
            'sample': sampling,
            'balance': balance,
            'heuristic': heuristic,
            'metric': evaluation,
            'weight': weight}
        for i, p in enumerate(ParameterGrid(param_grid)):
            # if p['heuristic'] in self._direct_ and p['heuristic'] != p['metric']:
            #     continue
            # else:
            p.update({'max': self.loop, 'size': self.N, 'id': i})
            EnsObj[i] = EnsembleSelect(**p)
        # ------------------------------------------------------------------------------ #
        # for each assemble-test fold, do the following
        # param_grid.pop('sample')
        for i in range(self.k):
            # 1) select assemble target and test target from data_info
            with db_conn:
                select_stmt = "select assmb_idx, test_idx from data_info where fold_idx = %d"
                for (id1, id2) in db_conn.execute(select_stmt % (i + 1)):
                    assmb_id, test_id = loads(id1), loads(id2)
            N1, N2 = len(assmb_id), len(test_id)
            # ----------------------------------------------------------------------------- #
            # for each sample method (smote, rus, etc)
            for _sample_ in sampling:
                #  model id -> the goal is to return subset of these IDs
                #  model acc -> for use of weighted ensemble
                #  model vote -> 2-d array (N, M) N assmb instances and M classifiers each entry is 0/1 vote
                # ------------------------------------------------------------------------- #
                model_id = np.zeros((self.n,), dtype='int')
                model_acc = np.zeros((self.n,))
                prune_vote, test_vote = np.zeros(
                    (self.n, N1)), np.zeros((self.n, N2))
                # ------------------------------------------------------------------------- #
                with db_conn:
                    select_stmt = "select model_idx, score, cv_probs, test_probs from model_info where fold_idx = ? and sampling = ?"
                    for (a, b, c, d) in db_conn.execute(select_stmt, [i + 1, _sample_]):
                        model_id[a], model_acc[a] = a, b
                        prune_vote[a], test_vote[a] = (loads(c) > 0.5).astype(
                            int), (loads(d) > 0.5).astype(int)
                        # print("prune_vote shape = {} and load shape = {}".format(prune_vote[a].shape, loads(c).shape))
                prune_vote, test_vote = prune_vote.T, test_vote.T
                # ------------------------------------------------------------------------- #
                # 3) now initiate a selector instance for ensemble selection
                _ES_ = Selector(y_prune=self.y[assmb_id], y_test=self.y[test_id], classifier=list(
                    model_id), acc=model_acc, prune=prune_vote, test=test_vote, loop=self.loop)
                # ---------------------------------------------------------------------- #
                # 4) for EnsObj which has sample method to be _sample_, do the following selection
                for tmpObj in EnsObj:
                	if tmpObj.sample == _sample_:
                		_id_ = tmpObj.id
                		cv_score, test_score, optimal, pred_prob[_id_, test_id] = _ES_.run(**tmpObj.withoutkey('sample'))
                		best_cv[_id_, i], best_test[_id_, i], select_size[_id_, i] = cv_score[optimal], test_score[optimal], optimal+1
                		tmpObj.update_score(cv_score, test_score)
                    # ------------------------------------------------------------------ #
        ##############################################################################
        # average self.k fold result and save it to the third table
        ##############################################################################
        # 1) average select size, best_cv, best_test for each _id_
        # 2) average cv_score and test_score for each _id_ EnsObj
        # 3) calculate test performance of the whole data for each EnsObj
        with db_conn:
            for tmpObj in EnsObj:
                if tmpObj:
                    # ---------------------------------------------------------------------------- #
                    # 1) fill in some basic information about this heuristic search algorithm
                    info = tuple(tmpObj.key[k] for k in (
                        'heuristic', 'metric', 'sample', 'weight', 'balance'))
                    _id_ = tmpObj.id
                    # ---------------------------------------------------------------------------- #
                    # 2) update instance val across self.k validation
                    tmpObj.divde_by(self.k)
                    size, cv, test = select_size.mean(axis=1), best_cv.mean(axis=1), best_test.mean(axis=1)
                    # ---------------------------------------------------------------------------- #
                    test_score = Evaluator(tmpObj.key['metric']).calculate(
                        self.y, pred_prob[_id_])
                    setattr(tmpObj, 'test_score', test_score)
                    setattr(tmpObj, 'pred_prob', pred_prob[_id_])
                    vals = info + \
                        (size[_id_], cv[_id_], test[_id_], test_score, memoryview(dumps(tmpObj)))
                    insert_stmt = """
                        insert into heuristic_info (heuristic, evaluation, sampling,
                        weight, balance, select_size, cv_score, test_avg, test_score, object) values
                        (?,?,?,?,?,?,?,?,?,?)
                    """
                    db_conn.execute(insert_stmt, vals)
                    db_conn.commit()
        db_conn.close()

    def get_integrate_result():
        pass


# def oracle(ensemble, X, y_true, metric=auc_score):
#     out = ensemble.output(X, mode='labels')
#     oracle = np.equal(out, y_true[:, np.newaxis])
#     mask = np.any(oracle, axis=1)
#     y_pred = out[:, 0]
#     y_pred[mask] = y_true[mask]
#     return metric(y_pred, y_true)


# def single_best(ensemble, X, y_true, metric=auc_score):
#     out = ensemble.output(X, mode='labels')
#     scores = np.zeros(len(ensemble), dtype=float)
#     for i in range(scores.shape[0]):
#         scores[i] = metric(out[:, i], y_true)
#     return np.max(scores)


if __name__ == "__main__":
    print("hello")
