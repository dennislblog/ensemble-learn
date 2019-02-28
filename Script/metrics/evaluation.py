import numpy as np

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from imblearn.metrics import geometric_mean_score

def _recall(y_true, y_probs):
    y_pred = (y_probs>0.5).astype(int)
    cnf = confusion_matrix(y_true, y_pred)
    return cnf[1,1]/(cnf[1,0]+cnf[1,1])

def _gm(y_true, y_probs):
    """return geometric mean score"""
    return geometric_mean_score(y_true, (y_probs>0.5).astype(int), average='binary')


def _auc(y_true, y_probs):
    """return AUC score (for binary problems only)"""
    return roc_auc_score(y_true, y_probs)


def _logloss(y_true, y_probs):
    """return negative log_loss since we're maximizing the score"""
    return -1.0 * (log_loss(y_true, y_probs))


def _rmse(y_true, y_probs):
    """return 1-rmse since we're maximizing the score for hillclimbing"""
    return 1.0 - np.sqrt(mean_squared_error(y_true, (y_probs>0.5).astype(int)))

def _accuracy(y_true, y_probs):
    """return accuracy score"""
    return accuracy_score(y_true, (y_probs>0.5).astype(int))

def _sar(*args):
    return (_rmse(*args)+_accuracy(*args)+_auc(*args))/3


class Evaluator(object):
    """Compute f1, auc, logloss, rmse, accuracy and max_entropy(xentropy)
    Examples
    --------
    from Script.metrics.evaluation import Evaluator
    accuracy = Evalutor('acc').calculate(y, y_bin, probs)
    """

    _metrics = {
        'gm': _gm,
        'auc': _auc,
        'logloss': _logloss,
        'rmse': _rmse,
        'sar': _sar,
        'recall': _recall,
        'accuracy': _accuracy
    }

    def __init__(self, metric='auc'):

        self.metric = metric
        metric_names = self._metrics.keys()
        if (metric not in metric_names):
            msg = "score_metric not in %s" % metric_names
            raise ValueError(msg)       
        self._metric = self._metrics[metric]

    def calculate(self, y_true, y_probs):
        return self._metric(y_true, y_probs)

    def change_metric(self, metric):
        return self.__class__(metric=metric)

    @classmethod
    def calculate_accuracy(Evaluator, *args):
        return Evaluator().calculate(*args)

