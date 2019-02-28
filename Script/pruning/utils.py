import numpy as np
from Script.metrics.evaluation import _recall, _gm, _sar, _auc
from sklearn.metrics import cohen_kappa_score
import warnings
import inspect


def name_and_args():
    caller = inspect.stack()[1][0]
    args, _, _, values = inspect.getargvalues(caller)
    return [(i, values[i]) for i in args]


def direct_select(true_y, select, candidates, votes, clsWeight, cnt_votes, prob, func):
    """Description
    Parameter
    ---------------------------------------------------------------
    candidates: indices of all remaining classifiers
    select: indices of selected classifiers in current ensemble
    votes: 2d-array with rows representing each instance, columns representing each classifier [-x,x]
    cnt_votes_: 2d np array of shape [N, 2] recording total weights for majority/minority classes\
    clsWeight: 1d-array indicating weight of each classifiers (M, )
    prob: whether calculation is based on probability or actual vote
    ===============================================================
    update_vote returns weighted votes after adding a weighted classifier (dynamic programming)
    in current version, we haven't included balance feature into the design of direct search as 
    well as kappa diversity search. 
    """
    def getensemblevote(array):
        # array is 1d np of shape N_instance, each is a weighted vote of candidate classifier
        # return 2d array of shape [N, 2] representing neg weight, pos weight 
        cnt = cnt_votes.copy()
        neg_filter = array <= 0; pos_filter = ~neg_filter
        cnt[neg_filter, 0] += -array[neg_filter]
        cnt[pos_filter, 1] += array[pos_filter]
        return cnt
    def get_score(array):
        return func(true_y, array)
    # ------------------------------------------------------------------------ #
    res = np.array(list(map(getensemblevote, votes[:, candidates].T)))
    update_vote, pred_prob = res[:,:,:2], res[:,:,1]/res[:,:,:2].sum(axis=2)
    if not prob:
        pred_prob = (pred_prob > 0.5).astype(int)
    scores = np.array(list(map(get_score, pred_prob)))
    best = scores.argmax()
    candidate, cnt_vote, score = candidates[best], update_vote[best], scores[best]
    return candidate, cnt_vote, clsWeight, score  

def select_candidate_by_kappa(**args):
    """Description
    kappa = difference between observed and expected agreemenet
    Cohen(1960s) pointed out some agreement comes from merely guessing
    kappa is ranged from -1 to +1; 1 means perfect agreement without 
        guessing at all where as ≤ 0 as indicating no agreement
    """
    args.update({'prob': False, 'func': cohen_kappa_score})
    return direct_select(**args)

def select_candidate_by_sar(**args):
    args.update({'func': _sar})
    return direct_select(**args)

def select_candidate_by_gm(**args):
    args.update({'func': _gm})
    return direct_select(**args)

def select_candidate_by_recall(**args):
    args.update({'func': _recall})
    return direct_select(**args)

def select_candidate_by_auc(**args):
    args.update({'func': _auc})
    return direct_select(**args)


def get_probability(matrix, enforcer=0.0):
    """calcuate total votes for +1 and -1 then calculate proportion
    Parameter
    ---
    array is an array of -x and +x if vote is false
    """
    def helper(array):
        abSum = np.abs(array).sum()+enforcer
        return sum(array[array>0])/abSum
    return np.array(list(map(helper, matrix)))


# def get_probability_by_count(matrix, enforcer=0.0):
#     """matrix of zero-one"""
#     def helper(array):
#         pos = sum(array==1)
#         return pos/(len(array)+)
#         if len(count) == 1:
#             return 0.0
#         else:
#             return count[1] / (count[0] + count[1])
#     return np.array(list(map(helper, matrix)))

# def select_candidate_by_cap(true_y, select, candidates, votes, Kmm, Kcm):
#     """Descriptiobn
#         CAP score too time consuming, abandon this method
#     """
#     Ns = len(select)
#     res = []
#     for candidate in candidates:

#         # 1) create temporary memory of following important variables
#         _selects = votes[:, select]
#         _candidate = votes[:, candidate]

#         # 2) dynamic programming: update Kmm and Kcm accordingly
#         if Ns == 1:  # the 2nd round when ensemble includes one classifier only
#             Kmm = cohen_kappa_score(_selects, _candidate)
#             Kcm = (cohen_kappa_score(_selects, true_y) +
#                    cohen_kappa_score(_candidate, true_y)) / 2
#         else:        # update m-m and c-m scores dynamically
#             Kmm = (Kmm * Ns + sum(map(lambda x: cohen_kappa_score(x,
#                                                                   _candidate), _selects.T))) / (Ns + 1)
#             Kcm = (Kcm * Ns + cohen_kappa_score(_candidate, true_y)) / (Ns + 1)

#         # 3） calculate merit score for this candidate
#         merits = Ns * Kcm / np.sqrt(Ns + Ns * (Ns - 1) * Kmm)
#         res.append(merits)
#     return candidates[np.array(res).argmax()], Kmm, Kcm


def heuristic_select(true_y, select, candidates, votes, clsWeight, insWeight, balance, heuristic, enforcer):
    """Description
    Parameter
    ---------------------------------------------------------------
    insWeight: (N, ) shape array
    clsWeight: (M, ) in current version classifier weight doens't play a role in boosting algorithm !!
    ===============================================================
    Return
    ---------------------------------------------------------------
    best candidate and new updated instance weight, possibly new classifier weight (update in the future)
    ===============================================================
    Special Notification
    ---------------------------------------------------------------
    For boosting based pruning: classifier weight doesn't make any difference in selection
    """
    def get_weighted_score(array):
        # array := candidate weighted vote on all instances
        if heuristic == 'boost':
            return sum(-weight * ((array>0).astype(int) != true_y)) 
        else:
            return sum(weight * (2 * ((array>0).astype(int) == true_y) - 1))
    #---------------------------------------------------------------------------#
    # Boosting based pruning #
    def update_by_boost(select_candidate, _insWeight):
        correct_filter = (votes[:, select_candidate] > 0.5).astype(int) == true_y
        incorrect_filter = ~correct_filter
        error = len(incorrect_filter)/N
        if error > 0.5:
            _insWeight = np.array([1 / N] * N)
        else:
            if error == 0.0:    error = 0.01  # avoid inf weight
            if error == 1.0:    error = 0.99  # avoid 0.0 weight
            _insWeight[correct_filter] /=  (2 * (1 - error))
            _insWeight[incorrect_filter] /=  (2 * error)
        return normal(_insWeight)
    #---------------------------------------------------------------------------#
    # Uncertainty Weighted Accuracy based pruning #
    def update_by_uwa(select_candidate, _insWeight):
        pred_prob = get_probability(votes[:, select + [select_candidate]])
        return normal(np.minimum(pred_prob, 1 - pred_prob))
    #---------------------------------------------------------------------------#
    # Difficulty Weighted Accuracy based pruning #
    def update_by_dwa(select_candidate, _insWeight):
        pred_prob = get_probability(votes[:, select + [select_candidate]])
        correct = true_y == (pred_prob > 0.5).astype(int)
        pred_prob[correct] = np.minimum(pred_prob[correct], 1 - pred_prob[correct])
        pred_prob[~correct] = np.maximum(pred_prob[~correct], 1 - pred_prob[~correct])
        return normal(pred_prob)
    #---------------------------------------------------------------------------#
    # Uncertainty & Difficulty Weighted Accuracy based pruning #
    def update_by_udwa(select_candidate, _insWeight):
        pred_prob = get_probability(votes[:, select + [select_candidate]])
        # ----- weighted by difficulty when instance is minority class -------- #
        flag = true_y == (pred_prob > 0.5).astype(int) & (true_y == 1)
        pred_prob[~flag] = np.minimum(pred_prob[~flag], 1 - pred_prob[~flag])
        pred_prob[flag] = np.maximum(pred_prob[flag], 1 - pred_prob[flag])
        return normal(pred_prob)
    #---------------------------------------------------------------------------#
    # Uncertainty & Difficulty Weighted Accuracy based pruning + System Bias#
    def update_by_udwa_enf(select_candidate, _insWeight):
        pred_prob = get_probability(votes[:, select + [select_candidate]],enforcer)
        # ----- weighted by difficulty when instance is minority class -------- #
        flag = true_y == (pred_prob > 0.5).astype(int) & (true_y == 1)
        pred_prob[~flag] = np.minimum(pred_prob[~flag], 1 - pred_prob[~flag])
        pred_prob[flag] = np.maximum(pred_prob[flag], 1 - pred_prob[flag])
        return normal(pred_prob)
    #############################################################################
    # 1) to balance the importance of majority/minority bias
    if balance:
        tmpWeight = insWeight.copy()
        major = true_y == 0; minor = ~major
        tmpWeight[major] = normal(tmpWeight[major])/2
        tmpWeight[minor] = normal(tmpWeight[minor])/2
    #############################################################################
    # 2) update instance weight 
    N = len(true_y)  # how many instances in total
    candidate = select[-1]  # added classifier in the last round
    insWeight = eval('update_by_' + heuristic + '(candidate, insWeight)')
    # 2) to balance the importance of majority/minority bias
    #############################################################################
    if balance:
        tmpWeight = insWeight.copy()
        major = true_y == 0; minor = ~major
        tmpWeight[major] = normal(tmpWeight[major])/2
        tmpWeight[minor] = normal(tmpWeight[minor])/2
    weight = tmpWeight if balance else insWeight
    #############################################################################
    # 3) select best candidate classifier
    scores = np.array(list(map(get_weighted_score, votes[:, candidates].T)))
    best = scores.argmax()
    candidate, score = candidates[best], scores[best]
    #############################################################################
    return candidates[best], insWeight, clsWeight



def normal(array):
    Sum = sum(array)
    if Sum == 0:
        return np.zeros(len(array))
    else:
        return array / Sum