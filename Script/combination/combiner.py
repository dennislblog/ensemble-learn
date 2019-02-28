import numpy as np
from . import rules


class Combiner(object):

    def __init__(self, rule='majority_vote'):
        self.combination_rule = rule

        if rule == 'majority_vote':
            self.rule = rules.majority_vote_rule

        elif rule == 'weighted_vote':
            self.rule = rules.weighted_vote_rule

        elif rule == 'min':
            self.rule = rules.min_rule

        elif rule == 'mean':
            self.rule = rules.mean_rule

        elif rule == 'median':
            self.rule = rules.median_rule

        else:
            raise Exception('invalid argument rule for Combiner class')

    def combine(self, probs):
        """Combine probs
        Parameters
        ----------
        probs:  Numpy 2d-array with rows representing each instance, columns
                representing each classifier and elements representing predicted probability. 
        """
        n_samples = len(probs)

        out = np.zeros((n_samples))

        for i in range(n_samples):
            out[i] = self.rule(probs[i, :])

        return out
