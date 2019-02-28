from __future__ import division

import numpy as np
import sqlite3
from sklearn.utils import check_random_state, shuffle
from sklearn.neighbors import NearestNeighbors

def rus(X, y, random_state=None):
    # step 1: get minority class and its count
    minor = y==1
    minorCnt = np.count_nonzero(minor)
    # step 2: get indices of majority
    majorIdx = np.flatnonzero(y==1)
    # step 3: random sample majority examples
    majorIdx = check_random_state(random_state).choice(majorIdx, minorCnt, replace=False)
    # step 4: concatenate and return balanced dataset
    minor[majorIdx] = True
    return X[minor], y[minor]


def smote(X, y, b=100, k=5, random_state=None):
    """
    X: feature ndarray
    y: target array
    b: sampling rate (b%) of majority class (1.0 meaning #minority = #majority)
    k: number of neighbors used
    """
    # return features and balanced target class
    ###########################################################
    count = np.bincount(y)  # number of instances of each class
    majorCls = count.argmax() #minority class
    minorCls = count.argmin() #minority class
    majorCnt = count.max()    #majority class count
    minorCnt = count.min()    #minoirty class count
    ###########################################################
    # 1. for majority class, sample b% original data
    n,m = X.shape
    features = np.empty((0,m)) ; target = np.empty((0,))
    size = int((majorCnt)*(b/100))
    idx = np.random.choice(majorCnt, size, replace=False)
    features = np.concatenate([features, X[(y==majorCls), :][idx, :]])
    target = np.concatenate([target, majorCls*np.ones((size, ))])
    ###########################################################
    # 2. for minority class, create synthetic data with k-nearest neighbor
    # issue: what if minorCnt == 0 ?
    minorInd = np.flatnonzero(y==minorCls)
    minorFeature = X[minorInd]
    nn_k = NearestNeighbors(n_jobs=-1, n_neighbors=k+1);  nn_k.fit(minorFeature)
    nns = nn_k.kneighbors(minorFeature, return_distance=False)[:,1:]
    random_state = check_random_state(None)
    samples_indices = random_state.randint(low=0,high=len(nns.flatten()),size=size)
    steps = 1.0 * random_state.uniform(size=size)      #stepsize  = 1.0
    rows = np.floor_divide(samples_indices, k) 
    cols = np.mod(samples_indices, k)             #sample_indices = rows * k + cols
    y_new = np.array([minorCls]*size, dtype=y.dtype)
    X_new = np.zeros((size, m), dtype=X.dtype)
    ###########################################################
    def _generate_sample(X, nn_data, nn_num, row, col, step):
        """Generate a synthetic sample.
                The rule for the generation is:
                new_sample = unif(0,1)*(current_sample - randomly selected neighbor) + current_sample
        """
        return X[row] - step * (X[row] - nn_data[nn_num[row, col]])
    for i, (row, col, step) in enumerate(zip(rows, cols, steps)): 
        X_new[i] = _generate_sample(minorFeature, minorFeature, nns, row, col, step)
    ###########################################################
    features = np.concatenate((features, X_new))
    target = np.concatenate((target, y_new))
    ###########################################################
    # 3. shuffle index so that first half is not all positive 
    features, target = shuffle(features, target, random_state = random_state)
    return features, target
