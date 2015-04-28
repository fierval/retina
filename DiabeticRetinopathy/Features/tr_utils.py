import numpy as np
from time import gmtime, strftime, localtime
import csv
import os
from os import path
import shutil
import pandas as pd
from pandas.io.parsers import csv

def prep_out_path(out_path):
    if path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

def append_to_arr(arr, a, axis = 0):
    '''
    Append a to a numpy array arr. a - scalar, list or numpy array
    '''
    if isinstance(a, list) or isinstance(a, np.ndarray):
        a = np.array(a)

        if arr.shape[0] == 0:
            arr = a.reshape(1, a.shape[0])
        else:
            arr = np.append(arr, a.reshape(1, a.shape[0]), axis = axis)
    else:
        if arr.size == 0:
            arr = np.array([a]) # make sure it is a 1-dimensional array
        else:
            arr = np.append(arr, a)
    return arr

def time_now_str():
    return strftime("%d %b %Y %H:%M:%S", localtime())

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict.
    '''
    z = x.copy()
    z.update(y)
    return z

def vote(proba_list, weight_list):
    '''
    Given a list of probability arrays and a list of weights,
    Compute the final array by summiing probabilities and multiplying by their weights
    '''
    wts = np.array(weight_list)
    if wts[wts == 1].shape[0] == wts.shape[0]:
        proba = np.array([x for x in proba_list])
        return proba.mean(0)
    else:
        proba = np.array([x[0] * x[1] for x in zip(proba_list, weight_list)])
        return proba.sum(0)
     
def vote_reduce(arrs, weights):
    '''
    Given two arrays and a list of two weights, apply the voting rule as in vote(), unless
    a 0 or a 1 is encountered. In the former case pick the unweighted non-zero element, in the latter - the element
    with value of 1.
    '''
    def func (x, y):
        w2 = y[1]; y = y[0]
        for i, k in enumerate(np.nditer(y, ['c_index'])):
            if x[i] == 0 or y[i] == 1.0:
                x[i] = y[i]
            elif x[i] != 1 and y[i] != 0:
                x[i] = x[i] + y[i] * w2
        return x

    def init(x):
        return np.array([x * weights[0] if x != 1.0 else x for x in np.nditer(x, ['c_index'])])

    res = np.array([])
    probs = np.array(arrs)

    for i in range(0, probs.shape[1]):
        samples = probs[:, i, :].reshape(probs.shape[0], probs.shape[2])
        cur_proba = reduce(func, zip(samples[1:, :], np.array(weights)[1:]), init(samples[0]))
        res = append_to_arr(res, cur_proba)    
    return res

def isEmpty(arr):
    return len(arr) == 0

def write_to_csv(task_labels, labels, probs, out_file):
    predict_columns = ["Prediction{:1d}".format(i) for i in range(1, 10) ]

    existing_rows = pd.read_csv(task_labels, header=0, quoting=csv.QUOTE_NONNUMERIC)
    file_names = pd.DataFrame(labels, columns= ["Id"])
    probas = pd.DataFrame(probs, columns = predict_columns)
    out = pd.concat([file_names, probas], axis=1)
    out = pd.concat([existing_rows, out])

    out.to_csv(out_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
