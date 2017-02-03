"""Utils module contains various utility functions that are used in different parts
of the conformal prediction library.
"""

import numpy as np

from Orange.data import Table


def get_instance(data, i):
    """Extract a single instance from data as a test instance and return the remainder as a training set."""
    train, test = None, data[i]
    if i == 0: train = data[1:]
    elif i == len(data)-1: train = data[:-1]
    else: train = Table(data.domain, np.vstack((data[:i], data[i+1:])))
    return train, test

def split_data(data, a, b):
    """"Split data in approximate ratio a:b."""
    k = a*len(data)//(a+b)
    return data[:k], data[k:]

def shuffle_data(data):
    """Randomly shuffle data instances."""
    return data[np.random.permutation(len(data))]
