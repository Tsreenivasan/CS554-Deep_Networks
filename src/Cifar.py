import numpy as np
import os
import tarfile
import pickle
import subprocess
import sys
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def cifar_for_library(channel_first=True, one_hot=False):
    # Raw data
    x_train, x_test, y_train, y_test = process_cifar()
    # Scale pixel intensity
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Reshape
    x_train = x_train.reshape(-1, 3, 32, 32)
    x_test = x_test.reshape(-1, 3, 32, 32)
    # Channel last
    if not channel_first:
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)
    # One-hot encode y
    if one_hot:
        y_train = np.expand_dims(y_train, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        enc = OneHotEncoder(categorical_features='all')
        fit = enc.fit(y_train)
        y_train = fit.transform(y_train).toarray()
        y_test = fit.transform(y_test).toarray()
    # dtypes
    x_train = x_train.astype(np.float64)
    x_test = x_test.astype(np.float64)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    return x_train, x_test, y_train, y_test

def process_cifar():
    '''Load data into RAM'''
    print('Preparing train set...')
    train_list = [read_batch('./cifar-10-batches-py/data_batch_{0}'.format(i + 1)) for i in range(5)]
    x_train = np.concatenate([t['data'] for t in train_list])
    y_train = np.concatenate([t['labels'] for t in train_list])
    print('Preparing test set...')
    tst = read_batch('./cifar-10-batches-py/test_batch')
    x_test = tst['data']
    y_test = np.asarray(tst['labels'])
    return x_train, x_test, y_train, y_test

def yield_mb(X, y, batchsize=64, shuffle=False):
    assert len(X) == len(y)
    if shuffle:
        X, y = shuffle_data(X, y)
    # Only complete batches are submitted
    for i in range(len(X) // batchsize):
        yield X[i * batchsize:(i + 1) * batchsize], y[i * batchsize:(i + 1) * batchsize]

def read_batch(src):
    '''Unpack the pickle files
    '''
    with open(src, 'rb') as f:
        if sys.version_info.major == 2:
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='latin1')
    return data

def shuffle_data(X, y):
    s = np.arange(len(X))
    np.random.shuffle(s)
    X = X[s]
    y = y[s]
    return X, y