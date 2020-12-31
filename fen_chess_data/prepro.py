
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import cv2
import os

def load_dataset(ldim=128):

    print('Loading dataset..')

    X_train_names = np.array( open('train.txt',  'r').read().split('\n') )
    X_test_names  = np.array( open('test.txt',   'r').read().split('\n') )

    X_train, y_train = load_names(X_train_names, ldim)    
    X_test, y_test = load_names(X_test_names, ldim)    

    print('Dataset loaded')

    return X_train, X_test, y_train, y_test, X_train_names, X_test_names

def load_names(names, ldim=128):

    pieces = {c:i for i, c in enumerate('0PRNBQKprnbqk')}

    X, y = [], []

    for fname in names:

        img = cv2.imread(fname)

        lbl_ar = np.zeros(13)

        lbl = pieces[ fname.split('/')[-1][0] ]
        lbl_ar[lbl] = 1
        
        X.append(img)
        y.append(lbl_ar)

    return np.array(X), np.array(y)