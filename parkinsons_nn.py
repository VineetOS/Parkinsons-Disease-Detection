#import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas import read_csv
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#load dataset
url = "parkinsons.data"
dataset = read_csv(url)

features=dataset.loc[:,dataset.columns!='status'].values[:,1:]
labels=dataset.loc[:,'status'].values

#feature scaling
scaler = MinMaxScaler((-1,1))
X = scaler.fit_transform(features)
y = labels

#cross-validation set
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=7)

