import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc

#Linear regression is a basic form of machine learning where we try to have 
#a linear correspondance between data points 
#had to do pip install sklearn

#Data set
#the data set is data from titanic 
#we are gonna try to perdict who will survive the titanic

#train data set
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')

#testing data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') 
print(dftrain.head())
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
print(dftrain.head())