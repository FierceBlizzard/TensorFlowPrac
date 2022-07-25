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
#head lets us see the data set 
print(dftrain.head())
#describe gives us stats on the data set
print(dftrain.describe())

#creating some graphs
print(dftrain.age.hist(bins=20))

#training data is used to create the model
#testing/eval data to correct the model
#we load data in batches because it's hard to throw large data sets in memory

#An epoch is simply one stream of our entire dataset
#start with a low amount of epochs and then slowly improve

#input functions essential figure out how our data is broken into epochs and batches
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=1):
    def input_function():
        #creates tf.data.Dataset object with data and its label
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            #randomize the order of the data
            ds= ds.shuffle(1000)
        #splits the dataset into branches of 32 and repeat process for number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)
        #returns the batch of data set
        return ds
    #return a function object for use
    return input_function
#Making functions for each dataset
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)