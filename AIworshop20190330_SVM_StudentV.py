#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:35:36 2019

@author: winniehong
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: winniehong
# Copyright 2019 The Hands-on Workshop on Python, ML, and CNN. All Rights Reserved.
#
# Here we mainly use scikit-learn library provided by 
# authors: http://scikit-learn.org/stable/about.html#people and 
# contributing: http://scikit-learn.org/stable/developers/contributing.html
# 
# In this session, You will learn how to implement SVM-algorithm 
# base on scikit-learn library. Also, apply it on the Iris dataset 
# in order to classify the type of the Iris flower.
# Ref: http://scikit-learn.org/stable/modules/svm.html
# Ref: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
# Ref: https://medium.com/codebagng/basic-analysis-of-the-iris-data-set-using-python-2995618a6342

#%%
#import libs
import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pdb
from sklearn import datasets
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from SVM_tutorial_func import *

# Setting basic directory we need
output_fig_dir = './output_imgs/'
if not os.path.exists(output_fig_dir):
    os.makedirs(output_fig_dir)
#%%
# # =================================================================================================
# # PART 1
# # Here we start from a 2D-data toy implementation, which is 
# # to let you have a simple and clear concept of how SVM works. 
# # Since in higher-feature dimension data, like 
# # 4 dimension would be hard to visualize in the figure.
# # Ref: https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
# # =================================================================================================
# Simple 2D dataset
X, y = datasets.samples_generator.make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    
#%% 
# SVM
# Here you need to decide the C-value we are going to use in SVM,
# and also decide that which classifier you are going to use.
# hint: you will need scikit-learn library, such as SVC, LinearSVC, ....
C = 1.0
model = SVC(kernel='linear', C=C)
model.fit(X, y)  #hint : what is the function that could fit the SVM model according to the given training data
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

# visualization
plot_svc_decision_function(model, output_fig_dir);


#%%   
# =================================================================================================
# PART 2
# Here we will apply SVM on classifying the Iris-dataset
# Ref: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
# Ref: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
# 
# Iris dataset Information
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
# =================================================================================================
# load dataset
iris_dataset = datasets.load_iris()

# Here you need to know how to access data information
# from the build in library provided by scikit-learn.
# hint: google it! 
X = iris_dataset.data
y = iris_dataset.target
class_names = iris_dataset.target_names

# Split data into train and test
# Here you need to know what kind of input we need to feed in,
# when splitting the dataset.
# hint: Definitely need data and corresponding label!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)  # hint: Use the sklearn build-in-function to split the data into 80% training and 20% test set.

# visualize our train data
plt.figure()
plt.scatter(X_train[:,0],X_train[:,1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.show()

#%%
# Create the SVM classifier
C = 1.0
clf = SVC(kernel='linear', C=C)

# Train the model(SVM classifier) base on train-data
# hint: Since we are about to train the model, we definitely need train-data and train-label!
clf.fit(X, y)  #hint : what is the function that could fit the SVM model according to the given training data

# only take first 2 features in order to show classifier result
Iris_SVM_visualization(X_train, y_train, C, output_fig_dir)
#%% 
# predict on the test-data
# hint: Definitely need test-data and corresponding test-label to see the UAR!
y_pred = clf.predict(X_test)
print("Recall Score:",metrics.recall_score(y_test, y_pred, average='macro'))

# Visualization
print("Confusion Matrix: ")
Iris_cm_visualization(y_test, y_pred, class_names, output_fig_dir)
