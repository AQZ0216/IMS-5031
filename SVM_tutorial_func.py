#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: winniehong

# Ref: https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
# Ref: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# Ref: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html

#import libs
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def Iris_SVM_visualization(X, y, C, save_dir):
    # only take first 2 features
    X = X[:,:2]
    clf = SVC(kernel='linear', C=C)
    clf.fit(X, y)

    # fig
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for i in range(2):
        if i == 1:
            plot_contours(axes[i], clf, xx, yy,
                          cmap=plt.cm.coolwarm, alpha=0.8)
        axes[i].scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        axes[i].set_xlim(xx.min(), xx.max())
        axes[i].set_ylim(yy.min(), yy.max())
        axes[i].set_xlabel('Sepal length')
        axes[i].set_ylabel('Sepal width')
        axes[i].set_xticks(())
        axes[i].set_yticks(())
        axes[i].set_title('Visulaize Iris data in first 2 dimensions')
        
    plt.savefig(save_dir+'Iris_SVM_visualization_in_first_two_feature.png')
    plt.show()

def plot_svc_decision_function(model, save_dir, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none',edgecolors='k');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.title('Toy Visualization of SVM')
    plt.savefig(save_dir+'Toy_2D_data_SVM_visualization_fig.png')
    plt.plot()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("CM - without normalization")
    else:
        print('CM - normalized result')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def Iris_cm_visualization(y_test, y_pred, class_names, save_dir):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='CM - without normalization')
    plt.tight_layout()
    plt.savefig(save_dir+'Iris_data_SVM_visualization_CM_non_normalized.png')
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='CM - normalized result')
    plt.tight_layout()
    plt.savefig(save_dir+'Iris_data_SVM_visualization_CM_normalized.png', )
    
    plt.show()