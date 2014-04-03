#!/usr/bin/env python
'''
###########################################################################
#
# CROHME: Competition on Recognition of Online Handwritten 
#         Mathematical Expressions
# Spring, 2014
# 
# Author: Wei Yao (wxy3806_AT_rit.edu) & Fan Wang (fxw6000_AT_rit.edu)
# Date: Feb 17 2014
#
###########################################################################

Part of this program was rewritten of the perl script crohme2lg.pl, which is
a part of CROHMElib (http://www.cs.rit.edu/~dprl/Software.html).

Thanks to the authors: H. Mouchere and R. Zanibbi


This program uses libsvm to perform SVM classifictaion.
http://www.csie.ntu.edu.tw/~cjlin/libsvm/

Thanks to the authors: Chih-Chung Chang and Chih-Jen Lin 

'''

import numpy as np

def SVM2list(X):
    xlist = []
    for sub in X:
        newsub = []
        for idx in range(1,len(sub)+1):
            newsub.append(sub[idx])
        xlist.append(newsub)
    return xlist
# end of SVM2list

def list2SVM(x):
    xSVM = []
    for line in x:
        sub = {}
        fIdx = 1
        for a in line:
            sub[fIdx] = float(a)
            fIdx += 1
        xSVM.append(sub)
    return xSVM
# end of list2SVM(x)

def center(x, mu, sigma):
    'center the data using the mean and sigma from training set a'
    return (x - mu)/sigma
# end of center(x, mu, sigma):

def project(Wt, mu, sigma, x):
    'project x onto the principle axes, dropping any axes where fraction of variance<minfrac'
    x = np.asarray(x)

    if (x.shape[-1]!=Wt.shape[-1]):
        raise ValueError('Expected an array with dims[-1]==%d'%Wt.shape[-1])


    Y = np.dot(Wt, center(x, mu, sigma).T).T

    return Y
# end of project(Wt, mu, sigma, x):

def scaleData(x, cof=None):
    if cof == None:
        cof = {}
        xmax = {}
        xmin ={}
        for (k, v) in x[0].iteritems():
            xmax[k] = v 
            xmin[k] = v
        for l in x[1:]:
            for (k,v) in l.iteritems():
                if v > xmax[k]:
                    xmax[k] = v
                if v < xmin[k]:
                    xmin[k] = v
                    
        for k in x[0].iterkeys():
            c = [xmin[k], xmax[k]-xmin[k]]
            cof[k] = c
    
    for l in x:
        for k in l.iterkeys():
            l[k] = 2 * (l[k] - cof[k][0]) / cof[k][1] - 1
    return cof
# end of genDataSet

