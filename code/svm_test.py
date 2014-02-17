#!/usr/bin/env python
'''
###########################################################################
#
# CSCI-737 Pattern Recognition
# Fall, 2013
# Project #1
# 
# Author: Wei Yao (wxy3806_AT_rit.edu) & Fan Wang (fxw6000_AT_rit.edu)
# Date: Sep 30 2013
#
###########################################################################

This program uses libsvm to perform SVM classifictaion.
http://www.csie.ntu.edu.tw/~cjlin/libsvm/

'''

import svmutil


if __name__ == '__main__':
    y = [0, 1]
    x = [{1:0, 2:0},{1:1,2:1}]
    m = svmutil.svm_train(y, x, '-c 4')
    aY, p_acc, p_val = svmutil.svm_predict(y, x, m)