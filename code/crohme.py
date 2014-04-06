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

import InkML
import os
import subprocess
import pickle
from numpy import random
import numpy as np
import svmutil
import time
import multiprocessing
from matplotlib.mlab import PCA
import itertools
from routines import SVM2list, list2SVM, project, scaleData

# define the path where the training data is located
TrainDataPath = "../TrainINKML_v3"
MaxSwap = 1000;
Nfolds = 3;

nSegPCA = 100

__location__ = os.path.realpath(os.path.join(os.getcwd(), \
os.path.dirname(__file__)))

allIMcache = os.path.join(__location__, "allIM.cache")
testcache = os.path.join(__location__, "test.cache")

featureList = ['NofStrokes', 'normalizedY', 'cos_slope', 'sin_curvature', 'density4', 'normalizedWidth','NofPoints']
segFeatureList = ['G', 'SPSCF', 'LNSCF', 'GSCF', 'C']

grmToSRT = {
    'LIMITWORD:SUB_SUBLIMIT':'Sub',     #LIMIT -> LIMITWORD SUB_SUBLIMIT EXP
    'LIMITWORD:EXP':'R',     #LIMIT -> LIMITWORD SUB_SUBLIMIT EXP
    'LIMITWORD:EXP':'R',     #LIMIT -> LIMITWORD EXP
    'OPSUM:TERM_OR_BRACED_LR':'R',     #EXP -> OPSUM TERM_OR_BRACED_LR
    'OPSUM:LEFTRIGHTPAIR':'R',     #EXP -> OPSUM LEFTRIGHTPAIR
#     'MUTE_LBRACE:OPSUM':'R',     #EXP -> MUTE_LBRACE OPSUM TERM MUTE_RBRACE EXP
#     'MUTE_LBRACE:TERM':'R',     #EXP -> MUTE_LBRACE OPSUM TERM MUTE_RBRACE EXP
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #EXP -> MUTE_LBRACE OPSUM TERM MUTE_RBRACE EXP
#     'MUTE_LBRACE:EXP':'R',     #EXP -> MUTE_LBRACE OPSUM TERM MUTE_RBRACE EXP
    'OPSUM:TERM':'R',     #EXP -> MUTE_LBRACE OPSUM TERM MUTE_RBRACE EXP
#     'OPSUM:MUTE_RBRACE':'R',     #EXP -> MUTE_LBRACE OPSUM TERM MUTE_RBRACE EXP
    'OPSUM:EXP':'R',     #EXP -> MUTE_LBRACE OPSUM TERM MUTE_RBRACE EXP
#     'TERM:MUTE_RBRACE':'R',     #EXP -> MUTE_LBRACE OPSUM TERM MUTE_RBRACE EXP
    'TERM:EXP':'R',     #EXP -> MUTE_LBRACE OPSUM TERM MUTE_RBRACE EXP
#     'MUTE_RBRACE:EXP':'R',     #EXP -> MUTE_LBRACE OPSUM TERM MUTE_RBRACE EXP
#     'MUTE_LBRACE:EXP_OR_BRACED_EXP':'R',     #SYMB_OR_BRACED_EXP -> MUTE_LBRACE EXP_OR_BRACED_EXP MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #SYMB_OR_BRACED_EXP -> MUTE_LBRACE EXP_OR_BRACED_EXP MUTE_RBRACE
#     'EXP_OR_BRACED_EXP:MUTE_RBRACE':'R',     #SYMB_OR_BRACED_EXP -> MUTE_LBRACE EXP_OR_BRACED_EXP MUTE_RBRACE
    'SUP:FRACTION':'R',     #SUP_FRACTION -> SUP FRACTION
    'SUP:DIGIT':'R',     #POWER -> SUP DIGIT
#     'MUTE_LBRACE:SYMB_OR_BRACED_SYMB':'R',     #SUBSERIES -> MUTE_LBRACE SYMB_OR_BRACED_SYMB OPEQ EXP MUTE_RBRACE
#     'MUTE_LBRACE:OPEQ':'R',     #SUBSERIES -> MUTE_LBRACE SYMB_OR_BRACED_SYMB OPEQ EXP MUTE_RBRACE
#     'MUTE_LBRACE:EXP':'R',     #SUBSERIES -> MUTE_LBRACE SYMB_OR_BRACED_SYMB OPEQ EXP MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #SUBSERIES -> MUTE_LBRACE SYMB_OR_BRACED_SYMB OPEQ EXP MUTE_RBRACE
    'SYMB_OR_BRACED_SYMB:OPEQ':'R',     #SUBSERIES -> MUTE_LBRACE SYMB_OR_BRACED_SYMB OPEQ EXP MUTE_RBRACE
    'SYMB_OR_BRACED_SYMB:EXP':'R',     #SUBSERIES -> MUTE_LBRACE SYMB_OR_BRACED_SYMB OPEQ EXP MUTE_RBRACE
    'SYMB_OR_BRACED_SYMB:MUTE_RBRACE':'R',     #SUBSERIES -> MUTE_LBRACE SYMB_OR_BRACED_SYMB OPEQ EXP MUTE_RBRACE
    'OPEQ:EXP':'R',     #SUBSERIES -> MUTE_LBRACE SYMB_OR_BRACED_SYMB OPEQ EXP MUTE_RBRACE
#     'OPEQ:MUTE_RBRACE':'R',     #SUBSERIES -> MUTE_LBRACE SYMB_OR_BRACED_SYMB OPEQ EXP MUTE_RBRACE
#     'EXP:MUTE_RBRACE':'R',     #SUBSERIES -> MUTE_LBRACE SYMB_OR_BRACED_SYMB OPEQ EXP MUTE_RBRACE
#     'MUTE_LBRACE:SUBSERIES':'R',     #SUBSERIES -> MUTE_LBRACE SUBSERIES MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #SUBSERIES -> MUTE_LBRACE SUBSERIES MUTE_RBRACE
#     'SUBSERIES:MUTE_RBRACE':'R',     #SUBSERIES -> MUTE_LBRACE SUBSERIES MUTE_RBRACE
    'MUTE_LBRACE:LEFTRIGHTPAIR':'R',     #TERM_OR_BRACED_LR -> MUTE_LBRACE LEFTRIGHTPAIR MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #TERM_OR_BRACED_LR -> MUTE_LBRACE LEFTRIGHTPAIR MUTE_RBRACE
#     'LEFTRIGHTPAIR:MUTE_RBRACE':'R',     #TERM_OR_BRACED_LR -> MUTE_LBRACE LEFTRIGHTPAIR MUTE_RBRACE
    'SUB:SUBLIMIT':'R',     #SUB_SUBLIMIT -> SUB SUBLIMIT
    'MUTE_LBRACE:OPSUM':'R',     #OPSUM -> MUTE_LBRACE OPSUM MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #OPSUM -> MUTE_LBRACE OPSUM MUTE_RBRACE
#     'OPSUM:MUTE_RBRACE':'R',     #OPSUM -> MUTE_LBRACE OPSUM MUTE_RBRACE
    'EXP_OR_BRACED_EXP:COMMA':'R',     #EXP_LIST -> EXP_OR_BRACED_EXP COMMA EXP_LIST_R
    'EXP_OR_BRACED_EXP:EXP_LIST_R':'R',     #EXP_LIST -> EXP_OR_BRACED_EXP COMMA EXP_LIST_R
    'COMMA:EXP_LIST_R':'R',     #EXP_LIST -> EXP_OR_BRACED_EXP COMMA EXP_LIST_R
    'FUNCTRIGO:SUP_SYMB_OR_BRACED_SYMB':'Sup',     #FUNCTION -> FUNCTRIGO SUP_SYMB_OR_BRACED_SYMB
    'FUNCLOG:SUB_SYMB_OR_BRACED_SYMB':'Sub',     #FUNCTION -> FUNCLOG SUB_SYMB_OR_BRACED_SYMB
#     'MUTE_LBRACE:FORMULA':'R',     #FORMULA -> MUTE_LBRACE FORMULA MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #FORMULA -> MUTE_LBRACE FORMULA MUTE_RBRACE
#     'FORMULA:MUTE_RBRACE':'R',     #FORMULA -> MUTE_LBRACE FORMULA MUTE_RBRACE
    'EXP_OR_BRACED_EXP:OPEQ':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'EXP_OR_BRACED_EXP:EXP_OR_BRACED_EXP':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'OPEQ:EXP_OR_BRACED_EXP':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'EXP_OR_BRACED_EXP:OPEQ':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'EXP_OR_BRACED_EXP:EXP_OR_BRACED_EXP':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'EXP_OR_BRACED_EXP:OPEQ':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'EXP_OR_BRACED_EXP:EXP_OR_BRACED_EXP':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'OPEQ:EXP_OR_BRACED_EXP':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'OPEQ:OPEQ':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'OPEQ:EXP_OR_BRACED_EXP':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'EXP_OR_BRACED_EXP:OPEQ':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'EXP_OR_BRACED_EXP:EXP_OR_BRACED_EXP':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'OPEQ:EXP_OR_BRACED_EXP':'R',     #FORMULA -> EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP OPEQ EXP_OR_BRACED_EXP
    'QUANTIFIER_LIST:FORMULA':'R',     #FORMULA -> QUANTIFIER_LIST FORMULA
#     'MUTE_LBRACE:FUNCLOG':'R',     #FUNCLOG -> MUTE_LBRACE FUNCLOG MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #FUNCLOG -> MUTE_LBRACE FUNCLOG MUTE_RBRACE
#     'FUNCLOG:MUTE_RBRACE':'R',     #FUNCLOG -> MUTE_LBRACE FUNCLOG MUTE_RBRACE
#     'MUTE_LBRACE:FUNCTRIGO':'R',     #FUNCTRIGO -> MUTE_LBRACE FUNCTRIGO MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #FUNCTRIGO -> MUTE_LBRACE FUNCTRIGO MUTE_RBRACE
#     'FUNCTRIGO:MUTE_RBRACE':'R',     #FUNCTRIGO -> MUTE_LBRACE FUNCTRIGO MUTE_RBRACE
    'FRACTIONBAR:NUMERATOR':'A',     #FRACTION -> FRACTIONBAR NUMERATOR DENOMINATOR
    'FRACTIONBAR:DENOMINATOR':'B',     #FRACTION -> FRACTIONBAR NUMERATOR DENOMINATOR
    'DOT:DOT':'R',     #DOTS -> DOT DOT DOT
    'DOT:DOT':'R',     #DOTS -> DOT DOT DOT
    'DOT:DOT':'R',     #DOTS -> DOT DOT DOT
    'SUB:SUBSERIES':'R',     #SUB_SUBSERIES -> SUB SUBSERIES
#     'MUTE_LBRACE:EXP_OR_BRACED_EXP':'R',     #TERM_OR_BRACED_EXP -> MUTE_LBRACE EXP_OR_BRACED_EXP MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #TERM_OR_BRACED_EXP -> MUTE_LBRACE EXP_OR_BRACED_EXP MUTE_RBRACE
#     'EXP_OR_BRACED_EXP:MUTE_RBRACE':'R',     #TERM_OR_BRACED_EXP -> MUTE_LBRACE EXP_OR_BRACED_EXP MUTE_RBRACE
    'SUP:SYMB_OR_BRACED_EXP':'R',     #SUP_SYMB_OR_BRACED_EXP -> SUP SYMB_OR_BRACED_EXP
    'LETTER:QUANTIFIER_OP':'R',     #RANGE -> LETTER QUANTIFIER_OP SET_EXP
    'LETTER:SET_EXP':'R',     #RANGE -> LETTER QUANTIFIER_OP SET_EXP
    'QUANTIFIER_OP:SET_EXP':'R',     #RANGE -> LETTER QUANTIFIER_OP SET_EXP
    'LETTER:QUANTIFIER_OP':'R',     #RANGE -> LETTER QUANTIFIER_OP PAREXP
    'LETTER:PAREXP':'R',     #RANGE -> LETTER QUANTIFIER_OP PAREXP
    'QUANTIFIER_OP:PAREXP':'R',     #RANGE -> LETTER QUANTIFIER_OP PAREXP
    'QUANTIFIER_ONE:COMMA':'R',     #QUANTIFIER_LIST -> QUANTIFIER_ONE COMMA
    'QUANTIFIER_SYMB:LETTER':'R',     #QUANTIFIER_ONE -> QUANTIFIER_SYMB LETTER
    'QUANTIFIER_SYMB:RANGE':'R',     #QUANTIFIER_ONE -> QUANTIFIER_SYMB RANGE
    'LETTER:LETTER':'R',     #WORD -> LETTER LETTER
    'LETTER:WORD':'R',     #WORD -> LETTER WORD
#     'MUTE_LBRACE:WORD':'R',     #WORD -> MUTE_LBRACE WORD MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #WORD -> MUTE_LBRACE WORD MUTE_RBRACE
#     'WORD:MUTE_RBRACE':'R',     #WORD -> MUTE_LBRACE WORD MUTE_RBRACE
#     'MUTE_LBRACE:EXP_OR_BRACED_EXP':'R',     #EXP_OR_BRACED_EXP -> MUTE_LBRACE EXP_OR_BRACED_EXP MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #EXP_OR_BRACED_EXP -> MUTE_LBRACE EXP_OR_BRACED_EXP MUTE_RBRACE
#     'EXP_OR_BRACED_EXP:MUTE_RBRACE':'R',     #EXP_OR_BRACED_EXP -> MUTE_LBRACE EXP_OR_BRACED_EXP MUTE_RBRACE
#     'MUTE_LBRACE:LEFTRIGHTPAIR':'R',     #LREND -> MUTE_LBRACE LEFTRIGHTPAIR MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #LREND -> MUTE_LBRACE LEFTRIGHTPAIR MUTE_RBRACE
#     'LEFTRIGHTPAIR:MUTE_RBRACE':'R',     #LREND -> MUTE_LBRACE LEFTRIGHTPAIR MUTE_RBRACE
#     'MUTE_LBRACE:EXP':'R',     #CONTISUM_END -> MUTE_LBRACE EXP MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #CONTISUM_END -> MUTE_LBRACE EXP MUTE_RBRACE
#     'EXP:MUTE_RBRACE':'R',     #CONTISUM_END -> MUTE_LBRACE EXP MUTE_RBRACE
#     'SUB:MUTE_LBRACE':'R',     #SUB_EXP_LIST -> SUB MUTE_LBRACE EXP_LIST MUTE_RBRACE
    'SUB:EXP_LIST':'R',     #SUB_EXP_LIST -> SUB MUTE_LBRACE EXP_LIST MUTE_RBRACE
#     'SUB:MUTE_RBRACE':'R',     #SUB_EXP_LIST -> SUB MUTE_LBRACE EXP_LIST MUTE_RBRACE
#     'MUTE_LBRACE:EXP_LIST':'R',     #SUB_EXP_LIST -> SUB MUTE_LBRACE EXP_LIST MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #SUB_EXP_LIST -> SUB MUTE_LBRACE EXP_LIST MUTE_RBRACE
#     'EXP_LIST:MUTE_RBRACE':'R',     #SUB_EXP_LIST -> SUB MUTE_LBRACE EXP_LIST MUTE_RBRACE
    'BIGOP:TERM_OR_BRACED_EXP':'R',     #SERIES -> BIGOP TERM_OR_BRACED_EXP
    'BIGOP:SUB_SUBSERIES':'Sub',     #SERIES -> BIGOP SUB_SUBSERIES TERM_OR_BRACED_EXP
    'BIGOP:TERM_OR_BRACED_EXP':'R',     #SERIES -> BIGOP SUB_SUBSERIES TERM_OR_BRACED_EXP
    'BIGOP:SUB_SUBSERIES':'Sub',     #SERIES -> BIGOP SUB_SUBSERIES SUP_SYMB_OR_BRACED_EXP TERM_OR_BRACED_EXP
    'BIGOP:SUP_SYMB_OR_BRACED_EXP':'Sup',     #SERIES -> BIGOP SUB_SUBSERIES SUP_SYMB_OR_BRACED_EXP TERM_OR_BRACED_EXP
    'BIGOP:TERM_OR_BRACED_EXP':'R',     #SERIES -> BIGOP SUB_SUBSERIES SUP_SYMB_OR_BRACED_EXP TERM_OR_BRACED_EXP
    'BIGOP:SUP_SYMB_OR_BRACED_EXP':'Sup',     #SERIES -> BIGOP SUP_SYMB_OR_BRACED_EXP SUB_SUBSERIES TERM_OR_BRACED_EXP
    'BIGOP:SUB_SUBSERIES':'Sub',     #SERIES -> BIGOP SUP_SYMB_OR_BRACED_EXP SUB_SUBSERIES TERM_OR_BRACED_EXP
    'BIGOP:TERM_OR_BRACED_EXP':'R',     #SERIES -> BIGOP SUP_SYMB_OR_BRACED_EXP SUB_SUBSERIES TERM_OR_BRACED_EXP
    'EXP_OR_BRACED_EXP:COMMA':'R',     #EXP_LIST_R -> EXP_OR_BRACED_EXP COMMA EXP_LIST_R
    'EXP_OR_BRACED_EXP:EXP_LIST_R':'R',     #EXP_LIST_R -> EXP_OR_BRACED_EXP COMMA EXP_LIST_R
    'COMMA:EXP_LIST_R':'R',     #EXP_LIST_R -> EXP_OR_BRACED_EXP COMMA EXP_LIST_R
    'DOTS:COMMA':'R',     #EXP_LIST_R -> DOTS COMMA EXP_LIST_R
    'DOTS:EXP_LIST_R':'R',     #EXP_LIST_R -> DOTS COMMA EXP_LIST_R
    'COMMA:EXP_LIST_R':'R',     #EXP_LIST_R -> DOTS COMMA EXP_LIST_R
    'LEFTRIGHTPAIR:OPSP':'R',     #LEFTRIGHTPAIR -> LEFTRIGHTPAIR OPSP LREND
    'LEFTRIGHTPAIR:LREND':'R',     #LEFTRIGHTPAIR -> LEFTRIGHTPAIR OPSP LREND
    'OPSP:LREND':'R',     #LEFTRIGHTPAIR -> LEFTRIGHTPAIR OPSP LREND
    'LEFTRIGHTPAIR:LREND':'R',     #LEFTRIGHTPAIR -> LEFTRIGHTPAIR LREND
    'LREND:LREND':'R',     #LEFTRIGHTPAIR -> LREND LREND
    'LREND:OPSP':'R',     #LEFTRIGHTPAIR -> LREND OPSP LREND
    'LREND:LREND':'R',     #LEFTRIGHTPAIR -> LREND OPSP LREND
    'OPSP:LREND':'R',     #LEFTRIGHTPAIR -> LREND OPSP LREND
    'SUB:SYMB_OR_BRACED_EXP':'R',     #SUB_SYMB_OR_BRACED_EXP -> SUB SYMB_OR_BRACED_EXP
    'LETTER:ARROW':'R',     #EXP_F_DEF -> LETTER ARROW EXP
    'LETTER:EXP':'R',     #EXP_F_DEF -> LETTER ARROW EXP
    'ARROW:EXP':'R',     #EXP_F_DEF -> LETTER ARROW EXP
    'LETTER:ARROW':'R',     #EXP_F_DEF -> LETTER ARROW EXP_F_DEF
    'LETTER:EXP_F_DEF':'R',     #EXP_F_DEF -> LETTER ARROW EXP_F_DEF
    'ARROW:EXP_F_DEF':'R',     #EXP_F_DEF -> LETTER ARROW EXP_F_DEF
    'SETS:SET_OP':'R',     #SET_EXP -> SETS SET_OP SET_EXP
    'SETS:SET_EXP':'R',     #SET_EXP -> SETS SET_OP SET_EXP
    'SET_OP:SET_EXP':'R',     #SET_EXP -> SETS SET_OP SET_EXP
    'SETS:POWER':'Sup',     #SET_EXP -> SETS POWER
    'OPENP:EXP_OR_BRACED_EXP':'R',     #PAREXP -> OPENP EXP_OR_BRACED_EXP CLOSEP
    'OPENP:CLOSEP':'R',     #PAREXP -> OPENP EXP_OR_BRACED_EXP CLOSEP
    'EXP_OR_BRACED_EXP:CLOSEP':'R',     #PAREXP -> OPENP EXP_OR_BRACED_EXP CLOSEP
    'OPENP:EXP_LIST':'R',     #PAREXP -> OPENP EXP_LIST CLOSEP
    'OPENP:CLOSEP':'R',     #PAREXP -> OPENP EXP_LIST CLOSEP
    'EXP_LIST:CLOSEP':'R',     #PAREXP -> OPENP EXP_LIST CLOSEP
    'OPEN_BRACKET:EXP_LIST':'R',     #PAREXP -> OPEN_BRACKET EXP_LIST CLOSE_BRACKET
    'OPEN_BRACKET:CLOSE_BRACKET':'R',     #PAREXP -> OPEN_BRACKET EXP_LIST CLOSE_BRACKET
    'EXP_LIST:CLOSE_BRACKET':'R',     #PAREXP -> OPEN_BRACKET EXP_LIST CLOSE_BRACKET
    'OPEN_BRACKET:EXP_OR_BRACED_EXP':'R',     #PAREXP -> OPEN_BRACKET EXP_OR_BRACED_EXP CLOSE_BRACKET
    'OPEN_BRACKET:CLOSE_BRACKET':'R',     #PAREXP -> OPEN_BRACKET EXP_OR_BRACED_EXP CLOSE_BRACKET
    'EXP_OR_BRACED_EXP:CLOSE_BRACKET':'R',     #PAREXP -> OPEN_BRACKET EXP_OR_BRACED_EXP CLOSE_BRACKET
    'OPEN_BRACE:EXP_LIST':'R',     #PAREXP -> OPEN_BRACE EXP_LIST CLOSE_BRACE
    'OPEN_BRACE:CLOSE_BRACE':'R',     #PAREXP -> OPEN_BRACE EXP_LIST CLOSE_BRACE
    'EXP_LIST:CLOSE_BRACE':'R',     #PAREXP -> OPEN_BRACE EXP_LIST CLOSE_BRACE
    'OPEN_BRACE:EXP_OR_BRACED_EXP':'R',     #PAREXP -> OPEN_BRACE EXP_OR_BRACED_EXP CLOSE_BRACE
    'OPEN_BRACE:CLOSE_BRACE':'R',     #PAREXP -> OPEN_BRACE EXP_OR_BRACED_EXP CLOSE_BRACE
    'EXP_OR_BRACED_EXP:CLOSE_BRACE':'R',     #PAREXP -> OPEN_BRACE EXP_OR_BRACED_EXP CLOSE_BRACE
    'SYMB_OR_BRACED_SYMB:FACTORIAL':'R',     #TERM -> SYMB_OR_BRACED_SYMB FACTORIAL
    'FUNCTION:TERM_OR_BRACED_LR':'R',     #TERM -> FUNCTION TERM_OR_BRACED_LR
    'TERM_OR_BRACED_LR:SUP_SYMB_OR_BRACED_EXP':'Sup',     #TERM -> TERM_OR_BRACED_LR SUP_SYMB_OR_BRACED_EXP
    'TERM_OR_BRACED_LR:SUP_SYMB_OR_BRACED_EXP':'Sup',     #TERM -> TERM_OR_BRACED_LR SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP
    'TERM_OR_BRACED_LR:SUB_SYMB_OR_BRACED_EXP':'Sub',     #TERM -> TERM_OR_BRACED_LR SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP
    'TERM_OR_BRACED_LR:SUP_FRACTION':'Sup',     #TERM -> TERM_OR_BRACED_LR SUP_FRACTION
    'PAREXP:SUP_SYMB_OR_BRACED_EXP':'Sup',     #TERM -> PAREXP SUP_SYMB_OR_BRACED_EXP
    'SYMB_OR_BRACED_SYMB:SUB_SYMB_OR_BRACED_EXP':'Sub',     #TERM -> SYMB_OR_BRACED_SYMB SUB_SYMB_OR_BRACED_EXP
#     'MUTE_LBRACE:WORD':'R',     #TERM -> MUTE_LBRACE WORD MUTE_RBRACE SUB_SYMB_OR_BRACED_EXP
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #TERM -> MUTE_LBRACE WORD MUTE_RBRACE SUB_SYMB_OR_BRACED_EXP
#     'MUTE_LBRACE:SUB_SYMB_OR_BRACED_EXP':'Sub',     #TERM -> MUTE_LBRACE WORD MUTE_RBRACE SUB_SYMB_OR_BRACED_EXP
#     'WORD:MUTE_RBRACE':'R',     #TERM -> MUTE_LBRACE WORD MUTE_RBRACE SUB_SYMB_OR_BRACED_EXP
    'WORD:SUB_SYMB_OR_BRACED_EXP':'Sub',     #TERM -> MUTE_LBRACE WORD MUTE_RBRACE SUB_SYMB_OR_BRACED_EXP
#     'MUTE_RBRACE:SUB_SYMB_OR_BRACED_EXP':'Sub',     #TERM -> MUTE_LBRACE WORD MUTE_RBRACE SUB_SYMB_OR_BRACED_EXP
    'SYMB_OR_BRACED_SYMB:SUB_EXP_LIST_MUTE_RBRACE':'Sub',     #TERM -> SYMB_OR_BRACED_SYMB SUB_EXP_LIST_MUTE_RBRACE
    'LETTER:PRIME':'R',     #TERM -> LETTER PRIME
    'SQRT:SYMB_OR_BRACED_EXP':'I',     #TERM -> SQRT SYMB_OR_BRACED_EXP
#     'SQRT:LBRACKET':'R',     #TERM -> SQRT LBRACKET SYMBOL RBRACKET SYMB_OR_BRACED_EXP
    'SQRT:SYMBOL':'A',     #TERM -> SQRT LBRACKET SYMBOL RBRACKET SYMB_OR_BRACED_EXP
#     'SQRT:RBRACKET':'R',     #TERM -> SQRT LBRACKET SYMBOL RBRACKET SYMB_OR_BRACED_EXP
    'SQRT:SYMB_OR_BRACED_EXP':'I',     #TERM -> SQRT LBRACKET SYMBOL RBRACKET SYMB_OR_BRACED_EXP
#     'LBRACKET:SYMBOL':'R',     #TERM -> SQRT LBRACKET SYMBOL RBRACKET SYMB_OR_BRACED_EXP
#     'LBRACKET:RBRACKET':'R',     #TERM -> SQRT LBRACKET SYMBOL RBRACKET SYMB_OR_BRACED_EXP
#     'LBRACKET:SYMB_OR_BRACED_EXP':'R',     #TERM -> SQRT LBRACKET SYMBOL RBRACKET SYMB_OR_BRACED_EXP
#     'SYMBOL:RBRACKET':'R',     #TERM -> SQRT LBRACKET SYMBOL RBRACKET SYMB_OR_BRACED_EXP
    'SYMBOL:SYMB_OR_BRACED_EXP':'R',     #TERM -> SQRT LBRACKET SYMBOL RBRACKET SYMB_OR_BRACED_EXP
    'RBRACKET:SYMB_OR_BRACED_EXP':'R',     #TERM -> SQRT LBRACKET SYMBOL RBRACKET SYMB_OR_BRACED_EXP
    'OPENVBAR:EXP_OR_BRACED_EXP':'R',     #TERM -> OPENVBAR EXP_OR_BRACED_EXP CLOSEVBAR
    'OPENVBAR:CLOSEVBAR':'R',     #TERM -> OPENVBAR EXP_OR_BRACED_EXP CLOSEVBAR
    'EXP_OR_BRACED_EXP:CLOSEVBAR':'R',     #TERM -> OPENVBAR EXP_OR_BRACED_EXP CLOSEVBAR
#     'MUTE_LBRACE:TERM':'R',     #TERM -> MUTE_LBRACE TERM MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #TERM -> MUTE_LBRACE TERM MUTE_RBRACE
#     'TERM:MUTE_RBRACE':'R',     #TERM -> MUTE_LBRACE TERM MUTE_RBRACE
#     'MUTE_LBRACE:OPPROD':'R',     #OPPROD -> MUTE_LBRACE OPPROD MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #OPPROD -> MUTE_LBRACE OPPROD MUTE_RBRACE
#     'OPPROD:MUTE_RBRACE':'R',     #OPPROD -> MUTE_LBRACE OPPROD MUTE_RBRACE
    'INTEGRAL:SUB_SYMB_OR_BRACED_EXP':'Sub',     #CONTISUM -> INTEGRAL SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP CONTISUM_END
    'INTEGRAL:SUP_SYMB_OR_BRACED_EXP':'Sup',     #CONTISUM -> INTEGRAL SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP CONTISUM_END
    'INTEGRAL:CONTISUM_END':'R',     #CONTISUM -> INTEGRAL SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP CONTISUM_END
    'INTEGRAL:SUP_SYMB_OR_BRACED_EXP':'Sup',     #CONTISUM -> INTEGRAL SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP CONTISUM_END
    'INTEGRAL:SUB_SYMB_OR_BRACED_EXP':'Sub',     #CONTISUM -> INTEGRAL SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP CONTISUM_END
    'INTEGRAL:CONTISUM_END':'R',     #CONTISUM -> INTEGRAL SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP CONTISUM_END
    'INTEGRAL:CONTISUM_END':'R',     #CONTISUM -> INTEGRAL CONTISUM_END
#     'MUTE_LBRACE:SYMB_OR_BRACED_SYMB':'R',     #SYMB_OR_BRACED_SYMB -> MUTE_LBRACE SYMB_OR_BRACED_SYMB MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #SYMB_OR_BRACED_SYMB -> MUTE_LBRACE SYMB_OR_BRACED_SYMB MUTE_RBRACE
#     'SYMB_OR_BRACED_SYMB:MUTE_RBRACE':'R',     #SYMB_OR_BRACED_SYMB -> MUTE_LBRACE SYMB_OR_BRACED_SYMB MUTE_RBRACE
    'OPEN_BRACKET:EXP':'R',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP
    'OPEN_BRACKET:CLOSE_BRACKET':'R',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP
    'OPEN_BRACKET:SUP_SYMB_OR_BRACED_EXP':'Sup',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP
    'OPEN_BRACKET:SUB_SYMB_OR_BRACED_EXP':'Sub',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP
    'EXP:CLOSE_BRACKET':'R',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP
    'EXP:SUP_SYMB_OR_BRACED_EXP':'Sup',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP
    'EXP:SUB_SYMB_OR_BRACED_EXP':'Sub',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP
    'CLOSE_BRACKET:SUP_SYMB_OR_BRACED_EXP':'Sup',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP
    'CLOSE_BRACKET:SUB_SYMB_OR_BRACED_EXP':'Sub',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUP_SYMB_OR_BRACED_EXP SUB_SYMB_OR_BRACED_EXP
    'OPEN_BRACKET:EXP':'R',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP
    'OPEN_BRACKET:CLOSE_BRACKET':'R',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP
    'OPEN_BRACKET:SUB_SYMB_OR_BRACED_EXP':'Sub',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP
    'OPEN_BRACKET:SUP_SYMB_OR_BRACED_EXP':'Sup',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP
    'EXP:CLOSE_BRACKET':'R',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP
    'EXP:SUB_SYMB_OR_BRACED_EXP':'Sub',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP
    'EXP:SUP_SYMB_OR_BRACED_EXP':'Sup',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP
    'CLOSE_BRACKET:SUB_SYMB_OR_BRACED_EXP':'Sub',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP
    'CLOSE_BRACKET:SUP_SYMB_OR_BRACED_EXP':'Sup',     #INTEGRATION -> OPEN_BRACKET EXP CLOSE_BRACKET SUB_SYMB_OR_BRACED_EXP SUP_SYMB_OR_BRACED_EXP
#     'MUTE_LBRACE:SYMB_OR_BRACED_SYMB':'R',     #SUBLIMIT -> MUTE_LBRACE SYMB_OR_BRACED_SYMB ARROW EXP MUTE_RBRACE
#     'MUTE_LBRACE:ARROW':'R',     #SUBLIMIT -> MUTE_LBRACE SYMB_OR_BRACED_SYMB ARROW EXP MUTE_RBRACE
#     'MUTE_LBRACE:EXP':'R',     #SUBLIMIT -> MUTE_LBRACE SYMB_OR_BRACED_SYMB ARROW EXP MUTE_RBRACE
#     'MUTE_LBRACE:MUTE_RBRACE':'R',     #SUBLIMIT -> MUTE_LBRACE SYMB_OR_BRACED_SYMB ARROW EXP MUTE_RBRACE
    'SYMB_OR_BRACED_SYMB:ARROW':'R',     #SUBLIMIT -> MUTE_LBRACE SYMB_OR_BRACED_SYMB ARROW EXP MUTE_RBRACE
    'SYMB_OR_BRACED_SYMB:EXP':'R',     #SUBLIMIT -> MUTE_LBRACE SYMB_OR_BRACED_SYMB ARROW EXP MUTE_RBRACE
#     'SYMB_OR_BRACED_SYMB:MUTE_RBRACE':'R',     #SUBLIMIT -> MUTE_LBRACE SYMB_OR_BRACED_SYMB ARROW EXP MUTE_RBRACE
    'ARROW:EXP':'R',     #SUBLIMIT -> MUTE_LBRACE SYMB_OR_BRACED_SYMB ARROW EXP MUTE_RBRACE
#     'ARROW:MUTE_RBRACE':'R',     #SUBLIMIT -> MUTE_LBRACE SYMB_OR_BRACED_SYMB ARROW EXP MUTE_RBRACE
    'EXP:MUTE_RBRACE':'R',     #SUBLIMIT -> MUTE_LBRACE SYMB_OR_BRACED_SYMB ARROW EXP MUTE_RBRACE
    'DIGIT:DECIMALSEP':'R',     #FLOAT -> DIGIT DECIMALSEP DIGIT
    'DIGIT:DIGIT':'R',     #FLOAT -> DIGIT DECIMALSEP DIGIT
    'DECIMALSEP:DIGIT':'R'     #FLOAT -> DIGIT DECIMALSEP DIGIT
        }

def buildTrainDataList(base, current, DataList):
    child_nodes = os.listdir(os.path.join(base, current))
    for dir_nodes in [c for c in child_nodes if os.path.isdir(os.path.join(base, current, c))]:
        buildTrainDataList(base, os.path.join(current,dir_nodes), DataList)
    for file_nodes in [c for c in child_nodes if os.path.isfile(os.path.join(base, current, c))]:
        if file_nodes.lower().endswith('.inkml'):
            DataList.append(os.path.join(current,file_nodes))
    pass
# end of buildTrainDataList

def generateTGLG():
    TrainDataList = [];
    buildTrainDataList(TrainDataPath, '', TrainDataList)
    myOutputPath = "../my"
    
    crohme2lg = "/Volumes/Home/wxy3806/Study/ThirdYear/CSCI-737_Pattern_Recognition/CROHMELib/bin/crohme2lg.pl"
    truthOutputPath = "../Truth"
    
    sid = 0
    for l in TrainDataList[sid:]:
        print "file %5d: reading %s" % (sid, l)
        inPath = os.path.join(TrainDataPath, l)
        im = InkML.InkML(inPath)

        outfile = l.lower().replace('.inkml', '.lg')
        
        print "            writing %s ..." % outfile,
        truthPath = os.path.join(truthOutputPath, outfile)
        if not os.path.exists(os.path.dirname(truthPath)):
            os.makedirs(os.path.dirname(truthPath))
        subprocess.call([crohme2lg, inPath, truthPath])
        print " Done!"
         

        print "            writing %s ..." % outfile,
        outPath = os.path.join(myOutputPath,outfile)
        if not os.path.exists(os.path.dirname(outPath)):
            os.makedirs(os.path.dirname(outPath))
        f = file(outPath, 'w')
        im.printTruthLG(f)
        f.close
        print " Done! "
        
        sid += 1

# end of generateTGLG()

def readAllTrainData(Path):
    TrainDataList = [];
    buildTrainDataList(Path, '', TrainDataList)
    TrainData = [InkML.InkML(os.path.join(Path,f)) for f in TrainDataList]
    return TrainData
# end of readAllTrainData

def getAllSymbols(IMData):
    
    commonSymbols = set(IMData[0].SymbolDict.keys())
    
    for IM in IMData[1:]:
        commonSymbols = commonSymbols | set(IM.SymbolDict.keys())
    symbolList = sorted([s for s in commonSymbols])
    symbolDict = {}
    i = 0;
    for s in symbolList:
        symbolDict[s] = i
        i += 1
    return (symbolList, symbolDict)
#end of getAllSymbols

def calSymbProbability(symb, IMs):
    probability = {}
    for IM in IMs:
        for (k,v) in IM.SymbolDict.items():
            if probability.has_key(k):
                probability[k] += v
            else:
                probability[k] = v
    
    N = float(sum(probability.values()))
    for k in probability.keys():
        probability[k] = float(probability[k]) / N
    
    arr =[]
    for k in symb:
        if probability.has_key(k):
            arr.append(probability[k])
        else:
            arr.append(0.0)
    return np.array(arr)   
# end of calSymbProbability

def calDkl(P, Qarr):
    sumDkl = 0.0
    for Q in Qarr:
        t = np.where(Q == 0)
        P1 = P.copy()
        P1[t] = 0.0
        Q1 = Q.copy()
        Q1[t] = 1.0
        Dkl = sum(np.log(P/Q1)*P1)
        sumDkl += Dkl
    return sumDkl/len(Qarr)

# end of calDkl

def trainDataSplit(IM, symbList):
    
    N = len(IM)
    
    symbP = calSymbProbability(symbList, IM)
#     random.seed(0)
    randomIdx = random.choice(N, N, None)
    random012 = randomIdx % Nfolds
    
    IMs = np.array(IM)
    symbQarr = []
    for i in range(Nfolds):
        t = np.where(random012 == i)
        symbQ = calSymbProbability(symbList, IMs[t])
        symbQarr.append(symbQ)
    Dkl_before = calDkl(symbP, symbQarr)
    
    swapTimes = 0
    while swapTimes < MaxSwap:
        a = 0
        b = 0
        while (a % Nfolds) == (b % Nfolds):
            randomPair = random.choice(N, 2, None)
            a = randomPair[0]
            b = randomPair[1]
        aPos = np.where(randomIdx == a)
        bPos = np.where(randomIdx == b)
        randomIdxNew = randomIdx.copy()
        randomIdxNew[aPos] = b
        randomIdxNew[bPos] = a
        random012 = randomIdxNew % Nfolds
        symbQarr = []
        for i in range(Nfolds):
            t = np.where(random012 == i)
            symbQ = calSymbProbability(symbList, IMs[t])
            symbQarr.append(symbQ)
        Dkl_after = calDkl(symbP, symbQarr)
        print '%04d before: %0.7f, after: %0.7f' %(swapTimes, Dkl_before, Dkl_after),
        if Dkl_after < Dkl_before:
            randomIdx = randomIdxNew
            Dkl_before = Dkl_after
            print "Swapped"
        else:
            print "Not Swapped"
        swapTimes += 1
    # end of while swapTimes < MaxSwap:
    
    folds = {}
    for i in range(N):
        IM[i].fold = random012[i]
        folds[IM[i].filename] = random012[i]

#     print symbP[:4]
#     for symbQ in symbQarr:
#         print symbQ[:4]
    
    return (folds, symbP, symbQarr)
# end of trainDataSplit

def formFeature(im, symbDict):
    xList = []
    yList = []
    idxList = []
    for (key,symb) in im.symbol.iteritems():
        k ="{}_{}".format(im.filename,key)
        if im.symbolTruth.has_key(key):
            y = symbDict[im.symbolTruth[key]['lab']]
        else:
            y = 0
        x = {}
        fIdx = 1
        fList = []
        for fKey in featureList:
            fList.extend(symb['features'][fKey])
            
        for a in fList:
            x[fIdx] = float(a)
            fIdx += 1
        xList.append(x)
        yList.append(y)
        idxList.append(k)
    return (xList, yList, idxList)
# end of formFeature

def genTrainDataSet(IMs, Imid, symbDict):
    trainningX = []
    trainningY = []
    trainningIdx = []
    
    for im in IMs:
        if im.fold != Imid:
            # trainning
            x,y,f = formFeature(im, symbDict)
            trainningX.extend(x)
            trainningY.extend(y)
            trainningIdx.extend(f)
    return (trainningY, trainningX, trainningIdx)
# end of genTrainDataSet

def genTestDataSet(IMs, symbDict):
    testingX = []
    testingY = []
    testingIdx = []

    for im in IMs:
        # testing
        x,y,f = formFeature(im, symbDict)
        testingX.extend(x)
        testingY.extend(y)
        testingIdx.extend(f)

    return (testingY, testingX, testingIdx)
# end of genTestDataSet


def writeSVMinput(filename, y, x):
    f = file(filename, 'w')
    for i in range(len(y)):
        f.write("{} ".format(y[i]))
        for (k,v) in x[i].iteritems():
            f.write("{}:{} ".format(k, v))
        f.write("\n")
    f.close()
# end of writeSVMinput


def knn(rX, rY, eX, eY, k=1):
    train = np.array(SVM2list(rX))
    test = np.array(SVM2list(eX))
    N = train.shape[0]
    M = test.shape[0]
    knnY = []
    i = 1
    for x in test:
        a = np.outer(np.ones(N), x)
        b = (a - train)**2
        d = np.sum(b,1)
        minIdx = np.argmin(d)
        y = rY[minIdx]
        knnY.append(y)
        if i % 10000 == 0:
            print "{}/{}".format(i,M)
        i += 1
    correct = sum(np.array(knnY) == np.array(eY))
    corr_rate = float(correct) / float(len(eY))
    print "the correct rate = {}\%".format(corr_rate * 100)
    return knnY
# end of knn

def findSymbByValue(symbDict, idx):
    for (k,v) in symbDict.iteritems():
        if v == idx:
            s = k
            break
    return s
# end of findSymbByValue

def saveTxtFolds(filename, folds):
    f = open(filename, "w")
    for k in sorted(folds.keys()):
        f.write("{} : {}\n".format(k, folds[k]))
    f.close()
# end of saveTxtFolds

def savePrior(filename, symbList, symbP, symbQarr):
    f = open(filename, "w")
    f.write(',')
    for symb in symbList:
        f.write('"{}",'.format(symb))
    f.write('\n')
    
    f.write('Total probability,')
    for P in symbP:
        f.write("{},".format(P))
    f.write('\n')
    
    idx = 0
    for symbQ in symbQarr:
        f.write('Fold {} probability,'.format(idx))
        idx += 1
        for Q in symbQ:
            f.write('{},'.format(Q))
        f.write('\n')
    f.close()
# end of savePrior

def applyFolds(IMs, folds):
    for IM in IMs:
        IM.fold = folds[IM.filename]
# end of applyFolds

def applyResults(IMs, Y, eIdx, symbDict):
    idx = {}
    i = 0
    for e in eIdx:
        idx[e] = i
        i += 1

    for im in IMs:
        for (key,symb) in im.symbol.iteritems():
            symb['lab'] = findSymbByValue(symbDict, Y[idx["{}_{}".format(im.filename,key)]])
            for sid in symb['strokes']:
                im.stroke[sid]['lab'] = symb['lab']
            
# end of applyResults

def applySegResults(IMs, Y, eIdx):
    idx = {}
    i = 0
    for e in eIdx:
        idx[e] = i
        i += 1

    for im in IMs:
        symbs = {}
        sidx = 0
        if im.hasPair:
            cur = ['0']
            for p in im.pair:
#             for p in sorted(im.pair, key=lambda k: [int(k['strokes'][0])]):
                p['result'] = int(Y[idx["{}_{}".format(im.filename,p['strokes'][0])]])
                if p['result']:
                    cur.append(p['strokes'][1])
                else:
                    sId = "AUTO_{}".format(sidx)
                    s = {'strokes':cur, 'lab':'', 'features':{}}
                    symbs[sId] = s
                    for strkId in cur:
                        im.stroke[strkId]['id'] = sId
                        im.stroke[strkId]['lab'] = ''
                    sidx += 1
                    cur = [p['strokes'][1]]
            sId = "AUTO_{}".format(sidx)
            s = {'strokes':cur, 'lab':'', 'features':{}}
            symbs[sId] = s
            for strkId in cur:
                im.stroke[strkId]['id'] = sId
                im.stroke[strkId]['lab'] = ''
        else:
            sId = "AUTO_{}".format(sidx)
            s = {'strokes':['0'], 'lab':'', 'features':{}}
            symbs[sId] = s
            im.stroke['0']['id'] = sId
            im.stroke['0']['lab'] = ''
        im.symbol = symbs
# end of applySegResults

def writeLG(filename, im):
    STRK = im.stroke
    SYMB = im.symbol
    
    output = open(filename, 'w')
    output.write("# IUD, {}\n".format(im.UI))
    output.write("# Nodes:\n")
    
    for strkKey in sorted(STRK.keys(),key=lambda k:int(k)):
        if STRK[strkKey].has_key('lab'):
            lab = STRK[strkKey]['lab']
            if '' == lab:
                lab = "_"
            lab = lab.replace(',', 'COMMA')
            output.write("N, {}, {}, 1.0\n".format(strkKey, lab))

    output.write("\n# Edges:\n")
    outlist = []
    usedSymb = {}
    for symbId in SYMB.keys():
        usedSymb[symbId] = 1
    
    for nextSymbol in SYMB.keys():
        if usedSymb.has_key(nextSymbol):
            for stroke in SYMB[nextSymbol]['strokes']:
                for stroke2 in SYMB[nextSymbol]['strokes']:
                    if stroke != stroke2:
                        outlist.append("E, {}, {}, *, 1.000\n".format(stroke, stroke2))
    if im.parsingSecc:
        generateLGEdge(im, outlist, im.tree)
    
    for l in sorted(outlist, key=lambda k:[int(k.split(',')[1]),int(k.split(',')[2])]):
        output.write(l)
    
    output.close()
# end of writeLG

def generateLGEdge(im, txt, node):
    if node.has_key('type'):
        if node.has_key('sub'):
            if len(node['sub']) > 1:
                nameList = [l['name'] for l in node['sub']]
                for (idx0, idx1) in itertools.combinations(range(len(nameList)), 2):
                    name0 = nameList[idx0]
                    name1 = nameList[idx1]
                    segId = '{}:{}'.format(name0, name1)
                    if grmToSRT.has_key(segId):
                        strkList0 = []
                        getSTRK(strkList0, node['sub'][idx0], im, grmToSRT[segId], 'L')
                        strkList1 = []
                        getSTRK(strkList1, node['sub'][idx1], im, grmToSRT[segId], 'R')
                        for (strk1, strk2) in itertools.product(strkList0, strkList1):
                            txt.append("E, {}, {}, {}, 1.0\n".format(strk1, strk2, grmToSRT[segId]))
            for sub in node['sub']:
                generateLGEdge(im, txt, sub)
# end of generateEdge(txt, node)

def getSTRK(strkList, node, im, rel, side):
    if node.has_key('type'):
        if node.has_key('sub'):
            for sub in node['sub']:
                getSTRK(strkList, sub, im, rel, side)    

    else:
        symbId = node['id']
        if im.symbol.has_key(symbId):
            strkIds = im.symbol[symbId]['strokes']
            strkList.extend(strkIds)
# end of getSTRK(strkList, node, im)

def formSegFeature(im):
    xList = []
    yList = []
    idxList = []
    for pair in im.pair:
        k ="{}_{}".format(im.filename,pair['strokes'][0])
        y = pair['truth']
        x = {}
        fIdx = 1
        fList = []
        for fKey in segFeatureList:
            fList.extend(pair[fKey])
            
        for a in fList:
            x[fIdx] = float(a)
            fIdx += 1
        xList.append(x)
        yList.append(y)
        idxList.append(k)
    return (xList, yList, idxList)
# end of formSegFeature

def genSegTrainDataSet(IMs, Imid):
    trainningX = []
    trainningY = []
    trainningIdx = []
    
    for im in IMs:
        if (im.hasPair) & (im.fold != Imid):
            # trainning
            x,y,f = formSegFeature(im)
            trainningX.extend(x)
            trainningY.extend(y)
            trainningIdx.extend(f)
    return (trainningY, trainningX, trainningIdx)
# end of genSegTrainDataSet

def genSegTestDataSet(IMs):
    testingX = []
    testingY = []
    testingIdx = []

    for im in IMs:
        if im.hasPair:
            # testing
            x,y,f = formSegFeature(im)
            testingX.extend(x)
            testingY.extend(y)
            testingIdx.extend(f)

    return (testingY, testingX, testingIdx)
# end of genSegTestDataSet

def formParsingFeature(im):
    xList = []
    yList = []
    idxList = []
    for pair in im.charPair:
        k ="{}_{}_{}".format(im.filename,pair['symbols'][0],pair['symbols'][1])
        y = pair['truth']
        x = {}
        fIdx = 1
            
        for a in pair['features']:
            x[fIdx] = float(a)
            fIdx += 1
        xList.append(x)
        yList.append(y)
        idxList.append(k)
    return (xList, yList, idxList)
# end of formSegFeature

def genParsingTrainDataSet(IMs, Imid):
    trainningX = []
    trainningY = []
    trainningIdx = []
    
    for im in IMs:
        if (im.hasCharPair) & (im.fold != Imid):
            # trainning
            x,y,f = formParsingFeature(im)
            trainningX.extend(x)
            trainningY.extend(y)
            trainningIdx.extend(f)
    return (trainningY, trainningX, trainningIdx)
# end of genSegTrainDataSet


def split(args):
    
    TrainDataPath = args.input
    assert os.path.isdir(TrainDataPath), "The input directory is not exist!"
         
    AllTrainData = readAllTrainData(TrainDataPath)
    symbList, symbDict = getAllSymbols(AllTrainData)
         
    folds, symbP, symbQarr = trainDataSplit(AllTrainData, symbList)
         
    foldsfile = open(args.folds, "w")
    pickle.dump(folds, foldsfile)
    foldsfile.close()
    
    symbfile = open(args.symbols, "w")
    pickle.dump([symbList, symbDict], symbfile)
    symbfile.close()
         
    if args.txtfolds:
        saveTxtFolds(args.txtfolds, folds)
         
    if args.prior:
        savePrior(args.prior, symbList, symbP, symbQarr)
# end of split

def train(args):
    TrainDataPath = args.input
    assert os.path.isdir(TrainDataPath), "The input directory is not exist!"
         
    foldsfile = args.folds
    assert os.path.isfile(foldsfile), "The folds file is not exist!"
 
    symbfile = args.symbols
    assert os.path.isfile(symbfile), "The symbols file is not exist!"
 
    trainset = args.trainset
    assert len(trainset) > 0, "The set of train data is not specified!"
         
    h_foldsfile = open(foldsfile, 'r')
    folds = pickle.load(h_foldsfile)
    h_foldsfile.close()
         
    h_symbfile = open(symbfile, 'r')
    symb = pickle.load(h_symbfile)
    h_symbfile.close()
         
    AllTrainData = readAllTrainData(TrainDataPath)
    symbDict = symb[1]
         
    print "apply the fold on training data ...",
    applyFolds(AllTrainData, folds)
    print "Done!"
         
    for IM in AllTrainData:
        IM.genFeatures()    
         
    for subset in trainset:
        if subset == '01':
            testset = 2
        elif subset == '02':
            testset = 1
        elif subset == '12':
            testset = 0
        else:
            testset = 3
                  
        rY, rX, rIdx = genTrainDataSet(AllTrainData, testset, symbDict)

        writeSVMinput("trianGridSearch_{}".format(subset), rY, rX)
        
        rX_scale = []
        for X in rX:
            rX_scale.append(X.copy())
                
        scale_cof = scaleData(rX_scale)
        
        start = time.clock()
        m = svmutil.svm_train(rY, rX_scale, '-c 32 -g 0.03125 -b 1')
        elapsed = (time.clock() - start)
        
        f = open("svm_traning.log", 'a')
        f.write("knn_model_{}, training time: {}\n".format(subset,elapsed))
        f.close()
        print  "knn_model_{}, training time: {}\n".format(subset,elapsed)
              
        svmutil.svm_save_model("svm_model_{}".format(subset), m)
              
        hf = open("scaling_{}".format(subset), 'w')
        pickle.dump(scale_cof, hf)
        hf.close()
              
        knn_train = [rY, rX]
        hf = open("knn_model_{}".format(subset), 'w')
        pickle.dump(knn_train, hf)
        hf.close()
# end of train



def classify(args):
    inputPath = args.input
    
    classifiertype = args.classifier
    
    symbfile = args.symbols
    assert os.path.isfile(symbfile), "The symbols file is not exist!"
 
    modelfile = args.model
    assert os.path.isfile(modelfile), "The model file is not exist!"
    
    scalingfile = args.scaling
    
    output = args.output
         
    foldsfile = args.folds
     
    testset = args.testset

    h_symbfile = open(symbfile, 'r')
    symb = pickle.load(h_symbfile)
    h_symbfile.close()
    
    symbDict = symb[1]
    
    if os.path.isdir(inputPath):
        AllTrainData = readAllTrainData(inputPath)
        symbDict = symb[1]
         
        if foldsfile:
            assert os.path.isfile(foldsfile), "The folds file is not exist!"
            h_foldsfile = open(foldsfile, 'r')
            folds = pickle.load(h_foldsfile)
            h_foldsfile.close()
            
            print "apply the fold on training data ...",
            applyFolds(AllTrainData, folds)
            print "Done!"
            
            testData = [IM for IM in AllTrainData if IM.fold == int(testset)]
        else:
            testData = AllTrainData
        
        for IM in testData:
            IM.genFeatures()
            
        eY, eX, eIdx = genTestDataSet(testData, symbDict)
        
    elif os.path.isfile(inputPath):
        im = InkML.InkML(inputPath)
        im.genFeatures()
        eX,eY,eIdx = formFeature(im, symbDict)
        testData = [im]
    else:
        raise NameError, 'Unknown input.'
    
    

    if classifiertype == "1nn":
        h_modelfile = open(modelfile, 'r')
        m = pickle.load(h_modelfile)
        h_modelfile.close()
        rY = m[0]
        rX = m[1]
 
        if scalingfile:
            assert os.path.isfile(scalingfile), "The scaling parameter file is not exist!"
            h_scalingfile = open(scalingfile, 'r')
            scaling_cof = pickle.load(h_scalingfile)
            h_scalingfile.close()
           
            scaleData(eX, scaling_cof)
            scaleData(rX, scaling_cof)

        
        start = time.clock()
        aY = knn(rX, rY, eX, eY, 1)
        elapsed = (time.clock() - start)

    elif classifiertype == "svm":
        
        assert os.path.isfile(scalingfile), "The scaling parameter file is not exist!"
        h_scalingfile = open(scalingfile, 'r')
        scaling_cof = pickle.load(h_scalingfile)
        h_scalingfile.close()
       
        scaleData(eX, scaling_cof)

        m = svmutil.svm_load_model(modelfile)

        start = time.clock()
        aY, p_acc, p_val = svmutil.svm_predict(eY, eX, m, '-b 1')
        print np.array(p_val).max(1)
        elapsed = (time.clock() - start)
    
    print "Classification time: {}".format(elapsed)

    applyResults(testData, aY, eIdx, symbDict)
    
    if os.path.isdir(inputPath):
        for im in testData:
            filename = os.path.basename(im.filename)
            outfile = filename.lower().replace('.inkml', '.lg')
            outpath = os.path.join(output,outfile)
            
            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
            writeLG(outpath, im)
    elif os.path.isfile(inputPath):
        writeLG(output, testData[0])
    
# end of classify

def fp(im):
    im.formPair()
    im.loadPairTruth()
    return im
# end of fp 

def fptest(im):
    im.formPair()
    return im
# end of fptest

def gf(im):
    im.genFeatures()
    return im
# end of gf

def fcp(im):
    im.formCharPair()
    return im
# end of fcp

def segtrain(args):
    TrainDataPath = args.input
    assert os.path.isdir(TrainDataPath), "The input directory is not exist!"
         
    foldsfile = args.folds
    assert os.path.isfile(foldsfile), "The folds file is not exist!"
 
    trainset = args.trainset
    assert len(trainset) > 0, "The set of train data is not specified!"
    
    nProcesses = args.processes
         
    h_foldsfile = open(foldsfile, 'r')
    folds = pickle.load(h_foldsfile)
    h_foldsfile.close()
         
    AllTrainData = readAllTrainData(TrainDataPath)
         
    print "apply the fold on training data ...",
    applyFolds(AllTrainData, folds)
    print "Done!"
    
    start = time.time()
    p = multiprocessing.Pool(processes=nProcesses)        
    AllTrainData = p.map(fp, AllTrainData)
    p.close()
    p.join()

#     for IM in AllTrainData:
#         IM.formPair()
#         IM.loadPairTruth()    
    
    print "run time = {}".format(time.time() - start)
    for subset in trainset:
        if subset == '01':
            testset = 2
        elif subset == '02':
            testset = 1
        elif subset == '12':
            testset = 0
        else:
            testset = 3
    
        rY, rX, rIdx = genSegTrainDataSet(AllTrainData, testset)
        
#         hf = open("rY_{}".format(subset), 'w')
#         pickle.dump(rY, hf)
#         hf.close()
#         hf = open("rX_{}".format(subset), 'w')
#         pickle.dump(rX, hf)
#         hf.close()
        
        fList = np.array(SVM2list(rX))
        PCAer = PCA(fList)
        Wt = PCAer.Wt
        mu = PCAer.mu
        sigma = PCAer.sigma
        
        hf = open("PCA_{}.dump".format(subset), 'w')
        pickle.dump(Wt, hf)
        pickle.dump(mu, hf)
        pickle.dump(sigma, hf)
        hf.close()
        
        pcList = PCAer.Y
        pcList = pcList[:,:nSegPCA]
        rX = list2SVM(pcList)
        
#         writeSVMinput("input_{}".format(subset), rY, rX)
        
        rX_scale = []
        for X in rX:
            rX_scale.append(X.copy())
                 
        scale_cof = scaleData(rX_scale)
         
        start = time.clock()
        m = svmutil.svm_train(rY, rX_scale, '-c 32 -g 0.5 -b 1')
        elapsed = (time.clock() - start)
         
        f = open("seg_traning.log", 'a')
        f.write("seg_model_{}, training time: {}\n".format(subset,elapsed))
        f.close()
        print  "seg_model_{}, training time: {}\n".format(subset,elapsed)
               
        svmutil.svm_save_model("seg_model_{}".format(subset), m)
               
        hf = open("seg_scaling_{}".format(subset), 'w')
        pickle.dump(scale_cof, hf)
        hf.close()
               
# end of segtrain

def segment(args):
    inputPath = args.input
    
    trainset = args.trainset
    
    symbfile = args.symbols
    assert os.path.isfile(symbfile), "The symbols file is not exist!"
 
    modelfile = args.model.format(trainset)
    assert os.path.isfile(modelfile), "The model file is not exist!"
    
    scalingfile = args.scaling.format(trainset)
    assert os.path.isfile(scalingfile), "The scaling parameter file is not exist!"
    
    segmodelfile = args.segmodel.format(trainset)
    assert os.path.isfile(segmodelfile), "The segmentation model file is not exist!"
    
    segscalingfile = args.segscaling.format(trainset)
    assert os.path.isfile(segscalingfile), "The segment scaling parameter file is not exist!"
    
    pcafile = args.pca.format(trainset)
    assert os.path.isfile(pcafile), "The PCA parameter file is not exist!"
    
    output = args.output
         
    foldsfile = args.folds
     
    testset = args.testset

    nProcesses = args.processes


    hf = open(symbfile, 'r')
    symb = pickle.load(hf)
    hf.close()
    
    symbDict = symb[1]
    
    hf = open(scalingfile, 'r')
    scaling_cof = pickle.load(hf)
    hf.close()
    
    hf = open(segscalingfile, 'r')
    segscaling_cof = pickle.load(hf)
    hf.close()
    
    hf = open(pcafile, 'r')
    Wt = pickle.load(hf)
    mu = pickle.load(hf)
    sigma = pickle.load(hf)
    hf.close()
    
    classify_m = svmutil.svm_load_model(modelfile)
    segment_m = svmutil.svm_load_model(segmodelfile)
    
    if os.path.isdir(inputPath):
        AllTrainData = readAllTrainData(inputPath)
        symbDict = symb[1]
         
        if foldsfile:
            assert os.path.isfile(foldsfile), "The folds file is not exist!"
            h_foldsfile = open(foldsfile, 'r')
            folds = pickle.load(h_foldsfile)
            h_foldsfile.close()
            
            print "apply the fold on training data ...",
            applyFolds(AllTrainData, folds)
            print "Done!"
            
            testData = [IM for IM in AllTrainData if IM.fold == int(testset)]
        else:
            testData = AllTrainData

        start = time.time()
        p = multiprocessing.Pool(processes=nProcesses)        
        testData = p.map(fp, testData)
        p.close()
        p.join()

#         for IM in testData:
#             IM.formPair()
#             IM.loadPairTruth()    
    
        print "run time = {}".format(time.time() - start)
        
            
        segY, segX, segIdx = genSegTestDataSet(testData)
        
    elif os.path.isfile(inputPath):
        im = InkML.InkML(inputPath)
        im.formPair()
        im.loadPairTruth()
        segX,segY,segIdx = formSegFeature(im)
        testData = [im]
    else:
        raise NameError, 'Unknown input.'
    
    fList = np.array(SVM2list(segX))

    pcList = project(Wt, mu, sigma, fList)
    pcList = pcList[:,:nSegPCA]
    segX = list2SVM(pcList)
         
    scaleData(segX, segscaling_cof)

    start = time.clock()
    segaY, p_acc, p_val = svmutil.svm_predict(segY, segX, segment_m, '-b 1')
    elapsed = (time.clock() - start)
    
#     for prob in np.array(p_val).max(1):
#         print prob
    probs = np.array(p_val).max(1)
    print "# of probs <= 0.55 is {}".format(np.sum(probs<=0.55))
    print "# of probs <= 0.6 is {}".format(np.sum(probs<=0.6))
    print "# of probs <= 0.65 is {}".format(np.sum(probs<=0.65))
    print "# of probs <= 0.7 is {}".format(np.sum(probs<=0.7))
    print "# of probs <= 0.75 is {}".format(np.sum(probs<=0.75))
    print "# of probs <= 0.8 is {}".format(np.sum(probs<=0.8))
    print "# of probs <= 0.85 is {}".format(np.sum(probs<=0.85))
    print "# of probs <= 0.9 is {}".format(np.sum(probs<=0.9))
    print "# of probs <= 0.95 is {}".format(np.sum(probs<=0.95))
    
    
    print "Segmentation time: {}".format(elapsed)

    applySegResults(testData, segaY, segIdx)


    p = multiprocessing.Pool(processes=nProcesses)        
    testData = p.map(gf, testData)
    p.close()
    p.join()

    eY, eX, eIdx = genTestDataSet(testData, symbDict)
    
    scaleData(eX, scaling_cof)

    start = time.clock()
    aY, p_acc, p_val = svmutil.svm_predict(eY, eX, classify_m, '-b 1')
    elapsed = (time.clock() - start)
    p_val = np.array(p_val)
    probs = np.array(p_val).max(1)
    print "# of probs <= 0.55 is {}".format(np.sum(probs<=0.55))
    print "# of probs <= 0.6 is {}".format(np.sum(probs<=0.6))
    print "# of probs <= 0.65 is {}".format(np.sum(probs<=0.65))
    print "# of probs <= 0.7 is {}".format(np.sum(probs<=0.7))
    print "# of probs <= 0.75 is {}".format(np.sum(probs<=0.75))
    print "# of probs <= 0.8 is {}".format(np.sum(probs<=0.8))
    print "# of probs <= 0.85 is {}".format(np.sum(probs<=0.85))
    print "# of probs <= 0.9 is {}".format(np.sum(probs<=0.9))
    print "# of probs <= 0.95 is {}".format(np.sum(probs<=0.95))    
    print "Classification time: {}".format(elapsed)
    
    unc_idx = np.argsort(probs)
    unc_idx = unc_idx[0]
    print eIdx[unc_idx]
    thisp = p_val[unc_idx,:]
    thissortidx = np.argsort(thisp)
    
    print findSymbByValue(symbDict, classify_m.label[thissortidx[-1]]),
    print thisp[thissortidx[-1]]
    print findSymbByValue(symbDict, classify_m.label[thissortidx[-2]]),
    print thisp[thissortidx[-2]]
    print 

    applyResults(testData, aY, eIdx, symbDict)

     
    if os.path.isdir(inputPath):
        for im in testData:
            filename = os.path.basename(im.filename)
            filename, _ = os.path.splitext(filename)
            outfile = filename + '.lg'
            outpath = os.path.join(output,outfile)
             
            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
            writeLG(outpath, im)
    elif os.path.isfile(inputPath):
        writeLG(output, testData[0])
     
# end of segment


def parsingtrain(args):
    TrainDataPath = args.input
    assert os.path.isdir(TrainDataPath), "The input directory is not exist!"
         
    foldsfile = args.folds
    assert os.path.isfile(foldsfile), "The folds file is not exist!"
 
    trainset = args.trainset
    assert len(trainset) > 0, "The set of train data is not specified!"
    
    nProcesses = args.processes
         
    h_foldsfile = open(foldsfile, 'r')
    folds = pickle.load(h_foldsfile)
    h_foldsfile.close()
         
    AllTrainData = readAllTrainData(TrainDataPath)
         
    print "apply the fold on training data ...",
    applyFolds(AllTrainData, folds)
    print "Done!"
    
    start = time.time()
    
    p = multiprocessing.Pool(processes=nProcesses)        
    AllTrainData = p.map(fcp, AllTrainData)
    p.close()
    p.join()

#     for IM in AllTrainData:
#         IM.formCharPair() 
    
    print "run time = {}".format(time.time() - start)
    for subset in trainset:
        if subset == '01':
            testset = 2
        elif subset == '02':
            testset = 1
        elif subset == '12':
            testset = 0
        else:
            testset = 3
    
        rY, rX, rIdx = genParsingTrainDataSet(AllTrainData, testset)
        
#         hf = open("rY_{}".format(subset), 'w')
#         pickle.dump(rY, hf)
#         hf.close()
#         hf = open("rX_{}".format(subset), 'w')
#         pickle.dump(rX, hf)
#         hf.close()
        
        fList = np.array(SVM2list(rX))
        PCAer = PCA(fList)
        Wt = PCAer.Wt
        mu = PCAer.mu
        sigma = PCAer.sigma
        
        hf = open("PCA_parsing_{}.dump".format(subset), 'w')
        pickle.dump(Wt, hf)
        pickle.dump(mu, hf)
        pickle.dump(sigma, hf)
        hf.close()
        
        pcList = PCAer.Y
        pcList = pcList[:,:nSegPCA]
        rX = list2SVM(pcList)
        
        writeSVMinput("input_parsing_{}".format(subset), rY, rX)
        
        rX_scale = []
        for X in rX:
            rX_scale.append(X.copy())
                 
        scale_cof = scaleData(rX_scale)
         
        start = time.clock()
        m = svmutil.svm_train(rY, rX_scale, '-c 32 -g 0.03125 -b 1')
        elapsed = (time.clock() - start)
         
        f = open("parsing_traning.log", 'a')
        f.write("parsing_model_{}, training time: {}\n".format(subset,elapsed))
        f.close()
        print  "parsing_model_{}, training time: {}\n".format(subset,elapsed)
               
        svmutil.svm_save_model("parsing_model_{}".format(subset), m)
               
        hf = open("parsing_scaling_{}".format(subset), 'w')
        pickle.dump(scale_cof, hf)
        hf.close()
               
# end of parsingtrain

def ps(im, parsingArg):
    print "parsing file: {}".format(im.filename),
    im.generateSymbList(parsingArg)
    print 'Done'
    
#     filename = os.path.basename(im.filename)
#     filename = "{}.pickle".format(filename)
#     h = open(filename, 'w')
#     pickle.dump(im.symbList, h)
#     pickle.dump(im.symbol, h)
#     pickle.dump(im.stroke, h)
#     h.close()
    print 'symbol list to mathml: {}'.format(im.filename),
    im.symbList2XML()
    print 'Done'
    return im
# end of ps(im, )

def ps_star(i):
    return ps(*i)
# end of def ps_star(i):

def parsing(args):
    inputPath = args.input
    
    trainset = args.trainset
    
    symbfile = args.symbols
    assert os.path.isfile(symbfile), "The symbols file is not exist!"
 
    modelfile = args.model.format(trainset)
    assert os.path.isfile(modelfile), "The model file is not exist!"
    
    scalingfile = args.scaling.format(trainset)
    assert os.path.isfile(scalingfile), "The scaling parameter file is not exist!"
    
    segmodelfile = args.segmodel.format(trainset)
    assert os.path.isfile(segmodelfile), "The segmentation model file is not exist!"
    
    segscalingfile = args.segscaling.format(trainset)
    assert os.path.isfile(segscalingfile), "The segment scaling parameter file is not exist!"
    
    pcafile = args.pca.format(trainset)
    assert os.path.isfile(pcafile), "The PCA parameter file is not exist!"
    
    parsingmodelfile = args.parsingmodel.format(trainset)
    assert os.path.isfile(parsingmodelfile), "The parsing model file is not exist!"
    
    parsingscalingfile = args.parsingscaling.format(trainset)
    assert os.path.isfile(parsingscalingfile), "The paring scaling parameter file is not exist!"
    
    parsingpcafile = args.parsingpca.format(trainset)
    assert os.path.isfile(parsingpcafile), "The parsing PCA parameter file is not exist!"
    
    output = args.output
         
    foldsfile = args.folds
     
    testset = args.testset

    nProcesses = args.processes


    hf = open(symbfile, 'r')
    symb = pickle.load(hf)
    hf.close()
    
    symbDict = symb[1]
    
    hf = open(scalingfile, 'r')
    scaling_cof = pickle.load(hf)
    hf.close()
    
    hf = open(segscalingfile, 'r')
    segscaling_cof = pickle.load(hf)
    hf.close()
    
    hf = open(parsingscalingfile, 'r')
    parsingscaling_cof = pickle.load(hf)
    hf.close()
    
    
    hf = open(pcafile, 'r')
    Wt = pickle.load(hf)
    mu = pickle.load(hf)
    sigma = pickle.load(hf)
    hf.close()
    
    hf = open(parsingpcafile, 'r')
    parsingWt = pickle.load(hf)
    parsingmu = pickle.load(hf)
    parsingsigma = pickle.load(hf)
    hf.close()
    
    classify_m = svmutil.svm_load_model(modelfile)
    segment_m = svmutil.svm_load_model(segmodelfile)
#     parsing_m = svmutil.svm_load_model(parsingmodelfile)
    
    parsingArg = {'scaling':   parsingscaling_cof,
                   'Wt':        parsingWt,
                   'mu':        parsingmu,
                   'sigma':     parsingsigma,
                   'mFile':     parsingmodelfile}
    
    if os.path.isdir(inputPath):
        AllTrainData = readAllTrainData(inputPath)
        symbDict = symb[1]
         
        if foldsfile:
            assert os.path.isfile(foldsfile), "The folds file is not exist!"
            h_foldsfile = open(foldsfile, 'r')
            folds = pickle.load(h_foldsfile)
            h_foldsfile.close()
            
            print "apply the fold on training data ...",
            applyFolds(AllTrainData, folds)
            print "Done!"
            
            testData = [IM for IM in AllTrainData if IM.fold == int(testset)]
        else:
            testData = AllTrainData

        start = time.time()
        p = multiprocessing.Pool(processes=nProcesses)        
        testData = p.map(fptest, testData)
        p.close()
        p.join()

#         for IM in testData:
#             IM.formPair()
#             IM.loadPairTruth()    
    
        print "run time = {}".format(time.time() - start)
        
            
        segY, segX, segIdx = genSegTestDataSet(testData)
        
    elif os.path.isfile(inputPath):
        im = InkML.InkML(inputPath)
        im.formPair()
        im.loadPairTruth()
        segX,segY,segIdx = formSegFeature(im)
        testData = [im]
    else:
        raise NameError, 'Unknown input.'
    
    fList = np.array(SVM2list(segX))

    pcList = project(Wt, mu, sigma, fList)
    pcList = pcList[:,:nSegPCA]
    segX = list2SVM(pcList)
         
    scaleData(segX, segscaling_cof)

    start = time.clock()
    segaY, p_acc, p_val = svmutil.svm_predict(segY, segX, segment_m, '-b 1')
    elapsed = (time.clock() - start)
    
    print "Segmentation time: {}".format(elapsed)

    applySegResults(testData, segaY, segIdx)


    p = multiprocessing.Pool(processes=nProcesses)        
    testData = p.map(gf, testData)
    p.close()
    p.join()

    eY, eX, eIdx = genTestDataSet(testData, symbDict)
    
    scaleData(eX, scaling_cof)

    start = time.clock()
    aY, p_acc, p_val = svmutil.svm_predict(eY, eX, classify_m, '-b 1')
    elapsed = (time.clock() - start)
    print "Classification time: {}".format(elapsed)
    
#     p_val = np.array(p_val)
#     probs = np.array(p_val).max(1)
#     print "# of probs <= 0.55 is {}".format(np.sum(probs<=0.55))
#     print "# of probs <= 0.6 is {}".format(np.sum(probs<=0.6))
#     print "# of probs <= 0.65 is {}".format(np.sum(probs<=0.65))
#     print "# of probs <= 0.7 is {}".format(np.sum(probs<=0.7))
#     print "# of probs <= 0.75 is {}".format(np.sum(probs<=0.75))
#     print "# of probs <= 0.8 is {}".format(np.sum(probs<=0.8))
#     print "# of probs <= 0.85 is {}".format(np.sum(probs<=0.85))
#     print "# of probs <= 0.9 is {}".format(np.sum(probs<=0.9))
#     print "# of probs <= 0.95 is {}".format(np.sum(probs<=0.95))    
#     unc_idx = np.argsort(probs)
#     unc_idx = unc_idx[0]
#     print eIdx[unc_idx]
#     thisp = p_val[unc_idx,:]
#     thissortidx = np.argsort(thisp)
#     
#     print findSymbByValue(symbDict, classify_m.label[thissortidx[-1]]),
#     print thisp[thissortidx[-1]]
#     print findSymbByValue(symbDict, classify_m.label[thissortidx[-2]]),
#     print thisp[thissortidx[-2]]
#     print 

    applyResults(testData, aY, eIdx, symbDict)
    
    
    start = time.clock()
    
    if len(testData) > nProcesses:

        p = multiprocessing.Pool(processes=nProcesses)
        mapArg = itertools.izip(testData, itertools.repeat(parsingArg))
        testData = p.map(ps_star, mapArg)
        p.close()
        p.join()
    else:
        for i in range(len(testData)):
            testData[i] = ps(testData[i], parsingArg)
    
#     for im in testData:
#         print "The Latex expression of {} is:".format(im.filename)
#         im.generateSymbList(parsingArg)
#         sec,mat,tree = cykpaser.parse(im.symbList)
#         im.parsingSecc = sec
#         for s in im.symbList:
#             print s['lab'],
#         print
# 
#         if sec:
#             print "There is(are) {} possible tree(s)".format(len(tree))
#             im.tree = tree[0]
# #             CYK.printTree(tree[0])
#             im.seccCYK = True
#         else:
#             print "parsing is failure!"    
                  
    elapsed = (time.clock() - start)
    
    
    print "Parsing time: {}".format(elapsed)
    if os.path.isdir(inputPath):
        for im in testData:
            filename = os.path.basename(im.filename)
            filename, _ = os.path.splitext(filename)
            outfile = filename + '.lg'
            outpath = os.path.join(output,outfile)
             
            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
            h = open(outpath, 'w')
            im.printLG(h)
            h.close()
    elif os.path.isfile(inputPath):
        h = open(output, 'w')
        testData[0].printLG(h)
        h.close()
     
# end of parsing


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Classifier for CSCI-737 Pattern Recognition Project #3.")
    subparsers = parser.add_subparsers(help = "Please specify the command")
    
    parser_split = subparsers.add_parser('split', help="split the training data into three folds")
    parser_split.add_argument("-i", "--input", required=True, help="specify the directory which contains the training data")
    parser_split.add_argument("-f", "--folds", default="folds.dump", help="specify the file which stores the splitting information")
    parser_split.add_argument("-b", "--symbols", default="symb.dump", help="specify the file which store the symbols")
    parser_split.add_argument("-t", "--txtfolds", help="specify a text file which saves the folds information")
    parser_split.add_argument("-p", "--prior", help="specify a csv file which saves the prior probability of symbols")
    parser_split.set_defaults(func=split)

    parser_train = subparsers.add_parser('train', help="train the classifier from the training data")
    parser_train.add_argument("-i", "--input", required=True, help="specify the directory which contains the training data")
    parser_train.add_argument("-f", "--folds", default="folds.dump", help="specify the file which stores the splitting information")
    parser_train.add_argument("-b", "--symbols", default="symb.dump", help="specify the file which store the symbols")
    parser_train.add_argument("-s", "--trainset", choices=['01','02','12','all'], action="append", help="specify the set of training data")
    parser_train.set_defaults(func=train)  
      
    parser_classify = subparsers.add_parser('classify', help="classify the test data by the specified classifier and parameter")
    parser_classify.add_argument("-i", "--input", required=True, help="specify a test file or a directory which contains the test files")
    parser_classify.add_argument("-f", "--folds", help="specify the file which stores the splitting information")
    parser_classify.add_argument("-b", "--symbols", default="symb.dump", help="specify the file which store the symbols")
    parser_classify.add_argument("-e", "--testset", choices=['0','1','2','all'], help="specify the set of testing data")
    parser_classify.add_argument("-c", "--classifier", default="svm", choices=['1nn','svm'], help="specify the type of classifier")
    parser_classify.add_argument("-m", "--model", required=True, help="specify the model file")
    parser_classify.add_argument("-a", "--scaling", help="specify the file which saves the scaling parameters")
    parser_classify.add_argument("-o", "--output", required=True, help="specify the output filename or directory (depends on the input)")
    parser_classify.set_defaults(func=classify)
    
    parser_segtrain = subparsers.add_parser('segtrain', help="train the segmentation classifier from the training data")
    parser_segtrain.add_argument("-i", "--input", required=True, help="specify the directory which contains the training data")
    parser_segtrain.add_argument("-f", "--folds", default="folds.dump", help="specify the file which stores the splitting information")
    parser_segtrain.add_argument("-s", "--trainset", choices=['01','02','12','all'], action="append", help="specify the set of training data")
    parser_segtrain.add_argument("-p", "--processes", default=6, type=int, choices=range(1, 25), help="specify the number of processes when extracting the features of stroke pairs")
    parser_segtrain.set_defaults(func=segtrain)  
 
    parser_segment = subparsers.add_parser('segment', help="segment and classify the test data by the specified parameter")
    parser_segment.add_argument("-i", "--input", required=True, help="specify a test file or a directory which contains the test files")
    parser_segment.add_argument("-f", "--folds", help="specify the file which stores the splitting information")
    parser_segment.add_argument("-b", "--symbols", default="symb.dump", help="specify the file which store the symbols")
    parser_segment.add_argument("-e", "--testset", choices=['0','1','2','all'], help="specify the set of testing data")
    parser_segment.add_argument("-s", "--trainset", choices=['01','02','12','all'], default='all', help="specify the set of training data")
    parser_segment.add_argument("-m", "--model", default="svm_model_{}", help="specify the model file for classification")
    parser_segment.add_argument("-g", "--segmodel", default="seg_model_{}", help="specify the model file for segmentation")
    parser_segment.add_argument("-a", "--scaling", default="scaling_{}",  help="specify the file which saves the scaling parameters")
    parser_segment.add_argument("-l", "--segscaling", default="seg_scaling_{}",  help="specify the file which saves the segmentation scaling parameters")
    parser_segment.add_argument("-c", "--pca", default="PCA_{}.dump", help="specify the file which saves the PCA paramenters")
    parser_segment.add_argument("-p", "--processes", default=6, type=int, choices=range(1, 25), help="specify the number of processes when extracting the features of stroke pairs")
    parser_segment.add_argument("-o", "--output", required=True, help="specify the output filename or directory (depends on the input)")
    parser_segment.set_defaults(func=segment)
       
    parser_parsingtrain = subparsers.add_parser('parsingtrain', help="train the parsing classifier from the training data")
    parser_parsingtrain.add_argument("-i", "--input", required=True, help="specify the directory which contains the training data")
    parser_parsingtrain.add_argument("-f", "--folds", default="folds.dump", help="specify the file which stores the splitting information")
    parser_parsingtrain.add_argument("-s", "--trainset", choices=['01','02','12','all'], action="append", help="specify the set of training data")
    parser_parsingtrain.add_argument("-p", "--processes", default=6, type=int, choices=range(1, 25), help="specify the number of processes when extracting the features of stroke pairs")
    parser_parsingtrain.set_defaults(func=parsingtrain)
    
    parser_parsing = subparsers.add_parser('parsing', help="segment, classify and parse the test data by the specified parameter")
    parser_parsing.add_argument("-i", "--input", required=True, help="specify a test file or a directory which contains the test files")
    parser_parsing.add_argument("-f", "--folds", help="specify the file which stores the splitting information")
    parser_parsing.add_argument("-b", "--symbols", default="symb.dump", help="specify the file which store the symbols")
    parser_parsing.add_argument("-e", "--testset", choices=['0','1','2','all'], help="specify the set of testing data")
    parser_parsing.add_argument("-s", "--trainset", choices=['01','02','12','all'], default='all', help="specify the set of training data")
    parser_parsing.add_argument("-m", "--model", default="svm_model_{}", help="specify the model file for classification")
    parser_parsing.add_argument("-g", "--segmodel", default="seg_model_{}", help="specify the model file for segmentation")
    parser_parsing.add_argument("-d", "--parsingmodel", default="parsing_model_{}", help="specify the model file for parsing")
    parser_parsing.add_argument("-a", "--scaling", default="scaling_{}",  help="specify the file which saves the scaling parameters")
    parser_parsing.add_argument("-l", "--segscaling", default="seg_scaling_{}",  help="specify the file which saves the segmentation scaling parameters")
    parser_parsing.add_argument("-n", "--parsingscaling", default="parsing_scaling_{}",  help="specify the file which saves the parsing scaling parameters")
    parser_parsing.add_argument("-c", "--pca", default="PCA_{}.dump", help="specify the file which saves the PCA paramenters")
    parser_parsing.add_argument("-r", "--parsingpca", default="PCA_parsing_{}.dump", help="specify the file which saves the PCA paramenters")
    parser_parsing.add_argument("-p", "--processes", default=6, type=int, choices=range(1, 25), help="specify the number of processes when extracting the features of stroke pairs")
    parser_parsing.add_argument("-o", "--output", required=True, help="specify the output filename or directory (depends on the input)")
    parser_parsing.set_defaults(func=parsing)
       
    args = parser.parse_args()
    args.func(args)
    

