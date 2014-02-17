#!/usr/bin/env python
'''
###########################################################################
#
# CSCI-737 Pattern Recognition
# Fall, 2013
# Project #3
# 
# Author: Wei Yao (wxy3806_AT_rit.edu) & Fan Wang (fxw6000_AT_rit.edu)
# Date: Dec 11 2013
#
###########################################################################

Part of this program was rewritten of the perl script crohme2lg.pl, which is
a part of CROHMElib (http://www.cs.rit.edu/~dprl/Software.html).

Thanks to the authors: H. Mouchere and R. Zanibbi


This program uses libsvm to perform SVM classifictaion.
http://www.csie.ntu.edu.tw/~cjlin/libsvm/

Thanks to the authors: Chih-Chung Chang and Chih-Jen Lin 
'''

import xml.etree.ElementTree as ET
import sys
import copy
import itertools

class CYK(object):
    '''
    classdocs
    '''


    def __init__(self, filename="OurGrammar.xml"):
        '''
        Constructor
        '''
        
        print "Loading grammar: {} ...".format(filename),
        self.filename = filename
        self.grammar = {}
        
        try:
            tree = ET.parse(filename)
        except ET.ParseError:
            print "An error was occurred while parsing file \"{}\". \nIt might be caused by an unrecognizable charactar in the file. Please search \"\\cdot\" in the file.".format(filename)
            sys.exit()
        except:
            print "Unknown error was occurred while parsing file \"{}\"!".format(filename)
            sys.exit()
            
        root = tree.getroot()
        for rule in root:
            left = rule.attrib['category']
            right = []
            for category in rule:
                nodename = category.attrib['name']
                node = {'name':nodename,'terminal':False}
                if category.attrib.has_key('terminal'):
                    if (category.attrib['terminal'] =='true'):
                        node['terminal'] = True
                right.append(node)
            if self.grammar.has_key(left):
                self.grammar[left].append(right)
            else:
                self.grammar[left] = [right]
        self.root = 'S'
        print "Done!"
    # end of __init__(self, filename):
    
    def toCNF(self):
        CNF = copy.deepcopy(self.grammar)
 
        # 1. Replace productions that produce single nonterminals
        SNT = {}      
        while (findSingleNT(CNF, SNT)):
            replaceSingleNT(CNF, SNT)
            SNT = {}
        delNonRef(CNF, self.root)
        
        # 2.  Isolate terminals
        replaceSingleT(CNF)
        
        # 3. Pair nonterminals in right sides of rules
        pairNT(CNF)
        
        rCNFt, rCNFnt = reverseCNF(CNF)

        self.CNF = CNF
        self.rCNFt = rCNFt
        self.rCNFnt = rCNFnt
        
    # end of toCNF(self):
    
    def printGrammar(self, obj = "grammar", ntonly = False):
        if obj == "CNF":
            grammar = self.CNF
        else:
            grammar = self.grammar
        
        for left in sorted(grammar.keys()):
            out = "{} -> ".format(left)
            rightstr = []
            for category in grammar[left]:
                rightstr.append(' '.join([d['name']+(" @" if d['terminal'] else "") for d in category]))
            out = out + '\n\t| '.join(rightstr)
            print out
            print
    # end of printGrammar(self):
    
    def printCNF(self):
        self.printGrammar("CNF")
    # end of printCNF(self):
    
    def parse(self, x):
        n = len(x)
        
        mat = []
        tree = []
        secc = False
        
        if n < 1:
            return (secc, mat, tree)
        
        for i in range(n):
            line = []
            for _ in range(i+1):
                line.append(None)
            mat.append(line)
        
        for j in range(n):
            cell = []
            left = self.rCNFt[x[j]['lab']]
            for l in left:
                sub = {'name':l, 'child':x[j]}
                cell.append(sub)
            mat[n-1][j] = cell
            
        for i in range(n-2,-1,-1):
            for j in range(i+1):
                cell = []
                for k in range(n-1-i):
                    sub1_i = i+k+1
                    sub1_j = j
                    
                    sub2_i = n-k-1
                    sub2_j = j+n-i-k-1
                    
                    sub1 = mat[sub1_i][sub1_j]
                    sub2 = mat[sub2_i][sub2_j]
                    
                    for (sub1_idx,sub1_c) in enumerate(sub1):
                        for (sub2_idx, sub2_c) in enumerate(sub2):
                            k = "{}_{}".format(sub1_c['name'],sub2_c['name'])
                            if self.rCNFnt.has_key(k):
                                left = self.rCNFnt[k]
                                for l in left:
                                    sub = {'name':l, 'sub1':[sub1_i,sub1_j,sub1_idx], 'sub2':[sub2_i,sub2_j,sub2_idx]}
                                    cell.append(sub)
                mat[i][j] = cell
        # end of for i in range(n-2,-1,-1):
        nTree = 0
        top = mat[0][0]
        roots = []
        for t in top:
            if t['name'] == self.root:
                nTree += 1
                roots.append(t)
                
        if nTree > 0:
            secc = True
            for r in roots:
                if r.has_key('sub1'):
                    sub = [buildTree(r['sub1'], mat), buildTree(r['sub2'], mat)]
                elif r.has_key('child'):
                    sub = [r['child']]
                t = {'type':'root','name':r['name'],'sub':sub}
                reduceTreeLayer(t)
                tree.append(t)

        
        return (secc, mat, tree)
# end of class CYK(object)

def reduceTreeLayer(node):
    idx = findPairinSub(node)
    if node.has_key('type'):
        while idx > -1:
            sub = node['sub'][idx]
            assert sub.has_key('sub'), "Unknown tree struture!"
            del node['sub'][idx]
            while len(sub['sub'])> 0:
                node['sub'].insert(idx,sub['sub'].pop())
            idx = findPairinSub(node)
        for s in node['sub']:
            reduceTreeLayer(s)
# end of reduceTreeLayer(node)

def findPairinSub(node):
    idx = -1
    if node.has_key('type'):
        assert node.has_key('sub'), "Unknown tree structure"
        for (i,s) in enumerate(node['sub']):
            if (s.has_key('type')):
                assert s.has_key('name'), "Unknown tree structure"
                if s['name'].startswith('Pair_'):
                    idx = i
                    break
    return idx
# end of findPairinSub(node)
def buildTree(node, mat):
    i = node[0]
    j = node[1]
    k = node[2]
    r = mat[i][j][k]
    if r.has_key('sub1'):
        sub = [buildTree(r['sub1'], mat), buildTree(r['sub2'], mat)]
    elif r.has_key('child'):
        sub = [r['child']]
    t = {'type':'nt', 'name':r['name'],'sub':sub}
    return t
# end of buildTree(noe, mat):

def findSingleNT(CNF, rule):
    found = False
    for left,right in CNF.iteritems():
        for i in range(len(right)):
            if (len(right[i]) == 1) & (not right[i][0]['terminal']):
                if rule.has_key(left):
                    rule[left].append(i)
                else:
                    rule[left]=[i]
                found = True
    return found
# end of findSingleNT(CNF, rule):

def replaceSingleNT(CNF, rule):
    for (key, idxs) in rule.iteritems():
        for idx in sorted(idxs, reverse=True):
            beRep = CNF[key][idx][0]['name']
            assert CNF.has_key(beRep), "Couldn't find {} on the left hand side!".format(beRep)
            CNF[key].extend(CNF[beRep])
            del CNF[key][idx]
            
# end of replaceSingleNT(CNF, rule):

def delNonRef(CNF, root):
    ref = dict.fromkeys(CNF.keys())
    assert ref.has_key(root), "Coundn't find the root rule {}".format(root)
    ref[root] = True
    
    for right in CNF.itervalues():
        for l in right:
            for t in l:
                if not t['terminal']:
                    ref[t['name']] = True
      
    for k,v in ref.iteritems():
        if not v:
            del CNF[k]
# end of delNonRef(CNF)

def replaceSingleT(CNF):
    n = 0
    nameTempl = "Terminal_{}"
    newRules = {}
    for right in CNF.itervalues():
        for l in right:
            if len(l) > 1:
                for (i,t) in enumerate(l):
                    if t['terminal']:
                        found = False
                        for (newL,newR) in newRules.iteritems():
                            if t['name'] == newR[0][0]['name']:
                                found = True
                                newLeft = newL
                                break

                        if not found:
                            newLeft = nameTempl.format(n)
                            n += 1

                        newRules[newLeft]=[[t]]
                        newt = {'name':newLeft,'terminal':False}
                        l[i] = newt
    CNF.update(newRules)
# end of replaceSingleT(CNF):

def pairNT(CNF, n = 0):
    redo = False
    nameTempl = "Pair_{}"
    newRules = {}

    for right in CNF.itervalues():
        for (i,l) in enumerate(right):
            if len(l) > 2:
                newLeft = nameTempl.format(n)
                n += 1
                newRules[newLeft] = [l[:-1]]
                
                for _ in range(len(l)-1):
                    del right[i][0]
                
                right[i].insert(0,{'name':newLeft, 'terminal':False})
                redo = True

    if redo:
        pairNT(newRules, n)

    CNF.update(newRules)
# end of pairNT(CNF):

def reverseCNF(CNF):
    revT = {}
    revNT = {}
    for (left,right) in CNF.iteritems():
        for l in right:
            if len(l)==2:
                k = "{}_{}".format(l[0]['name'],l[1]['name'])
                if revNT.has_key(k):
                    revNT[k].append(left)
                else:
                    revNT[k] = [left]

            elif len(l)==1:
                k = l[0]['name']
                if revT.has_key(k):
                    revT[k].append(left)
                else:
                    revT[k] = [left]
    return (revT, revNT)
# end of reverseCNF(CNF)

def printMat(mat):
    for line in mat:
        for cell in line:
            print '|',
            for c in cell:
                print c['name'],
                print ',',
        print '|'
# end of printMat(mat)

def printTree(tree):
    txt = []
    n = 0
    genLine(txt, n, tree)
    
    for l in txt:
        print l+'|'
# end of printTree(tree)

def genLine(txt, nLine, node):
    
    if node.has_key('type'):
        if len(txt) > nLine:
            txt[nLine]+= ("|"+node['name'])
        elif len(txt) == nLine:
            txt.append('|'+node['name'])
        
        if node.has_key('sub'):
            for sub in node['sub']:
                genLine(txt, nLine+1, sub)
    else:
        if len(txt) > nLine:
            txt[nLine]+= ('|'+node['lab'])
        elif len(txt) == nLine:
            txt.append('|'+node['lab'])
# end of genLine

def buildRelDict(grm):
    for (left, right) in grm.iteritems():
        for line in right:
            if len(line) > 1:
                nameList = [l['name'] for l in line]
                for (name0, name1) in itertools.combinations(nameList, 2):
                    rel = "R"
                    if (name0.startswith('SUB_') | name0.startswith('SUP_') | (name0 == 'POWER') |
                        (name0 == 'NUMERATOR') | (name0 == 'DENOMINATOR')):
                        continue
                    
                    if name1.startswith('SUP_'):
                        rel = 'Sup'
                    elif name1 =='POWER':
                        rel = 'Sup'
                    elif name1.startswith('SUB_'):
                        rel = 'Sub'
                    elif (name0 =='SQRT') & (name1=='SYMB_OR_BRACED_EXP'):
                        rel = 'I'
                    elif (name0 =='SQRT') & (name1=='SYMBOL'):
                        rel = 'A'
                    elif name1 == 'NUMERATOR':
                        rel = 'A'
                    elif name1 == 'DENOMINATOR':
                        rel = 'B'
                    print "\t'{}:{}':'{}',\t #{} -> {}".format(name0, name1, rel, left, ' '.join(nameList))
# end of buildRelDict
if __name__ == '__main__':
    cyk = CYK()
    cyk.root = 'S'
#     cyk.printGrammar()
    cyk.toCNF()
    
    print
    print
#     cyk.printCNF()
#     print cyk.rCNFt
#     print cyk.rCNFnt
    
    print
    print
    print
    print    
    
    exp = [{'lab':'('},{'lab':'a'},{'lab':'-'},{'lab':'b'},{'lab':')'},{'lab':'/'},{'lab':'c'}]
    #exp = ['a','(','b','/','c',')']
    sec,mat,tree = cyk.parse(exp)
    printMat( mat )
    if sec:
        print len(tree)
        print tree[0]
        printTree(tree[0])
    print
    print
    
    #buildRelDict(cyk.grammar)
