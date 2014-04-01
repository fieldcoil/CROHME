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
#from xml.dom import minidom
import sys
import re
import collections
import numpy as np


# Number of points in a symbol after resampling
Nsampledpoints = 30

# Number of points in a stroke after resampling
Nstrokesampledpoints = 30;

# define edges creation depending of the mathML tag : {"tag" => liste of edges with label },  index -1 means that there are no corresponding child but the current node should be used (eq msqrt and mfrac)
tagToSRT = {
    "mrow" : [[0,1,'R']],
    "msup" : [[0,1,'Sup']],
    "msub" : [[0,1,'Sub']],
    "mfrac" : [[-1,0,'A'],[-1,1,'B']],
    "msqrt" : [[-1,0,'I'],[-1,1,'I'],[0,1,'R']],
    "mroot" : [[-1,0,'I'],[-1,1,'A']],
    "munder" : [[0,1,'B']],
    "munderover" :  [[0,1,'B'], [0,2,'A']],
    "msubsup" :  [[0,1,'Sub'], [0,2,'Sup']]
    }
tagMainSymb = {
    "mrow" : 1,
    "msup" : 0,
    "msub" : 0,
    "mfrac" : -1,
    "msqrt" : -1,
    "mroot" : -1,
    "munder" : 0,
    "munderover" : 0,
    "msubsup" : 0,
    "mo" : -1,
    "mi" : -1,
    "mn" : -1
    }

nameCharPair = [              
               'R'          ,
               'beginSUP'   ,
               'beginSUB'   ,
               'endSUP'     ,
               'endSUB'     ,
               'SUP2SUB'    ,
               'SUB2SUP'    ]

tagCharPair = {}
for i in range(len(nameCharPair)):
    tagCharPair[nameCharPair[i]] = i

class InkML(object):
    '''
    classdocs
    '''


    def __init__(self, filename):
        '''
        Constructor
        '''

        print "Loading file: {} ...".format(filename),
        self.filename = filename
        
        self.UI = ""
         
        self.stroke = {}
        self.strokeTruth = {}
         
        self.symbol = {}
        self.symbolTruth = {}
         
        self.XML_GT = []
        self.hasPair = False
        self.pair = [];
        self.hasCharPair = False
        self.charPair = []
        self.fold = 0;
        self.tree = None
        self.symbList = None
        self.parsingSecc = False
            
        namespace = "{http://www.w3.org/2003/InkML}"
        # it looks that register_namespace doesn't work
        #ET.register_namespace('ns', 'http://www.w3.org/2003/InkML')
        try:
            tree = ET.parse(filename)
        except ET.ParseError:
            print "An error was occurred while parsing file \"{}\". \nIt might be caused by an unrecognizable charactar in the file. Please search \"\\cdot\" in the file.".format(filename)
            sys.exit()
        except:
            print "Unknown error was occurred while parsing file \"{}\"!".format(filename)
            sys.exit()
                
        root = tree.getroot()
 
        for ann in root.iter('{0}annotation'.format(namespace)):
            if ann.attrib['type'] == "UI":
                self.UI = ann.text
                 
        annXML = root.find('{0}annotationXML'.format(namespace))
        Load_xml_truth(self.XML_GT, annXML[0])
        norm_mathMLNorm(self.XML_GT);
        
        for trace in root.iter('{0}trace'.format(namespace)):
            strk = {}
            strk['trace'] = trace.text.strip()
            self.strokeTruth[trace.attrib['id']] = strk
         
        #symbol ID, to distinguish different symbols with same label, if symbol without any annotationXML
        symbID = 0;
        group = root.find('{0}traceGroup'.format(namespace))
        for subgroup in group.findall('{0}traceGroup'.format(namespace)):
             
            lab = subgroup.find('{0}annotation'.format(namespace)).text
            annXML_G = subgroup.find('{0}annotationXML'.format(namespace))
            sid = ''
            if (annXML_G is not None) and annXML_G.attrib.has_key('href'):
                sid = annXML_G.attrib['href']
            if sid == '':
                sid = "AUTO_{}".format(symbID)
             
            strokeList = []
             
            for traceView in subgroup.findall('{0}traceView'.format(namespace)):
                self.strokeTruth[traceView.attrib['traceDataRef']]['id'] = sid
                self.strokeTruth[traceView.attrib['traceDataRef']]['lab'] = lab
                strokeList.append(traceView.attrib['traceDataRef'])
             
            if len(strokeList) > 0:
                symb = {'lab': lab, 'strokes': strokeList}
                self.symbolTruth[sid] = symb
                symbID += 1
                
        self.NBSYMB = symbID
        
        self.genSymbolDict()
        self.init_symbol()
        self.init_stroke()
        # self.genFeatures()
        print " Done!"
    # end of def __init__(self, filename):
    
    def printLG(self, output = sys.stdout, printTruth = False):
        if printTruth:
            SYMB = self.symbolTruth
            STRK = self.strokeTruth
        else:
            SYMB = self.symbol
            STRK = self.stroke
            
        output.write("# IUD, {}\n".format(self.UI))
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
        
        SegSRT = {}
        getSegSRT(self.XML_GT, SegSRT)
        usedSymb = {}
        for segId in sorted(SegSRT.keys()):
            regex = re.compile(r'\[(.*)\],\[(.*)\]')
            match = regex.search(segId)
            if None != match:
                id1 = match.groups()[0]
                id2 = match.groups()[1]
                assert "" != id1 and "" != id2, " !! Skipping empty segment relationship for: {}; see {}\n".format(segId, self.UI)
                if SYMB.has_key(id1):
                    usedSymb[id1] = 1
                    for strId1 in sorted(SYMB[id1]['strokes'], key=lambda k:int(k)):
                        if SYMB.has_key(id2):
                            usedSymb[id2] = 1
                            for strId2 in sorted(SYMB[id2]['strokes'], key=lambda k:int(k)):
                                outlist.append("E, {}, {}, {}, 1.0\n".format(strId1, strId2, SegSRT[segId]))
            
        # RZ: Adding segmentation edges.
        # FILTER symbols not included in the SRT.
        for nextSymbol in SYMB.keys():
            if usedSymb.has_key(nextSymbol):
                for stroke in SYMB[nextSymbol]['strokes']:
                    for stroke2 in SYMB[nextSymbol]['strokes']:
                        if stroke != stroke2:
                            outlist.append("E, {}, {}, *, 1.000\n".format(stroke, stroke2))
        
        for l in sorted(outlist, key=lambda k:[int(k.split(',')[1]),int(k.split(',')[2])]):
            output.write(l)
        
    # end of def printLG(self, output = sys.stdout, type = 'Truth'):
   
    
    def printTruthLG(self, output = sys.stdout):
        return self.printLG(output, True)
    # end of def printTruthLG(self, output = STDOUT):
    
    
    def genSymbols(self):
        keys = self.symbolTruth.keys()
        symbols = [self.symbolTruth[k]['lab'] for k in keys]
        return symbols
    # end of get Symbols
    
    def genSymbolDict(self):
        symbolDict = collections.Counter(self.genSymbols())
        self.SymbolDict = dict(symbolDict)
    # end of getSymbolDict
        
    def init_symbol(self):
        for (sK,sT) in self.symbolTruth.iteritems():
            s = {}
            s['strokes'] = sT['strokes'][:]
            s['lab'] = ''
            s['features'] = {}
            self.symbol[sK] = s
    # end of initSYMB
    
    def init_stroke(self):
        for (k, sT) in self.strokeTruth.iteritems():
            s = {}
            s['id'] = ''
            traceIN = sT['trace'].split(',')
            traceOUT = []
            for point in traceIN:
                traceOUT.append(map(float,point.strip().split(' ')))
            trace = np.array(traceOUT)            
            s['trace'] = trace
            
            for (sid, symb) in self.symbol.iteritems():
                if k in symb['strokes']:
                    s['id'] = sid
                    s['lab'] = symb['lab']
                    break
            self.stroke[k] = s

    # end of init_stroke

    def genFeatures(self):
        
        print "Extract features from file: {} ...".format(self.filename),
        # preprocessing
        # 1. Duplicate point filtering
        for stroke in self.stroke.itervalues():
            stroke['trace'] = removeDuplicatePoint(stroke['trace'])
            
           
        # 2. Size normalization
        for symbol in self.symbol.itervalues():
            trace = normalizeY(symbol, self.stroke)
            symbol['trace'] = trace
        
        # 3. Smoothing
        for symbol in self.symbol.itervalues():
            lines = symbol['trace']
            for line in lines:
                line = smooth(line)
        
        # 4. Resampling
        for symbol in self.symbol.itervalues():
            lines = symbol['trace']
            (relines, NpStroke) = resample(lines)
            symbol['trace'] = np.vstack(relines)
            symbol['NpStroke'] = NpStroke
        
        # get original height of expression and width of each symbol
#         exp_orig_max_X = np.max(self.symbol.values()[0]['trace'][0][:,0])
#         exp_orig_min_X = np.min(self.symbol.values()[0]['trace'][0][:,0])
        exp_orig_max_Y = np.max(self.stroke[self.symbol.values()[0]['strokes'][0]]['trace'][:,1])
        exp_orig_min_Y = np.min(self.stroke[self.symbol.values()[0]['strokes'][0]]['trace'][:,1])
        for symbol in self.symbol.itervalues():
            stroke_ids = symbol['strokes']
            symb_orig_max_X = np.max(self.stroke[stroke_ids[0]]['trace'][:,0])
            symb_orig_min_X = np.min(self.stroke[stroke_ids[0]]['trace'][:,0])
            symb_orig_max_Y = np.max(self.stroke[stroke_ids[0]]['trace'][:,1])
            symb_orig_min_Y = np.min(self.stroke[stroke_ids[0]]['trace'][:,1])
             
            for stroke_id in stroke_ids[1:]:
                stroke_orig_max_X = np.max(self.stroke[stroke_id]['trace'][:,0])
                stroke_orig_min_X = np.min(self.stroke[stroke_id]['trace'][:,0])
                stroke_orig_max_Y = np.max(self.stroke[stroke_id]['trace'][:,1])
                stroke_orig_min_Y = np.min(self.stroke[stroke_id]['trace'][:,1])
                if stroke_orig_max_X > symb_orig_max_X:
                    symb_orig_max_X = stroke_orig_max_X
                if stroke_orig_min_X < symb_orig_min_X:
                    symb_orig_min_X = stroke_orig_min_X
                if stroke_orig_max_Y > symb_orig_max_Y:
                    symb_orig_max_Y = stroke_orig_max_Y
                if stroke_orig_min_Y < symb_orig_min_Y:
                    symb_orig_min_Y = stroke_orig_min_Y
                     
            if symb_orig_max_Y > exp_orig_max_Y:
                exp_orig_max_Y = symb_orig_max_Y
            if symb_orig_min_Y < exp_orig_min_Y:
                exp_orig_min_Y = symb_orig_min_Y
            symbol['features']['OriginalWidth'] = [symb_orig_max_X - symb_orig_min_X]
        
        # Extracting the features
        for symb in self.symbol.itervalues():
            
            # 1. the number of strokes in a symble
            symb['features']['NofStrokes'] = [len(symb['strokes'])]
        
            # 2. normalized y-position
            symb['features']['normalizedY'] = symb['trace'][:,1].tolist()
            
            # 3-4. vicinity slope and curvature
            pts = symb['trace']
            slope = []
            curvature = []
            for t in range(2,Nsampledpoints-2):
                line = pts[t+2][:2] - pts[t-2][:2]
                line_mag  = np.linalg.norm(line)
                if line_mag == 0:
                    cos_alpha = 0.0
                else:
                    cos_alpha = np.inner(line, np.array([1,0])) / line_mag
                slope.append(cos_alpha)
                
                line1 = pts[t-2][:2] - pts[t][:2]
                line2 = pts[t+2][:2] - pts[t][:2]
                
                line1_mag = np.linalg.norm(line1)
                line2_mag = np.linalg.norm(line2)
                if line1_mag == 0 or line2_mag == 0:
                    cos_beta = 0.0
                else:
                    cos_beta = np.inner(line1, line2)/ (line1_mag * line2_mag)
                
                if cos_beta < -1.0 or cos_beta > 1.0:
                    sin_beta = 0.0
                else:
                    sin_beta = np.sqrt(1 - cos_beta**2)
                curvature.append(sin_beta)
            symb['features']['cos_slope'] = slope
            symb['features']['sin_curvature'] = curvature
            
            # 5. the density in the center of the 3 by 3 matrix
            X = symb['trace'][:,0]
            Y = symb['trace'][:,1]
            
            max_X = np.max(X)
            min_X = np.min(X)
            
            max_Y = np.max(Y)
            min_Y = np.min(Y)
            
            if max_X != min_X:
                X13 = min_X + (max_X - min_X) / 3
                X23 = min_X + (max_X - min_X) * 2 / 3
            else:
                X13 = min_X - .17
                X23 = max_X + .17
                
            if max_Y != min_Y:
                Y13 = min_Y + (max_Y - min_Y) / 3
                Y23 = min_Y + (max_Y - min_Y) * 2 / 3
            else:
                Y13 = min_Y - .17
                Y23 = max_Y + .17
            
            tX = (X >= X13) * (X <= X23)
            tY = (Y >= Y13) * (Y <= Y23)
            symb['features']['density4'] = [np.sum(tX * tY)]
            
            # 6. the normalized width
            symb['features']['normalizedWidth'] = [symb['features']['OriginalWidth'][0]/(exp_orig_max_Y - exp_orig_min_Y)]
            
            # 7. the number of points in a symbol
            nofp = 0
            for stroke_id in symb['strokes']:
                nofp += (self.strokeTruth[stroke_id]['trace'].count(',') + 1)
            symb['features']['NofPoints'] = [nofp]
        
        # clean the obj, change all numpy array to list
        for stroke in self.stroke.itervalues():
            stroke['trace'] = stroke['trace']
        
        for symbol in self.symbol.itervalues():
            symbol['trace'] = symbol['trace']
        
        print "Done!"
    # end of genFeatures
    
    def formPair(self):
        print "Forming pairs from file: {} ...".format(self.filename),
        # preprocessing
        # 1. Duplicate point filtering
        for stroke in self.stroke.itervalues():
            stroke['trace1'] = removeDuplicatePoint(stroke['trace'])
        
        # 2. Smoothing
        for stroke in self.stroke.itervalues():
            smooth(stroke['trace1'])
 
        # 3. Size normalization
        normalizeExp(self.stroke)
        
        # 4. Resampling
        for stroke in self.stroke.itervalues():
            stroke['trace1'] = resampleStroke(stroke['trace1'])

        strkIdList = sorted(self.stroke.keys(), key=int)
        if len(strkIdList) > 1:
            for i, strkId in enumerate(strkIdList[:-1]):
                pair = {}
                pair['strokes'] = [strkId, strkIdList[i+1]]
                pair['G'] = self.extractG(pair['strokes'])
                pair['SPSCF'] = self.extractSCF(pair['strokes'],'SP')
                pair['LNSCF'] = self.extractSCF(pair['strokes'], 'LN')
                pair['GSCF'] = self.extractSCF(pair['strokes'], "G")
                pair['C'] = None
                pair['truth'] = 0
                self.pair.append(pair)
                
            l_symb,l_strk = self.preExtractC()
            self.extractC(l_symb, l_strk)
            self.hasPair = True
         
        print "Done!"
    # end of formPair(self):
    
    def loadPairTruth(self):
        for symb in self.symbolTruth.itervalues():
            if len(symb['strokes']) > 1:
                sList = sorted([int(d) for d in symb['strokes']])
                for i, sId in enumerate(sList[:-1]):
                    if sList[i+1] == sId+1:
                        self.pair[sId]['truth'] = 1
                    else:
                        print "the strokes in a symbol are not successive: {}".format(self.filename)

    # end of loadPairTruth(self):
    
    def extractG(self, pairIds):
        features = []

        line1 = self.stroke[pairIds[0]]['trace1']
        line2 = self.stroke[pairIds[1]]['trace1']

        # 1. features related to the bounding box
        x1_min, y1_min, x1_max, y1_max, x1_cen, y1_cen = boundingBox(line1)
        x2_min, y2_min, x2_max, y2_max, x2_cen, y2_cen = boundingBox(line2)
        
        x_overlapping = min(x1_max, x2_max) - max(x1_min, x2_min)
        y_overlapping = min(y1_max, y2_max) - max(y1_min, y2_min)
        
        if (x_overlapping > 0) & (y_overlapping > 0):
            overlapping_area = x_overlapping * y_overlapping
        else:
            overlapping_area = 0
        
        x_cen_offset = x2_cen - x1_cen
        y_cen_offset = y2_cen - y1_cen
        dist_cen = np.sqrt(x_cen_offset**2 + y_cen_offset**2)
        
        size1 = max(x1_max-x1_min, y1_max-y1_min)
        size2 = max(x2_max-x2_min, y2_max-y2_min)
        size_diff = size1 - size2
        
        features.append(x_overlapping)
        features.append(y_overlapping)
        features.append(overlapping_area)
        features.append(x_cen_offset)
        features.append(y_cen_offset)
        features.append(dist_cen)
        features.append(size_diff)
        
        
        # 2. features related to the average center
        
        x1_avg = np.mean(line1[:,0])
        y1_avg = np.mean(line1[:,1])
        x2_avg = np.mean(line2[:,0])
        y2_avg = np.mean(line2[:,1])
        
        x_avg_offset = x2_avg - x1_avg
        y_avg_offset = y2_avg - y1_avg
        dist_avg = np.sqrt(x_avg_offset**2 + y_avg_offset**2)
        
        features.append(x_avg_offset)
        features.append(y_avg_offset)
        features.append(dist_avg)
        
        
        # 3. features related to the points
        pts_dist = (line1[:,0]-line2[:,0])**2+(line1[:,1]-line2[:,1])**2
        min_pts_dist = np.min(pts_dist)
        max_pts_dist = np.max(pts_dist)
        
        N1 = line1.shape[0]
        N2 = line2.shape[0]
        for i1 in range(0,N1-1):
            for i2 in range(i1+1,N2):
                pts_dist = (line1[i1,0]-line2[i2,0])**2+(line1[i1,1]-line2[i2,1])**2
                min_pts_dist = min(min_pts_dist, pts_dist)
                max_pts_dist = max(max_pts_dist, pts_dist)
        min_pts_dist = np.sqrt(min_pts_dist)
        max_pts_dist = np.sqrt(max_pts_dist)
        
        esline = line2[0] - line1[-1]
        esline_mag  = np.linalg.norm(esline)
        if esline_mag == 0:
            writing_slope = 0.0
        else:
            writing_slope = np.inner(esline, np.array([1,0])) / esline_mag

        seline1 = line1[-1] - line1[0]
        seline2 = line2[-1] - line2[0]
        seline1_mag = np.linalg.norm(seline1)
        seline2_mag = np.linalg.norm(seline2)
        if seline1_mag == 0 or seline2_mag == 0:
            cos_beta = 0.0
        else:
            cos_beta = np.inner(seline1, seline2)/ (seline1_mag * seline2_mag)
                
        if cos_beta < -1.0 or cos_beta > 1.0:
            curvature = 0.0
        else:
            curvature = np.sqrt(1 - cos_beta**2)
        
        x_es_offset = esline[0]
        y_es_offset = esline[1]
        
        x_start_offset = line2[0,0] - line1[0,0]
        y_start_offset = line2[0,1] - line1[0,1]
        start_dist = np.sqrt(x_start_offset**2 + y_start_offset**2)
        
        x_end_offset = line2[-1,0] - line1[-1,0]
        y_end_offset = line2[-1,1] - line1[-1,1]
        end_dist = np.sqrt(x_end_offset**2 + y_end_offset**2)
        
        features.append(min_pts_dist)
        features.append(max_pts_dist)
        features.append(writing_slope)
        features.append(curvature)
        features.append(x_es_offset)
        features.append(y_es_offset)
        features.append(x_start_offset)
        features.append(y_start_offset)
        features.append(start_dist)
        features.append(x_end_offset)
        features.append(y_end_offset)
        features.append(end_dist)
        # 4. other features
        
        return features
    # end of extractG(self, pairIds):
    
    def extractSCF(self, pairIds, ft = 'SP'):
        features = []
        if ft == 'SP':
            sIds = pairIds[:]
        elif ft =='LN':
            sIds = pairIds[:]
            
            pre_id = str(int(pairIds[0])-1)
            if self.stroke.has_key(pre_id):
                sIds.append(pre_id)
                
            next_id = str(int(pairIds[1])+1)
            if self.stroke.has_key(next_id):
                sIds.append(next_id)
        elif ft == 'G':
            sIds = self.stroke.keys()
        
        pts = self.stroke[sIds[0]]['trace1']
        for sId in sIds[1:]:
            pts = np.concatenate((pts, self.stroke[sId]['trace1']), axis = 0)
            
        _, _, _, _, x1_cen, y1_cen = boundingBox(self.stroke[pairIds[0]]['trace1'])
        
        x = pts[:,0] - x1_cen
        y = pts[:,1] - y1_cen
        
        mag = np.sqrt(x**2+y**2)
        ang = np.arctan2(y, x)
        
        mag_max = np.max(mag)
        
        mag_grid = np.array([0., 1./16., 1./8., 1./4., 1./2., 1.]) *  mag_max
        ang_grid = np.array(range(-6, 7)) * np.pi / 6.
        
        mag_split = []
        for mag_idx in range(1, mag_grid.shape[0]):
            mag_lower = mag_grid[mag_idx - 1]
            mag_upper = mag_grid[mag_idx]
            t = ((mag_lower < mag) & (mag <= mag_upper))
            mag_split.append(t)
        
        ang_split = []
        for ang_idx in range(1, ang_grid.shape[0]):
            ang_lower = ang_grid[ang_idx - 1]
            ang_upper = ang_grid[ang_idx]
            t = ((ang_lower < ang) & (ang <= ang_upper))
            ang_split.append(t)
        
        for a in ang_split:
            for m in mag_split:
                f = sum(a * m)
                features.append(f)
        
        return features
    # end of extractSCF(self, pairIds, ft = 'SP'):

    def preExtractC(self):
        import copy
        
        # generate the local stroke dictionary to fit the exist code
        l_strk = copy.deepcopy(self.stroke)
        
        # generate the local symbol dictionary to fit the exist code
        l_symb = {}
        for p in self.pair:
            s1 = {}
            s1['strokes'] = p['strokes'][:1]
            s1['lab'] = ''
            s1['features'] = {}
            l_symb["{}a".format(p['strokes'][0])] = s1
            
            s2 = {}
            s2['strokes'] = p['strokes'][:]
            s2['lab'] = ''
            s2['features'] = {}
            l_symb["{}b".format(p['strokes'][0])] = s2
        
        # preprocessing
        # 1. Duplicate point filtering
        for stroke in l_strk.itervalues():
            stroke['trace'] = removeDuplicatePoint(stroke['trace'])
            
           
        # 2. Size normalization
        for symbol in l_symb.itervalues():
            trace = normalizeY(symbol, l_strk)
            symbol['trace'] = trace
        
        # 3. Smoothing
        for symbol in l_symb.itervalues():
            lines = symbol['trace']
            for line in lines:
                line = smooth(line)
        
        # 4. Resampling
        for symbol in l_symb.itervalues():
            lines = symbol['trace']
            (relines, NpStroke) = resample(lines)
            symbol['trace'] = np.vstack(relines)
            symbol['NpStroke'] = NpStroke
        
        # get original height of expression and width of each symbol
        exp_orig_max_Y = np.max(l_strk[l_symb.values()[0]['strokes'][0]]['trace'][:,1])
        exp_orig_min_Y = np.min(l_strk[l_symb.values()[0]['strokes'][0]]['trace'][:,1])
        for strk in l_strk.itervalues():
            stroke_orig_max_Y = np.max(strk['trace'][:,1])
            stroke_orig_min_Y = np.min(strk['trace'][:,1])
            exp_orig_max_Y = np.max([exp_orig_max_Y, stroke_orig_max_Y])
            exp_orig_min_Y = np.min([exp_orig_min_Y, stroke_orig_min_Y])
           
        for symbol in l_symb.itervalues():
            stroke_ids = symbol['strokes']
            symb_orig_max_X = np.max(l_strk[stroke_ids[0]]['trace'][:,0])
            symb_orig_min_X = np.min(l_strk[stroke_ids[0]]['trace'][:,0])
            symb_orig_max_Y = np.max(l_strk[stroke_ids[0]]['trace'][:,1])
            symb_orig_min_Y = np.min(l_strk[stroke_ids[0]]['trace'][:,1])
             
            for stroke_id in stroke_ids[1:]:
                stroke_orig_max_X = np.max(l_strk[stroke_id]['trace'][:,0])
                stroke_orig_min_X = np.min(l_strk[stroke_id]['trace'][:,0])
                stroke_orig_max_Y = np.max(l_strk[stroke_id]['trace'][:,1])
                stroke_orig_min_Y = np.min(l_strk[stroke_id]['trace'][:,1])
                if stroke_orig_max_X > symb_orig_max_X:
                    symb_orig_max_X = stroke_orig_max_X
                if stroke_orig_min_X < symb_orig_min_X:
                    symb_orig_min_X = stroke_orig_min_X
                if stroke_orig_max_Y > symb_orig_max_Y:
                    symb_orig_max_Y = stroke_orig_max_Y
                if stroke_orig_min_Y < symb_orig_min_Y:
                    symb_orig_min_Y = stroke_orig_min_Y
                     
            symbol['features']['OriginalWidth'] = [symb_orig_max_X - symb_orig_min_X]
        
        # Extracting the features
        for symb in l_symb.itervalues():
            
            # 1. the number of strokes in a symble
            symb['features']['NofStrokes'] = [len(symb['strokes'])]
        
            # 2. normalized y-position
            symb['features']['normalizedY'] = symb['trace'][:,1].tolist()
            
            # 3-4. vicinity slope and curvature
            pts = symb['trace']
            slope = []
            curvature = []
            for t in range(2,Nsampledpoints-2):
                line = pts[t+2][:2] - pts[t-2][:2]
                line_mag  = np.linalg.norm(line)
                if line_mag == 0:
                    cos_alpha = 0.0
                else:
                    cos_alpha = np.inner(line, np.array([1,0])) / line_mag
                slope.append(cos_alpha)
                
                line1 = pts[t-2][:2] - pts[t][:2]
                line2 = pts[t+2][:2] - pts[t][:2]
                
                line1_mag = np.linalg.norm(line1)
                line2_mag = np.linalg.norm(line2)
                if line1_mag == 0 or line2_mag == 0:
                    cos_beta = 0.0
                else:
                    cos_beta = np.inner(line1, line2)/ (line1_mag * line2_mag)
                
                if cos_beta < -1.0 or cos_beta > 1.0:
                    sin_beta = 0.0
                else:
                    sin_beta = np.sqrt(1 - cos_beta**2)
                curvature.append(sin_beta)
            symb['features']['cos_slope'] = slope
            symb['features']['sin_curvature'] = curvature
            
            # 5. the density in the center of the 3 by 3 matrix
            X = symb['trace'][:,0]
            Y = symb['trace'][:,1]
            
            max_X = np.max(X)
            min_X = np.min(X)
            
            max_Y = np.max(Y)
            min_Y = np.min(Y)
            
            if max_X != min_X:
                X13 = min_X + (max_X - min_X) / 3
                X23 = min_X + (max_X - min_X) * 2 / 3
            else:
                X13 = min_X - .17
                X23 = max_X + .17
                
            if max_Y != min_Y:
                Y13 = min_Y + (max_Y - min_Y) / 3
                Y23 = min_Y + (max_Y - min_Y) * 2 / 3
            else:
                Y13 = min_Y - .17
                Y23 = max_Y + .17
            
            tX = (X >= X13) * (X <= X23)
            tY = (Y >= Y13) * (Y <= Y23)
            symb['features']['density4'] = [np.sum(tX * tY)]
            
            # 6. the normalized width
            symb['features']['normalizedWidth'] = [symb['features']['OriginalWidth'][0]/(exp_orig_max_Y - exp_orig_min_Y)]
            
            # 7. the number of points in a symbol
            nofp = 0
            for stroke_id in symb['strokes']:
                nofp += (self.strokeTruth[stroke_id]['trace'].count(',') + 1)
            symb['features']['NofPoints'] = [nofp]
        return (l_symb,l_strk)
    # end of preExtractC    
    def extractC(self, l_symb, l_strk, scalingfile = 'scaling_all', modelfile = 'svm_model_all'):
        
        import os
        import pickle
        import svmutil
        
        featureList = ['NofStrokes', 'normalizedY', 'cos_slope', 'sin_curvature', 'density4', 'normalizedWidth','NofPoints']
        xList = []
        idxList = []
        for (key, symb) in l_symb.iteritems():
            x = {}
            fIdx = 1
            fList = []
            for fKey in featureList:
                fList.extend(symb['features'][fKey])
            
            for a in fList:
                x[fIdx] = float(a)
                fIdx += 1
            xList.append(x)
            idxList.append(key)
        
        assert os.path.isfile(modelfile), "The model file is not exist!"
        assert os.path.isfile(scalingfile), "The scaling parameter file is not exist!"
        h_scalingfile = open(scalingfile, 'r')
        scaling_cof = pickle.load(h_scalingfile)
        h_scalingfile.close()
       
        scaleData(xList, scaling_cof)

        m = svmutil.svm_load_model(modelfile)
        y = len(xList) * [0]
        _, _, p_val = svmutil.svm_predict(y, xList, m, '-b 1')
        
        
        for p in self.pair:
            ida = "{}a".format(p['strokes'][0])
            idb = "{}b".format(p['strokes'][0])
            
            idxa = idxList.index(ida)
            idxb = idxList.index(idb)
            
            p['C'] = p_val[idxa] + p_val[idxb]


    # end of extractC(self, pairIds):
    
    def generateSymbList(self):
        #self.generateSymbList2()
        try:
            self.generateSymbList2()
            if len(self.symbList) < 1:
                self.generateSymbList1()
        except:
            
            print "**** An error was occurred during generating 2D Latex string! 1D Latex string is generated."
            self.generateSymbList1()
    # end of generateSymbList(self):
    
    def generateSymbList1(self):
        self.symbList = []
        for key in sorted(self.symbol.keys(), key=lambda k: int(k.split('_')[1])):
            symb = {'id':key, 'lab':self.symbol[key]['lab']}
            self.symbList.append(symb)
    # end of generateSymbList(self)
    
    def generateSymbList2(self):
        symbList = []
        for key in self.symbol.iterkeys():
            box = np.array(self.symbBox(key))
            width = box[2]-box[0]
            height = box[3]-box[1]
            symb = {'id':key, 'lab':self.symbol[key]['lab'], 
                    'box':box, 'width':width, 'height':height}
            symbList.append(symb)
        
        # Formating symbol list in a reasonable sequence
        symbList = sorted(symbList, key=lambda x: x['box'][4])
        
        # 1. Find the fraction bar, numerator, and denominator
        nSYMB = len(symbList)
        fbIdxs = []
        fnIdxs = []
        fdIdxs = []
        for bIdx in range(nSYMB):
            bar = symbList[bIdx]
            if bar['lab'] == '-':
                nHO = 0
                nIdxs = []
                dIdxs = []
                for nIdx in range(nSYMB): 
                    num = symbList[nIdx]
                    if bIdx!= nIdx:
                        HO = hop(bar['box'], num['box'])
                        if HO > 0.5*num['width']:
                            nHO += 1
                            if num['box'][5] < bar['box'][5]:
                                nIdxs.append(nIdx)
                            else:
                                dIdxs.append(nIdx)
                if nHO > 1:
                    fbIdxs.append(bIdx)
                    fnIdxs.append(nIdxs)
                    fdIdxs.append(dIdxs)
        
        if len(fbIdxs) > 0: # There is at least one fraction in the expression
            
            # we need to solve the multi fraction problem
            for i in range(len(fbIdxs)-1):
                nIdxs1 = fnIdxs[i]
                dIdxs1 = fdIdxs[i]
                bIdx1 = fbIdxs[i]
                for j in range(i+1, len(fbIdxs)):
                    nIdxs2 = fnIdxs[j]
                    dIdxs2 = fdIdxs[j]
                    bIdx2 = fbIdxs[j]
                    removeDuplicatedIdx(nIdxs1, bIdx1, nIdxs2, bIdx2)
                    removeDuplicatedIdx(nIdxs1, bIdx1, dIdxs2, bIdx2)
                    removeDuplicatedIdx(dIdxs1, bIdx1, nIdxs2, bIdx2)
                    removeDuplicatedIdx(dIdxs1, bIdx1, dIdxs2, bIdx2)
            
            removedIdx = []
            nums = []
            dens = []
            bars = []
            for (i, fbIdx) in enumerate(fbIdxs):
                num = []
                den = []
                symbList[fbIdx]['signed'] = True
                bars.append(symbList[fbIdx])
                for fnIdx in fnIdxs[i]:
                    removedIdx.append(fnIdx)
                    num.append(symbList[fnIdx])
                for fdIdx in fdIdxs[i]:
                    removedIdx.append(fdIdx)
                    den.append(symbList[fdIdx])
                nums.append(num)
                dens.append(den)
            base = symbList[:]
            for rIdx in sorted(set(removedIdx), reverse=True):
                del base[rIdx]
            
            if len(base) > 0:
                base = sortSYMB(base)
            
            for i in range(len(fbIdxs)-1, -1, -1):
                if len(nums[i]) > 0:
                    num = sortSYMB(nums[i])
                if len(dens[i]) > 0:
                    den = sortSYMB(dens[i])
                
                found = False
                for j in range(len(base)-1, -1, -1):
                    if ((base[j]['lab'] == '-') and (base[j]['box'][4] == bars[i]['box'][4])):
                        insPos = j
                        found = True
                        break
                if found:
                    assert base[insPos]['lab'] == '-', "Unknow error!"
                    base[insPos]['lab'] = '\\frac'
                    insPos += 1
                    
                    box = den[-1]['box'][:]
                    box[0]=box[2]
                    box[4]=box[2]
                    height = den[-1]['height']
                    closeb = {'id':'NULL', 'lab':'}','box':box,'width':0., 'height':height}
    
                    box = den[0]['box'][:]
                    box[2]=box[0]
                    box[4]=box[0]
                    height = den[0]['height']
                    openb = {'id':'NULL', 'lab':'{','box':box,'width':0., 'height':height}
    
                    base.insert(insPos, closeb)
                    while len(den) > 0:
                        base.insert(insPos, den.pop())
                    base.insert(insPos, openb)
            
                    box = num[-1]['box'][:]
                    box[0]=box[2]
                    box[4]=box[2]
                    height = num[-1]['height']
                    closeb = {'id':'NULL', 'lab':'}','box':box,'width':0., 'height':height}
    
                    box = num[0]['box'][:]
                    box[2]=box[0]
                    box[4]=box[0]
                    height = num[0]['height']
                    openb = {'id':'NULL', 'lab':'{','box':box,'width':0., 'height':height}
    
                    base.insert(insPos, closeb)
                    while len(num) > 0:
                        base.insert(insPos, num.pop())
                    base.insert(insPos, openb)
                
            symbList = base
        else:
            symbList = sortSYMB(symbList)
        
        self.symbList = symbList
    # end of generateSymbList(self)
    
    def symbBox(self,symbId):
        strkIds = self.symbol[symbId]['strokes']
        line = self.stroke[strkIds[0]]['trace1']
        for strkId in strkIds[1:]:
            line = np.vstack((line, self.stroke[strkId]['trace1']))

        x_max = np.max(line[:,0])
        x_min = np.min(line[:,0])
        y_max = np.max(line[:,1])
        y_min = np.min(line[:,1])
    
        x_cen = (x_max + x_min) / 2.0
        y_cen = (y_max + y_min) / 2.0

        return (x_min, y_min, x_max, y_max, x_cen, y_cen)
    # end of symbBox(self,symbId)
    
    def sortSymbId(self):
        '''
        sort symbols according to the stroke ids
        '''
        
        symbs = self.symbolTruth
        symbIdList = []
        
        for symbId in symbs.iterkeys():
            s = symbId, min(symbs[symbId]['strokes'])
            symbIdList.append(s)

        sortedSymbIdList = [i[0] for i in sorted(symbIdList, key=lambda x:int(x[1]))]
        return sortedSymbIdList
    # end of sortSymbId(self):
    
    def formCharPair(self):
        print "Forming character pairs from file: {} ...".format(self.filename),
        # preprocessing
        # 1. Duplicate point filtering
        for stroke in self.stroke.itervalues():
            stroke['trace1'] = removeDuplicatePoint(stroke['trace'])
        
        # 2. Smoothing
        for stroke in self.stroke.itervalues():
            smooth(stroke['trace1'])
 
        # 3. Size normalization
        normalizeExp(self.stroke)

        SegSRT = {}
        getSegSRT(self.XML_GT, SegSRT)
                
        symbIdList = self.sortSymbId()
        if len(symbIdList) > 1:
            for i, symbId in enumerate(symbIdList[:-1]):
                k = '[{}],[{}]'.format(symbId, symbIdList[i+1])
                if SegSRT.has_key(k):
                    pair = {}
                    pair['symbols'] = [symbId, symbIdList[i+1]]
                    pair['truth'] = None
                    if SegSRT[k] == 'R':
                        pair['truth'] = tagCharPair['R']
                    elif SegSRT[k] == 'Sub':
                        pair['truth'] = tagCharPair['beginSUB']
                        pair2 = findEndOfSux(self.XML_GT, 'msub', symbId)
                        if len(pair2) > 0:
                            self.charPair.append(pair2)
                    elif SegSRT[k] == 'Sup':
                        pair['truth'] = tagCharPair['beginSUP']
                        pair2 = findEndOfSux(self.XML_GT, 'msup', symbId)
                        if len(pair2) > 0:
                            self.charPair.append(pair2)
                    else:
                        # we don't handle others
                        pass
                    
                    if pair['truth'] != None:
                        pair['features'] = self.extractCharPairSCF(pair['symbols'])
                        self.charPair.append(pair)
            
        if len(self.charPair) > 0:
            self.hasCharPair = True
        print "Done!"
    # end of formCharPair(self)
    
    
    def extractCharPairSCF(self,symbList):
        assert len(symbList)==2, "The char pair should contains two symbols"
        symb1 = self.symbolTruth[symbList[0]]
        symb2 = self.symbolTruth[symbList[1]]
        
        
        
        features = []
        return features
    # end of extractCharPairSCF(self,symbList):
    
    def plot(self):
        import matplotlib.pyplot as plt
        for key,strk in im1.stroke.iteritems():
            plt.plot(strk['trace1'][:,0], -strk['trace1'][:,1])
            plt.text(strk['trace1'][0,0], -strk['trace1'][0,1],key)
        plt.show()
    # end of plot(self):
# end of class InkML(object):

def Load_xml_truth(truth, data):
    '''
    #############################################
    #### Load xml truth from raw data, fill the current xml truth struct    ####
    #### param 1 :  reference to current xml truth struct (ARRAY)      ####
    #### param 2 :  reference to current xml XML::LibXML::Node         ####
    #############################################
    '''
    current = {}
    regex = re.compile('\{.+?\}')
    current['name'] = regex.sub('', data.tag)
    current['sub'] = []
      
    truth.append(current)
    
    regex = re.compile('^\{.+?\}id$')
    for k in data.attrib.keys():
        if regex.search(k):
            current['id'] = data.attrib[k]
            break
    
    if len(data)>0:
        for subExp in data:
            Load_xml_truth(current['sub'], subExp)
    else:
        current['lab'] = data.text


# end of def Load_xml_truth(truth, data):

def norm_mathMLNorm(current):
    """
    ########################################################
    #### Normalization of xml truth struct to assume  CROHME normalization rules    ####
    #### param 1 :  reference to current xml truth struct (ARRAY)      ####
    #######################################################
    """
    symbTags = {"mi":1, "mo":1, "mn": 1}
    subExpNames = {"msub":1,
                   "msup":1, 
                   "mfrac":2, 
                   "mroot":2, 
                   "msubsup":3,
                   "munderover":3,
                   "munder":2}

    redoFather = False
    redoFromChild = True
    while redoFromChild:
        redoFromChild = False
        for exp in current:
            redo = True
            while redo:
                redo = False
                if 'math' == exp['name']:
                    #start : noting to do
                    #start : check if there is one child
                    nb = len(exp['sub'])
                    if nb > 1: #to much children => merge remove the fisrt one and process others
                        newRow = {} # init new node
                        newRow['name'] = 'mrow'
                        newRow['sub'] = exp['sub']
                        exp['sub'] = []
                        exp['sub'].append(newRow)
                        redo = True
                    # end of if nb > 1:
                elif 'mrow' == exp['name']:
                    # rule 1 : no more than 2 symbols in a mrow
                    nb = len(exp['sub'])
                    if nb > 2: #to much children => merge remove the fisrt one and process others
                        newRow = {} # init new node
                        newRow['name'] = 'mrow'
                        newRow['sub'] = exp['sub'][1:nb]  #remove first
                        exp['sub'] = exp['sub'][:1]
                        exp['sub'].append(newRow)
                        redo = True
                    elif 1 == nb: #not enought children => replace  the current mrow by its child
                        exp['name'] = exp['sub'][0]['name']
                        if exp['sub'][0].has_key('id'):
                            exp['id'] = exp['sub'][0]['id']
                        if exp['sub'][0].has_key('lab'):
                            exp['lab'] = exp['sub'][0]['lab']
                        exp['sub'] = exp['sub'][0]['sub']
                        redo = True
                    elif 0 == nb:
                        pass
                    else:
                        #rule 2 : use right recursive for mrow , so left child should NOT be mrow
                        if 'mrow' == exp['sub'][0]['name']:
                            children = exp['sub'][0]['sub']
                            exp['sub'] = exp['sub'][1:nb] # remove first
                            children.extend(exp['sub'])
                            exp['sub'] = children
                            redo = True
                elif 'msqrt' == exp['name']:
                    # rule 1 : no more than 2 symbols in a mrow
                    nb = len(exp['sub'])
                    if nb > 2: #to much children => merge remove the fisrt one and process others
                        newRow = {} # init new node
                        newRow['name'] = 'mrow'
                        newRow['sub'] = exp['sub'][1:nb]  #remove first
                        exp['sub'] = exp['sub'][:1]
                        exp['sub'].append(newRow)
                        redo = True
                    elif 2 == nb:
                        #rule 2 : use right recursive for mrow , so left child should NOT be mrow
                        if 'mrow' == exp['sub'][0]['name']:
                            children = exp['sub'][0]['sub']
                            exp['sub'] = exp['sub'][1:nb] # remove first
                            children.extend(exp['sub'])
                            exp['sub'] = children
                            redo = True
                    elif 1 == nb: 
                        if 'mrow' == exp['sub'][0]['name']:
                            children = exp['sub'][0]['sub']
                            exp['sub'] = exp['sub'][1:nb] # remove first
                            children.extend(exp['sub'])
                            exp['sub'] = children
                            redo = True
                elif symbTags.has_key(exp['name']) and 1 == symbTags[exp['name']]:
                    # nothing to normalise
                    pass
                elif subExpNames.has_key(exp['name']) and 1 == subExpNames[exp['name']]:
                    # no more than 2 children
                    if len(exp['sub']) > 2:
                        #print($exp->{name}." problem detected : more than 2 children, nb=".@{$exp->{sub}}."\n");
                        pass
                    elif len(exp['sub']) == 2 and exp['sub'][0]['name'] == 'mrow':
                        # if left child is 1 mrow, the mrow should be removed and the relation is put on the last child of the mrow
                        theChildren = exp['sub'][0]['sub']
                        if len(theChildren) > 0: # we can to it
                            # built a new msub/msup relation and put it at the end of the mrow
                            newSR = {}
                            newSR['name'] = exp['name']
                            newSR['sub'] = []
                            if exp.has_key('id'):
                                newSR['id'] = exp['id']
                            newSR['sub'].append(theChildren[-1]) # the base of SR
                            newSR['sub'].append(exp['sub'][1]) # the child
                            exp['name'] = 'mrow'
                            if exp.has_key('id'):
                                del exp['id']
                            exp['sub'] = exp['sub'][0]['sub'] # remove the last element (old base of SR)
                            exp['sub'].pop() # remove the last element (old base of SR)
                            exp['sub'].append(newSR) # and replace by the new one
                            redo = True
                            redoFather = True
                    elif len(exp['sub']) == 1 and exp['sub'][0]['name'] == 'mrow':
                        #print($exp->{name}." problem detected : if only one child it should not be a mrow\n");
                        pass
                    elif len(exp['sub']) == 0:
                        #print($exp->{name}." problem detected : no child !\n");
                        pass
                        
                elif 1 < subExpNames[exp['name']]:
                    # for special relations with multi sub exp, we should have the exact number of children
                    if exp['sub'] > subExpNames[exp['name']]:
                        pass
                        #print($exp->{name}." problem detected : more than ".$subExpNames{$exp->{name}}." children, nb=".@{$exp->{sub}}."\n");
                else:
                    # reject other tags
                    print "unknown tag : {}".format(exp['name'])
            # end of while redo
            if not redoFather:
                #recursivity : process sub exp
                for subExp in exp['sub']:
                    redoFromChild = redoFromChild or norm_mathMLNorm([subExp])
                # end of for subExp in exp['sub']:
            # end of if not redoFather:
        # end of for exp in current:
    # end of while redoFromChild::
    return redoFather
# end of def norm_mathMLNorm(current):

def getSegSRT(current, SRT):
    '''
    ########################################################
    #### build recursively the SRT with segmentation id from MathML        ####
    #### param 1 :  current MathML Graph (ARRAY)             ####
    #### param 2 :  current SRT (HASH)            ####
    #### return list of all children id, the first one being the main stroke holding the spatial relation ####
    #######################################################
    '''
    children = []
    for exp in current:
        # deep first to set children and main symbole in sub exp
        currentChildren = getSegSRT(exp['sub'], SRT)
        #set up children
        children.extend(currentChildren)
        if exp.has_key('id'):
            children.append(exp['id'])
        exp['children'] = currentChildren
        
        #set up the main symb ID
        if tagMainSymb.has_key(exp['name']):
            if -1 == tagMainSymb[exp['name']]:
                exp['mainSymbId'] = exp['id']
            else:
#                 print exp['name']
#                 print tagMainSymb[exp['name']]
                exp['mainSymbId'] = exp['sub'][tagMainSymb[exp['name']]]['mainSymbId']
            assert "" != exp['mainSymbId'], " !! {} {} tag is missing symbol ID {}\n".format(exp['name'], exp['id'], exp['mainSymbId']);
        else:
            assert 'math' == exp['name'], '!! {} not in symbol ID list\n'.format(exp['name'])
            pass
        
        # add the link depending of tag name
        if tagToSRT.has_key(exp['name']):
            for link in tagToSRT[exp['name']]:
                ids1 = []
                if -1 == link[0]:
                    ids1.append(exp['mainSymbId'])
                else:
                    ids1.append(exp['sub'][link[0]]['mainSymbId'])
                    
                if link[1] < len(exp['sub']):
                    for id1 in ids1:
                        if exp['sub'][link[1]].has_key('id'): # link to the symbol if any
                            SRT['[{}],[{}]'.format(id1,exp['sub'][link[1]]['id'])] = link[2]
                        
                        for id2 in exp['sub'][link[1]]['children']:# link to the children
                            SRT['[{}],[{}]'.format(id1,id2)] = link[2]
    return children
# end of getSegSRT(current, STR):


def removeDuplicatePoint(stroke):
    N = stroke.shape[0]
    idx = []
    for i in range(N-1):
        if stroke[i][0] == stroke[i+1][0] and stroke[i][1] == stroke[i+1][1]:
            idx.append(i+1)
    out = np.delete(stroke, idx, 0)
    return out
# end of removeDuplicatePoint

def normalizeY(symbol, stroke):
    trace = []
    local_max_X = [];
    local_min_X = [];
    local_max_Y = [];
    local_min_Y = [];
    for strK in symbol['strokes']:
        line = stroke[strK]['trace']
        local_max_X.append(np.max(line[:,0]))
        local_min_X.append(np.min(line[:,0]))
        local_max_Y.append(np.max(line[:,1]))
        local_min_Y.append(np.min(line[:,1]))
    max_X = max(local_max_X)
    min_X = min(local_min_X)
    max_Y = max(local_max_Y)
    min_Y = min(local_min_Y)
    
    for strK in symbol['strokes']:
        line = stroke[strK]['trace'][:,:2].copy()
        if max_Y != min_Y:
            line[:,1] = line[:,1] - min_Y
            line /= (max_Y - min_Y)
        else:
            line[:,1] = np.ones(min_Y.shape, 'float') * 0.5
            if max_X != min_X:
                line /= (max_X - min_X)

        trace.append(line)
    return trace
# end of normalizeY

def smooth(line):
    N = line.shape[0]
    oldline = line.copy()
    for i in range(1, N-1):
        line[i] = (oldline[i-1] + oldline[i] + oldline[i+1]) / 3
# end of smooth

def resample(lines):
    length = []
    dist_length = []
    Nperline = []
    for line in lines:
        dist = []
        Npoints = line.shape[0]
        for i in range(Npoints - 1):
            dist.append(np.sqrt((line[i][0]-line[i+1][0])**2+(line[i][1]-line[i+1][1])**2))
        length.append(np.sum(dist))
        dist_length.append(dist)
        Nperline.append(Npoints)
    length = np.array(length)
    total_length = np.sum(length)
    Nperline = np.array(Nperline)
    
    if total_length != 0:
    
        Npl = np.int8(np.round(length * Nsampledpoints / total_length))
    
        # if there is only one point in a stroke, then the length of the stroke is zero
        # But we need to keep this point
        for n in np.nditer(Npl, op_flags=['readwrite']):
            if n == 0:
                n += 1
    
        while (np.sum(Npl) < Nsampledpoints):
            dlength = length / Npl
            idx = np.argmax(dlength)
            Npl[idx] += 1
        
        while (np.sum(Npl) > Nsampledpoints):
            dlength = length / Npl
            idx = np.argmin(dlength)
            while Npl[idx] < 2:
                dlength[idx] = np.max(dlength) + 1.0
                idx = np.argmin(dlength)
            Npl[idx] -= 1
        
        newlines = []
        Nlines = len(lines)
        for idx in range(Nlines):
            line = lines[idx]
            dist = dist_length[idx]
            line_length = length[idx]
        
            Nsample = line.shape[0]
            newNsample = int(Npl[idx])
            newline = np.zeros([newNsample,line.shape[1]])
        
            if newNsample == 1:
                newline[0] = line[0]
            else:
        
                newdist = line_length / (newNsample - 1)
        
                newline[0] = line[0]
                newline[-1] = line[-1]
        
                for i in range(1, newNsample-1):
                    curr_length = i * newdist
                    for j in range(0,Nsample-1):
                        if sum(dist[:j]) <= curr_length and sum(dist[:j+1]) > curr_length:
                            break
                    s = curr_length - sum(dist[:j])
                    p = s / dist[j]
                    q = 1 - p
                    newline[i] = line[j]*q + line[j+1]*p
            newlines.append(newline)
    else:
        Npl = np.int8(np.round(Nperline * Nsampledpoints / np.sum(Nperline)))
    
        # if there is only one point in a stroke, then the length of the stroke is zero
        # But we need to keep this point
        for n in np.nditer(Npl, op_flags=['readwrite']):
            if n == 0:
                n += 1
    
        while (np.sum(Npl) < Nsampledpoints):
            dlength = length / Npl
            idx = np.argmax(dlength)
            Npl[idx] += 1
        
        while (np.sum(Npl) > Nsampledpoints):
            dlength = length / Npl
            idx = np.argmin(dlength)
            while Npl[idx] < 2:
                dlength[idx] = np.max(dlength) + 1.0
                idx = np.argmin(dlength)
            Npl[idx] -= 1
        
        newlines = []
        Nlines = len(lines)
        for idx in range(Nlines):
            line = lines[idx]
            newNsample = int(Npl[idx])
            
            newline = np.repeat(line, newNsample, 0)
            newlines.append(newline)
    return (newlines,Npl)
# end of resample

def normalizeExp(strokes):
    local_max_X = [];
    local_min_X = [];
    local_max_Y = [];
    local_min_Y = [];
    for strk in strokes.itervalues():
        line = strk['trace1']
        local_max_X.append(np.max(line[:,0]))
        local_min_X.append(np.min(line[:,0]))
        local_max_Y.append(np.max(line[:,1]))
        local_min_Y.append(np.min(line[:,1]))
    max_X = max(local_max_X)
    min_X = min(local_min_X)
    max_Y = max(local_max_Y)
    min_Y = min(local_min_Y)
    
    for strk in strokes.itervalues():
        line = strk['trace1'][:,:2].copy()
        if max_Y != min_Y:
            line[:,1] = line[:,1] - min_Y
            line /= (max_Y - min_Y)
        else:
            line[:,1] = np.ones(min_Y.shape, 'float') * 0.5
            if max_X != min_X:
                line /= (max_X - min_X)

        strk['trace1'] = line
# end of normalizeExp

def resampleStroke(line):
    
    dist = []
    Nsample = line.shape[0]
    for i in range(Nsample - 1):
        dist.append(np.sqrt((line[i][0]-line[i+1][0])**2+(line[i][1]-line[i+1][1])**2))        
    length = np.sum(dist)
    
    newNsample = Nstrokesampledpoints
    
    if length != 0:
        newline = np.zeros([newNsample,line.shape[1]])
        newdist = length / (newNsample - 1)
        newline[0] = line[0]
        newline[-1] = line[-1]
        
        for i in range(1, newNsample-1):
            curr_length = i * newdist
            for j in range(0,Nsample-1):
                if sum(dist[:j]) <= curr_length and sum(dist[:j+1]) > curr_length:
                    break
            s = curr_length - sum(dist[:j])
            p = s / dist[j]
            q = 1 - p
            newline[i] = line[j]*q + line[j+1]*p

    else:
        newline = np.repeat(line, newNsample, 0)

    return newline
# end of resampleStroke

def boundingBox(line):
    x_max = np.max(line[:,0])
    x_min = np.min(line[:,0])
    y_max = np.max(line[:,1])
    y_min = np.min(line[:,1])
    
    x_cen = (x_max + x_min) / 2.0
    y_cen = (y_max + y_min) / 2.0

    return (x_min, y_min, x_max, y_max, x_cen, y_cen)
# end of boundingBox(line):

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

def hop(box1, box2):
    x1_min = box1[0]
    x1_max = box1[2]
    x2_min = box2[0]
    x2_max = box2[2]

    return min(x1_max, x2_max) - max(x1_min, x2_min)
# end of hop(box1, box2)

def vop(box1, box2):
    y1_min = box1[1]
    y1_max = box1[3]
    y2_min = box2[1]
    y2_max = box2[3]

    return min(y1_max, y2_max) - max(y1_min, y2_min)
# end of vop(box1, box2)

def sortSYMB(symbList):
    # 1. sort the symbols by the bounding box center in the horizontal direction
    symbList = sorted(symbList, key=lambda x: x['box'][4])
    
    # find \sqrt
    nSYMB = len(symbList)
    sqIdxs = []
    spIdxs = []
    sbIdxs = []
    for sIdx in range(nSYMB):
        sqrt  = symbList[sIdx]
        if sqrt['lab'] == '\\sqrt':
            nHO = 0
            pIdxs = []
            bIdxs = []
            for nIdx in range(nSYMB):
                num = symbList[nIdx]
                if sIdx != nIdx:
                    HO = hop(sqrt['box'], num['box'])
                    if HO > 0.4*num['width']:
                        nHO += 1
                        if num['box'][4] < sqrt['box'][0] + num['width']*1.:
                            pIdxs.append(nIdx)
                        else:
                            bIdxs.append(nIdx)
            if nHO > 0:
                sqIdxs.append(sIdx)
                spIdxs.append(pIdxs)
                sbIdxs.append(bIdxs)

    if len(sqIdxs) > 0: # There is at least one \sqrt sign in the expression 
#         print "there are {} \\sqrt".format(len(sqIdxs))
        for i in range(len(sqIdxs)-1, -1, -1):
            sqIdx = sqIdxs[i]
            bIdxs = sbIdxs[i]
            pIdxs = spIdxs[i]
            
            while len(pIdxs) > 1:
                buf = pIdxs.pop()
                bIdxs.insert(0,buf)
            
            if len(bIdxs) < 1:
                if len(pIdxs) > 0:
                    bIdxs.append(pIdxs.pop())
                else:
                    bIdxs.append(sqIdx+1)
                
            if len(pIdxs) == 1:
                pIdx = pIdxs[0]
                
                if sqIdx > pIdx:
                    sqrt = symbList.pop(sqIdx)
                    pIdx_old = pIdx
                    symbList.insert(pIdx_old, sqrt)
                    for (j, bIdx) in enumerate(bIdxs):
                        if bIdx < sqIdx:
                            bIdxs[j] += 1
                    pIdx += 1
                    sqIdxs[i] = pIdx_old
                
                box = symbList[max(bIdxs)]['box'][:]
                box[0]=box[2]
                box[4]=box[2]
                height = symbList[max(bIdxs)]['height']
                closeb = {'id':'NULL', 'lab':'}','box':box,'width':0., 'height':height}
                symbList.insert(max(bIdxs)+1, closeb)
                
                box = symbList[min(bIdxs)]['box'][:]
                box[2]=box[0]
                box[4]=box[0]
                height = symbList[min(bIdxs)]['height']
                openb = {'id':'NULL', 'lab':'{','box':box,'width':0., 'height':height}
                symbList.insert(min(bIdxs), openb)
                
                box = symbList[pIdx]['box'][:]
                box[0]=box[2]
                box[4]=box[2]
                height = symbList[pIdx]['height']
                closeb = {'id':'NULL', 'lab':']','box':box,'width':0., 'height':height}
                symbList.insert(pIdx+1, closeb)
                
                symbList[pIdx]['signed'] = True
                
                box = symbList[pIdx]['box'][:]
                box[2]=box[0]
                box[4]=box[0]
                height = symbList[pIdx]['height']
                openb = {'id':'NULL', 'lab':'[','box':box,'width':0., 'height':height}
                symbList.insert(pIdx, openb)
            else:
                if sqIdx > min(bIdxs):
                    sqrt = symbList.pop(sqIdx)
                    sqrt['signed'] = True
                    bIdxs_min = min(bIdxs)
                    symbList.insert(bIdxs_min, sqrt)
                    for (j, bIdx) in enumerate(bIdxs):
                        if bIdx < sqIdx:
                            bIdxs[j] += 1
                    sqIdxs[i] = bIdxs_min
                
                box = symbList[max(bIdxs)]['box'][:]
                box[0]=box[2]
                box[4]=box[2]
                height = symbList[max(bIdxs)]['height']
                closeb = {'id':'NULL', 'lab':'}','box':box,'width':0., 'height':height}
                symbList.insert(max(bIdxs)+1, closeb)
                
                box = symbList[min(bIdxs)]['box'][:]
                box[2]=box[0]
                box[4]=box[0]
                height = symbList[min(bIdxs)]['height']
                openb = {'id':'NULL', 'lab':'{','box':box,'width':0., 'height':height}
                symbList.insert(min(bIdxs), openb)
    # end of if len(sqIdxs) > 0: 
        
    # find sub and sup:
#     highest = max(symbList, key=lambda s: s['height'])
#     candidate = []
#     for symb in symbList:
#         if symb['lab'] != '\\sqrt':
#             candidate.append(symb)
#     if len(candidate) > 4:
#         avg_cen_y = np.mean([s['box'][5] for s in candidate])
#         for i in range(len(candidate)-1, -1, -1):
#             if abs(candidate[i]['box'][5]-avg_cen_y) > highest['height']*.25:
#                 candidate.pop(i)
#         if len(candidate) > 1:
#             highest = max(candidate, key=lambda s: s['height'])
# 
#     subIdxs=[]
#     supIdxs=[]
#     for (i,symb) in enumerate(symbList):
#         if symb['id'] == 'NULL':
#             continue
#         elif symb.has_key('signed'):
#             if symb['signed']:
#                 continue
#             
#         if symb['height'] < 0.6*highest['height']:
#             if symb['box'][5] < highest['box'][5] - highest['height'] * 0.2:
#                 supIdxs.append(i)
#             elif symb['box'][5] > highest['box'][5] + highest['height'] * 0.2:
#                 subIdxs.append(i)
#         elif symb['height'] < 0.8*highest['height']:     
#             if symb['box'][5] < highest['box'][5] - highest['height'] * 0.3:
#                 supIdxs.append(i)
#             elif symb['box'][5] > highest['box'][5] + highest['height'] * 0.3:
#                 subIdxs.append(i)
#         else:
#             if symb['box'][5] < highest['box'][5] - highest['height'] * 0.4:
#                 supIdxs.append(i)
#             elif symb['box'][5] > highest['box'][5] + highest['height'] * 0.4:
#                 subIdxs.append(i)
#     
#     if len(subIdxs) > 0:
#         if subIdxs[0] == 0:
#             del subIdxs[0]
#             supIdxs.insert(0,1)
#     if len(supIdxs) > 0:
#         if supIdxs[0] == 0:
#             del supIdxs[0]
#             subIdxs.insert(0,1)
#             
#     su = subIdxs[:]
#     su.extend(supIdxs)
#     su=sorted(su)
#     
#     if len(su) > 0:
#         start = [su[0]]
#         end = []
#         mast = []
#         for i in range(len(su)-1):
#             if su[i]+1 == su[i+1]:
#                 continue
#             elif su[i]+2 == su[i+1]:
#                 if ((hop(symbList[su[i]]['box'],symbList[su[i]+1]['box']) > symbList[su[i]]['width']*.5) or
#                     (hop(symbList[su[i+1]]['box'],symbList[su[i]+1]['box']) > symbList[su[i+1]]['width']*.5)):
#                     mast.append(su[i]+1)
#                 else:
#                     end.append(su[i])
#                     start.append(su[i+1])
#             else:
#                 end.append(su[i])
#                 start.append(su[i+1])
#         end.append(su[-1])
#         
#         for i in range(len(start)-1, -1, -1):
#             for idxs in range(start[i],end[i]+1):
#                 if idxs in mast:
#                     symbList.insert(start[i],symbList.pop(idxs))
#                     start[i] += 1
#                     break
#             sup_start = start[i]
#             current = sup_start
#             nSup = 0
#             for idx in range(start[i], end[i]+1):
#                 if symbList[idx]['box'][5] < highest['box'][5]:
#                     symbList.insert(current, symbList.pop(idx))
#                     current += 1
#                     nSup += 1
#             sub_end = end[i]
#             
#             nSub = sub_end - sup_start + 1 - nSup
#             
#             if nSub > 0:
#                 sub_start = sup_start + nSup
#                 box = symbList[sub_end]['box'][:]
#                 box[0]=box[2]
#                 box[4]=box[2]
#                 height = symbList[sub_end]['height']
#                 closeb = {'id':'NULL', 'lab':'}','box':box,'width':0., 'height':height}
#                 symbList.insert(sub_end+1, closeb)
#                 
#                 box = symbList[sub_start]['box'][:]
#                 box[2]=box[0]
#                 box[4]=box[0]
#                 height = symbList[sub_start]['height']
#                 openb = {'id':'NULL', 'lab':'{','box':box,'width':0., 'height':height}
#                 symbList.insert(sub_start, openb)
#                 sub_sign = {'id':'NULL', 'lab':'_','box':box,'width':0., 'height':height}
#                 symbList.insert(sub_start, sub_sign)
#                 
#             if nSup > 0:
#                 sup_end = sup_start + nSup - 1
#                 box = symbList[sup_end]['box'][:]
#                 box[0]=box[2]
#                 box[4]=box[2]
#                 height = symbList[sup_end]['height']
#                 closeb = {'id':'NULL', 'lab':'}','box':box,'width':0., 'height':height}
#                 symbList.insert(sup_end+1, closeb)
#                 
#                 box = symbList[sup_start]['box'][:]
#                 box[2]=box[0]
#                 box[4]=box[0]
#                 height = symbList[sup_start]['height']
#                 openb = {'id':'NULL', 'lab':'{','box':box,'width':0., 'height':height}
#                 symbList.insert(sup_start, openb)
#                 sup_sign = {'id':'NULL', 'lab':'^','box':box,'width':0., 'height':height}
#                 symbList.insert(sup_start, sup_sign)
            
    return symbList
# end of sortSYMB(symbList):

def removeDuplicatedIdx(nIdxs1, bIdx1, nIdxs2, bIdx2):
    for i in range(len(nIdxs1)-1, -1, -1):
        for j in range(len(nIdxs2)-1, -1, -1):
            if nIdxs1[i] == nIdxs2[j]:
                if bIdx1 in nIdxs2:
                    del nIdxs2[j]
                elif bIdx2 in nIdxs1:
                    del nIdxs1[i]
                else:
                    # This situation shouldn't happen
                    pass
# end of removeDuplicatedIdx

def findEndOfSux(current, name, mainSymbId, parent=None):
    pair = {}
    n = len(current)
    for i in range(n):
        exp = current[i]
        if exp.has_key('name') and exp.has_key('mainSymbId'):
            if (exp['name'] == name) and (exp['mainSymbId'] == mainSymbId):
                pair = {}
                if i < (n-1):
                    nextExp = current[i+1]
                    if len(exp['children']) > 0:
                        symb1 = exp['children'][-1];
                    else:
                        symb1 = exp['id']
                        
                    if len(nextExp['children']) > 0:
                        symb2 = nextExp['children'][0];
                    else:
                        symb2 = nextExp['id']
                        
                    pair['symbols'] = [symb1, symb2]
                    pair['features'] = []
                    if name == 'msup':
                        pair['truth'] = tagCharPair['endSUP']
                    elif name == 'msub':
                        pair['truth'] = tagCharPair['endSUB']
                break
            elif (exp['name'] == 'msubsup' ) and ('msub' == name) and (exp['mainSymbId'] == mainSymbId):
                pair = {}
                if len(exp['sub']) == 3:
                    if len(exp['sub'][1]['children']) > 0:
                        symb1 = exp['sub'][1]['children'][-1]
                    else:
                        symb1 = exp['sub'][1]['id']
                    
                    if len(exp['sub'][2]['children']) > 0:
                        symb2 = exp['sub'][2]['children'][0]
                    else:
                        symb2 = exp['sub'][2]['id']
                    
                    pair['symbols'] = [symb1, symb2]
                    pair['features'] = []
                    pair['truth'] = tagCharPair['SUP2SUB']
                break
            elif (exp['name'] == 'msubsup' ) and ('msup' == name) and (exp['mainSymbId'] == mainSymbId):
                pair = {}
                if len(exp['sub']) == 3:
                    if len(exp['sub'][2]['children']) > 0:
                        symb1 = exp['sub'][2]['children'][-1]
                    else:
                        symb1 = exp['sub'][2]['id']
                    
                    if len(exp['sub'][1]['children']) > 0:
                        symb2 = exp['sub'][1]['children'][0]
                    else:
                        symb2 = exp['sub'][1]['id']
                    
                    pair['symbols'] = [symb1, symb2]
                    pair['features'] = []
                    pair['truth'] = tagCharPair['SUB2SUP']
                break
            else:
                pair = findEndOfSux(exp['sub'], name, mainSymbId, current)
        else:
            pair = findEndOfSux(exp['sub'], name, mainSymbId, current)
    return pair
# end of findEndOfSu(SegSRT,a):

def plotXML_GT(current, level):
    for i in range(len(current)):
        
        print '{}: {}'.format(i,current[i])

if __name__ == '__main__':
    im1 = InkML('../TrainINKML_v3/expressmatch/81_daniel.inkml')
    # im1 = InkML('../TrainINKML_v3/MathBrush/2009210-947-105.inkml')
    # im1 = InkML('../TrainINKML_v3/KAIST/traindata2_25_sub_88.inkml')

    im1.formCharPair()
    
    for p in im1.charPair:
        print p
    im1.plot()
    print im1.symbol

    

    

