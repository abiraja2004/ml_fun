#!/usr/bin/python -u
"""
pip install numpy, scipy
git and python setup.py installed scikitl-learn
ML refs:
http://scikit-learn.org/stable/modules/naive_bayes.html
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit_transform
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
"""
import sys, re
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib 


def _debug_pause(pauseStr=''):
    try:
        raw_input(pauseStr)
    except:
        e = sys.exc_info()[0]
        exit('\nAborted: %s' % repr(e))

def load_inputs(fname, neg_ftol):
    print '>Refine negative targets with feature tolerance:', neg_ftol
    neg_ftol += 2
    inpX, inpY = [], []
    LIN = open(fname,'r')
    n_pos = 0
    
    for line in LIN:
        if not line.count(')\t'): continue
#         if allpos and line[0] != '*': continue
        flds = line.strip().split('\t')
#         print flds,'>>',
        if flds[0].count('*'):
            inpY.append(1)
            n_pos += 1
        else:
            if len(flds)>neg_ftol: continue # Ignore negative target with features > neg_ftol
            inpY.append(0)
        row = np.zeros(18, dtype=np.int)
        for idx in range(1,len(flds)):
            try:
                row[int(flds[idx][1:])-1] = 1
            except: # ftxx for file type
                row[17] = int(flds[idx][2:])                
        inpX.append(row)
    print '># of loaded Pos=', n_pos ,'Neg=', len(inpY) - n_pos
    return inpX, inpY

def analyse_test(fname, inpX, inpY, model, mthd):
    LIN = open(fname,'r')
    
    # Test with trained models    
    print 'Predicting (%s)...' % mthd
    outY = model.predict(inpX)
    
    TP,TN,FP,FN = 0, 0, 0, 0
    for idx in range(0,len(inpY)):
        obsd, pred = inpY[idx], outY[idx]
        if pred:
            if pred == obsd:
                TP += 1
            else:
                FP += 1
#                     print 'FP#',idx+1
        else:
            if pred == obsd:
                TN += 1
            else:
                FN += 1  
    print 'TP=',TP,'FP=',FP, 'PPV= %.1f%%' % float(TP*100.0/(TP+FP)), 'Sens= %.1f%%' % float(TP*100.0/(TP+FN)),
    print 'TN=',TN,'FN=',FN, 'Spec= %.1f%%' % float(TN*100.0/(TN+FP))    

    line = LIN.readline()
    while not line.count('/User'):
        line = LIN.readline()
        
    isObsd, isPred, idx = 0, 0, 0
    file_obsd, file_pred = [], []
    for line in LIN:
        if line.count(')\t'):
            if line[0] == '*':
                isObsd = 1
                isPred = outY[idx]
            idx += 1
        if line.count('/User'):
            file_obsd.append(isObsd)
            file_pred.append(isPred)
            isObsd, isPred = 0, 0
    file_obsd.append(isObsd)
    file_pred.append(isPred)
    num_predf, num_obsdf = sum(file_pred), sum(file_obsd)
    print 'Correct prediction in file base= %d/%d (%.1f%%)' % (num_predf, num_obsdf, float(num_predf*100.0/num_obsdf))
        
    LIN.close()

def nbc_train(inpX, inpY):
    nbc = BernoulliNB()
    print 'Starting ML training (NaiveBayes)...'
    mdl_nbc = nbc.fit(inpX, inpY)
    return mdl_nbc

def svc_train(inpX, inpY):
    # Fit regression model
    svc_rbf = SVC(kernel='rbf', C=1e3, gamma=0.1)
#     svr_lin = SVR(kernel='linear', C=1e3)
#     svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    print 'Starting ML training (SVM)...'
    mdl_svc = svc_rbf.fit(inpX, inpY)
    return mdl_svc

def svr_train(inpX, inpY):
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#     svr_lin = SVR(kernel='linear', C=1e3)
#     svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    print 'Starting ML training (SVR)...'
    mdl_svm_rbf = svr_rbf.fit(inpX, inpY)
    return mdl_svm_rbf

def rfc_train(inpX, inpY, n_trees=10):       
    # Fit regression model
    rfc = RandomForestClassifier(n_estimators=n_trees)
    print 'Starting ML training (RandomForestClassifier):...'
#     newX = rfc.fit_transform(inpX,inpY)
#     mdl_rfc = rfc.fit(newX, inpY)
    mdl_rfc = rfc.fit(inpX, inpY)
    return mdl_rfc

def gbc_train(inpX, inpY, n_trees=100):
    # Fit regression model
    gbc = GradientBoostingClassifier(n_estimators=n_trees)
    print 'Starting ML training (GradientBoostClassifier):...'
#     newX = rfc.fit_transform(inpX,inpY)
#     mdl_rfc = rfc.fit(newX, inpY)
    mdl_gbc = gbc.fit(inpX, inpY)
    return mdl_gbc

"""
Main Program
"""	
if __name__ == '__main__':

#     allpos = True
    Debug = False
    
    if len(sys.argv) > 1:
        inps = sys.argv[1:]
        try:
            fnm_train = inps.pop(0)
            fnm_test = inps.pop(0)
#             print '>>', fname
        except:
            pass
    else:
#         exit('Need training set file as input!')

#         fnm_test = 'trainset_1st200.txt'
#         fnm_test = 'trainset_2nd200.txt'
        fnm_test = 'trainset_400_simp.txt' 
#         fnm_train = 'trainset_1st200.txt'
#         fnm_train = 'trainset_2nd200.txt'
        fnm_train = 'trainset_400_simp.txt'

    mtd_list = ['NaiveBayes','SVM','RandomForest', 'GradientBoost']
#     mtd_list = ['RandomForest']
    max_negf = 20

    # Load training data
    print 'Read in all training data from:', fnm_train
    trainX, trainY = load_inputs(fnm_train, max_negf)
    print '# of training points:', len(trainY)
    
    # Load test data
    print 'Read in test data from:', fnm_test
    testX, testY = load_inputs(fnm_test, 20)
    print '# of loaded points:', len(testY)
    
    # Train and test 
    for mthd in mtd_list:
        print 'Training model:', mthd
        if 'NaiveB' in mthd:
            model = nbc_train(trainX, trainY)
        elif 'SVM' in mthd:
            model = svc_train(trainX, trainY)
        elif 'RandomF' in mthd:
            model = rfc_train(trainX, trainY)
        elif 'GradientB' in mthd:
            model = gbc_train(trainX, trainY)

        analyse_test(fnm_test, testX, testY, model, mthd)
        
        # Save to model files
        try:
            inp = raw_input('Save model? (y/n): ')
            if inp.count('y'): 
                modelf = '%s_nf%s.pkl' % (mthd,max_negf)
                joblib.dump(model,modelf)
                print 'Model saved to file:', modelf    
        except:
            e = sys.exc_info()[0]
            exit('\nAborted: %s' % repr(e))        
        
    

    
