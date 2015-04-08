#!/usr/bin/python -u
import sys, re, os
import subprocess, ner, datetime
import json
import numpy as np

from subprocess import Popen
from bs4 import BeautifulSoup as bs
from dateutil import parser as dtParser

from sklearn.externals import joblib
from imo_utils import _debug_pause, _printf

def helper():
    print """\
Usage: %s [-para] [inputFile1 inputFile2 ...] (By WL)
para: 'L'  -> Load trained ML model(s) from file(s) separated like "m1.pkl,m2.pkl" \
      'D'  -> Turn on expert debug mode
""" % sys.argv[0]

class Features:
    def __init__(self, cmodels, jsOut=False):
        self.client_lenlim = 9
        self.file_types = ['doc', 'pdf', 'ppt', 'xls', 'rtf', 'vsd', 'txt']
        self.kwds_client = re.compile(r'^client\b', flags=re.I)
        self.kwds_date = re.compile('date|period.end|date.created', flags=re.I)
        self.kwds_author = re.compile('prepared.by|reviewed.by|changed.by|participants', flags=re.I)
        self.kwds_ignore = r'\b(CLIENT|PROJECT|ENGAGEMENT|EMPLOYEE|NAME|PARTNER|MANAGER|REPORT|TABLE|REVIEW|CONTROL|US|PERIOD|IT|IS|IN|MOBI|YES|NO|OFF|ON|DATE|END)\b'
        self.kwds_replace = r'\b(OF|IN|AT|WITH|OR|FOR|BY|THE|AS|TO|AND)\b'
        # Ad-hoc definitions of high frequency phrases with long feature list
#         self.kwds_hfreq = '|INC|CORP|LLP|INVOICE|COMPUTERS|STAFF|CYCLE|SALES COMMISSIONS|ELECTRIC CARS|FORM|PROCUREMENT|PRODUCTION|PAYROLL|REPORTING|MARKET|PRIVACY|AND|RISK|\
# PARTICIPANTS|PLANNING|UP|SET|FUNCTION|BUDGET|NARRATIVE|APPROVAL|APPROVED|BAD DEBT|CONTRACT MANAGEMENT|CUSTOMER CARE|FIXED ASSETS|GENERAL ACCOUNTING|HR|COMPENSATION|\
# PRELIMINARY|BACKUP|JOBS|ISSUES|MESSAGES|OPINIONS|FILTERS|DRAFTS|STATUS|BACKGROUNDS|TESTING|ALL|CONTROLS|OPEN|BANK|CHANGES|PROCESS|SELF ASSESSMENT|REVIEWED|FINAL|\
# DOCUMENTATION REQUEST|DATA CENTER|LIST|MAP|MEETING|RETIREMENT|COMPLIANCE|SMALL|RULES|SYSTEMS DEVELOPMENTS|MATRIX|INFORMATIONS|CREDIT EVALUATION|STANDARD|WORKING|\
# QUESTION|DIAGNOSTIC|FINDINGS|SECURITY|DESCRIPTION|POLICY|ITGC|VALIDATION|ACCESS|PROPOSALS|'
        self.mdls = cmodels
        self.jsOut = jsOut

    def _cleanup_(self, inpStr):
        outStr = re.sub('[^a-zA-Z]',' ',inpStr)
        return re.sub('\s{2,}',' ',outStr).strip()

    def _freq_stat_entity(self, klist):
        kCount = {}
        all_list_str = '\t'.join(klist)
        for k in klist:
            if k.count(' '):
                try:
                    kCount[k] += 1
                except KeyError:
                    kCount[k] = 1
            else:
                # Possible acronym
                kCount[k] = all_list_str.count(k)        
                
        # sort dict by key: sorted(kCount, key=kCount.get)
        return kCount
    
    def _testClientFreq(self, entity, low_freq=True, thres=0.5):
    # Check for matching entity in low frequency KPMG client dictionary
        items = entity.split()
        cnt = sum([x in cname_lf for x in items])
        isLowFreq = False
        if len(items) > 1:
            isLowFreq = cnt > thres*len(items)
        elif cnt:
            isLowFreq = cname_lf[items[0]] < 11
        return isLowFreq == low_freq

    def _expand_nerDict(self, content, nerDict):
        ner_dict = nerTagger.get_entities(content)
        for key in nerDict:
            if key not in ner_dict: continue
            newlist = []
            for itm in ner_dict[key]:
                if 'KPMG' not in itm: # purify NER entities
                    newlist.append(self._cleanup_(itm).encode('utf-8').upper())
            nerDict[key] = newlist

    def _extract_entity(self, jsdict, soup, ftype):
        # Process NER for entities
        content = jsdict['ContentText'].replace('\n','. ')        
        nerDict = {'ORGANIZATION':[], 'PERSON':[]}
        if len(content) > 1E6:
            cont_sz, content_lines = 0, ''
            print '>>proc large content size =', len(content),': ',
            for sentence in content.split('. '):
                content_lines += sentence
                cont_sz += len(sentence)
                if cont_sz > 1E5:
                    sys.stdout.write('*')
                    self._expand_nerDict(content_lines, nerDict)
                    cont_sz, content_lines = 0, ''
            print ' [done!]'
        else:
            self._expand_nerDict(content, nerDict)
#         if Debug:
#             print nerDict
        if  len(nerDict['PERSON']):
            auth_dict = self._freq_stat_entity(nerDict['PERSON'])
            print 'Possible authors from NER:', auth_dict
            # Consider single author as possible client names
            for nmKey in auth_dict:
                # Check for matching entity in low frequency KPMG client dictionary
                if self._testClientFreq(nmKey):
                    try:
                        nerDict['ORGANIZATION'].extend([nmKey,nmKey])
                    except:
                        nerDict['ORGANIZATION'] = [nmKey,nmKey]
                
        ## Run to check for features ##
        candi_list, feat_list = set(),[]

        # f12 - "client" on previous line (eg. <p class="client">AGC Chemicals</p>)
        clientTag = soup.find('p',{'class':self.kwds_client})
        if clientTag: 
            client_name = clientTag.text
            if client_name and re.search('\w',client_name):
#                 client_name = filter(lambda x: ord(x)<128,client_name).encode('utf-8').strip()
                client_name = ''.join(map(lambda x: x if ord(x)<128 else ' ',client_name)).strip()
                if client_name:
                    client_name = self._cleanup_(client_name.upper())
                    print '>client class= "%s"' % client_name
                    if len(client_name.split()) < self.client_lenlim and self._testClientFreq(client_name, thres=0):
                        candi_list.add(client_name)
                        feat_list.extend(['f12'])
        # f14 - "Date" or "Period-end" date on first page
        date_list = soup.findAll('p',{'class':self.kwds_date})
        if date_list: 
            print '>dates class=', date_list
#             feat_list.append('f14')
        # f15 - "Prepared by" or "Reviewed by" etc. author on first page
        # f16 - "Prepared by" or "Reviewed by" etc. author on last page
        author_list = soup.findAll('p',{'class':self.kwds_author})
        if author_list: 
            print '>author class=', author_list
#             feat_list.append('f15')

        # Another attempt at <table> format for possible template
        if not candi_list:
            tag_list = soup.find_all('table')
            if 'doc' in ftype:
                for clientTag in tag_list:
                    if re.search(self.kwds_client,clientTag.text.strip()):
                        tr_list = clientTag.find_all('tr')
                        try:
                            trStr = tr_list[0].text.encode('utf-8').strip()
                            title_list = trStr.split('\n')
                            trStr = tr_list[1].text.encode('utf-8').strip()
                            value_list = trStr.split('\n')
                            if len(title_list) == len(value_list):
                                for idx, title in enumerate(title_list):
                                    if re.search(self.kwds_client, title):
                                        if re.search('Client$|Name', title, flags=re.I):
                                            client_name = value_list[idx].strip()
                                            if not client_name: continue
                                            print '>client table entry= "%s"' % client_name
                                            client_name = self._cleanup_(client_name.upper())
                                            if len(client_name.split()) < self.client_lenlim and self._testClientFreq(client_name, thres=0):
                                                candi_list.add(client_name)
                                                feat_list.extend(['f12'])
                                        continue
                                    if re.search(self.kwds_date, title):
                                        print '>date entry= "%s"' % value_list[idx]
                                        continue
                                    if re.search(self.kwds_author, title):
                                        print '>author entry= "%s"' % value_list[idx]
                                        continue
                        except:
                            print 'Found <table> template but failed to extract client name...'
                            pass
            elif 'xls' in ftype:
                for table in tag_list:
                    for rowTag in table.find_all('tr')[:6]: # Check only first six rows
                        rowText = rowTag.text.strip()
                        if re.search(self.kwds_client,rowText):
                            client_name = rowTag.find_all('td')[1].text.encode('utf-8').strip()
#                             if not client_name or re.search(self.kwds_ignore, client_name, flags=re.I): continue
                            if not client_name or self._testClientFreq(client_name,low_freq=False, thres=0): continue
#                             client_name = re.sub(self.kwds_ignore,'',self._cleanup_(rowText),flags=re.I).strip()
                            print '>client entry= "%s"' % client_name
                            candi_list.add(self._cleanup_(client_name.upper()))
                            feat_list.extend(['f12'])
                            break
                    break

        if candi_list: # Already determined client_name
            pass
#             print 'Date & Author features:', feat_list 
        else:
            if len(nerDict['ORGANIZATION']): # Obtain potential candidate from NER            
                print 'Cleaned client candidates from NER:'
                og_list = self._freq_stat_entity(nerDict['ORGANIZATION'])
                for ogKey in sorted(og_list, key=og_list.get, reverse=True):
                    cnt = og_list[ogKey] 
#                     if cnt<2: break # limit for occurrence
#                     if len(ogKey) < 2 or re.search(self.kwds_ignore,ogKey): continue                    
                    if self._testClientFreq(ogKey, thres=0):
                        print '>',ogKey,':',cnt
                        candi_list.add(ogKey)
            og_list = list(candi_list) # Re-exam NER extractions
            # Consider folder acronym and check if in file name
            dirStr =  jsdict['FileDir'].split('DCRE')[-1]
            dstr_list = re.split('[/\\\]',dirStr[1:])
            for itm in dstr_list:
                if not itm or re.search('\d',itm): continue
                if itm in jsdict['FileName'] or re.search(itm,jsdict['ContentText'],flags=re.I):
                    candi_list.add(itm)
                    print 'from path>', itm
            # Consider string/acronym in file name
            dirStr = ' '.join(os.listdir(jsdict['FileDir']))
            fstr_list = jsdict['FileName'].split()
            og, fstr = False, ''
            for itm in fstr_list:
                if re.search('^\W',itm):
                    break
                fstr += itm+' '
                if len(re.findall(r'\b%s' % fstr, dirStr, flags=re.I)) > 1:
                    og = self._cleanup_(fstr)
                    if og and self._testClientFreq(og): #low_freq=len(og.split())>1 #for possible acronym
#                     if og and og not in og_list:
                        candi_list.add(og)
                        print 'from filename>', og
                
            # Take consideration of first ten lines candidate with NER pool
            print 'Re-checking candidate name pool with first 10 lines or less:'
            last_idx = 10
            content_lines = jsdict['ContentText'].split('\n')
            if last_idx > len(content_lines):
                last_idx = len(content_lines)
            for idx in range(0,last_idx):
                line =  content_lines[idx].upper()
                line = re.sub('SHEET\d+|KPMG','',line)
#                 line = re.sub('\s+',' ',line)
                line = re.sub('[-:()]|\s{2,}',';',line)
                line = line.translate(None,'.&#').strip()
                if line:
                    og_list.extend(line.split(';'))
            for og in og_list: 
                og = og.strip()
                if len(og)>1 and len(og.split()) < self.client_lenlim:
                    if re.search('\d+', og):
                        try:
                            t = dtParser.parse(og)
                            if isinstance(t, datetime.datetime):
                                dt = t.date().isoformat()
                                date_list.append(dt)
#                                 print '>date_list=', date_list
                                continue
                        except:
                            pass
#                     og = og.translate(None,'.&#')
#                     if re.search(self.kwds_ignore,og): continue
                    og = self._cleanup_(og)
                    # Check for matching entity in low frequency KPMG client dictionary
                    if self._testClientFreq(og, low_freq=False): continue
                    if len(og)<3 or og in self.kwds_replace: continue
                    if og not in candi_list:
                        candi_list.add(og)
                        print 'added>', og
                    og = re.sub(self.kwds_replace,'',og).strip()
                    if len(og.split())>2: # 3+ letters
                        new_og = ''.join([itm[0] for itm in og.split()])
                        if new_og not in candi_list:
                            candi_list.add(new_og)
                            print 'added acron>',new_og
                    # f11 check for good string/acronym 
                    new_og_list = og.split()
                    list_len = len(new_og_list) 
                    if list_len>1:
                        for idx in range(2,list_len+1):
                            new_og_list.append(' '.join(new_og_list[:idx]))
                        for new_og in new_og_list:
                            if len(new_og)<3 or new_og in candi_list: continue 
                            if new_og in jsdict['FilePath']:
                                candi_list.add(new_og)
                                print 'added split>',new_og

        # Final purge on candi_list
#         dirStr = jsdict['FileDir'].split('DCRE')[-1]
#         for client_candi in list(candi_list):
#             if client_candi in dirStr:
#                 if '|'+client_candi+'|' in self.kwds_ignore+self.kwds_hfreq:
#                     candi_list.remove(client_candi)
#             elif client_candi in self.kwds_ignore+self.kwds_hfreq:
#                 candi_list.remove(client_candi)

        if not candi_list:
            print 'Empty candidate list!\n'
            return False,False
        
        if date_list:
            print '>date_list=', date_list            
        
        return candi_list, feat_list

    def _check_feature(self, jsdict, candi_list, feat_list, ftype):
        content_lines = jsdict['ContentText'].split('\n')
        # Exam each client candidate in list    
        candi_score = {}
        print 'Client name candidates with more than one features:'
        for client_candi in candi_list:
            
            # For consideration of special character in between
            cli_candi_pat = r'\b%s\b' % client_candi.replace(' ','\W{,3}') 

            # f1 - name contain common suffix (Inc, LLP, corporation, company, bank etc)
            if re.search(r'Inc|LLP|corp|company|bank', client_candi, flags=re.I):
                feat_list.append('f01')

            # f2 - occupy a single line, and less than 8 words
            last_line = 25
            if len(content_lines) < last_line:
                last_line = len(content_lines)
            
            for line in content_lines[:last_line]:
                line = line.upper()
                if len(line.split()) < 8:
                    if re.search(cli_candi_pat, line, flags=re.I):
                        feat_list.append('f02')
                        break

            # f3 - appear on first 6 lines of the document
            last_idx = 6
            if len(content_lines) < last_idx:
                last_idx = len(content_lines)
            for line in content_lines[:last_idx]:
                if re.search(cli_candi_pat, line, flags=re.I):
#                 line = line.upper()
                    feat_list.append('f03')
                    break
                
            # f14 - appear at the end of the file
            for line in content_lines[-last_idx:]:
                if re.search(cli_candi_pat, line, flags=re.I):
#                 line = line.upper()
#                 if client_candi in line or seqm(None, line, client_candi).ratio() > 0.7:
                    feat_list.append('f14')
                    break            
            
            # f4 - appear on first page of a document
            for line in content_lines[:last_line]:
                if re.search(cli_candi_pat, line, flags=re.I):
                    feat_list.append('f04')
                    break
            
            # f5 - appear on last page of a document
            last_line = 25
            if len(content_lines) > last_line*2:
                for line in content_lines[-last_line:]:
                    if re.search(cli_candi_pat, line, flags=re.I):
                        feat_list.append('f05')
                        break
            
            if client_candi.count(' '):
            # f6 - same string appear multiple times in the same file
            # f7 - same string appear in other files under the same directory
                if len(re.findall(cli_candi_pat,jsdict['ContentText'],flags=re.I)) > 1:
                    feat_list.append('f06')
                if len(re.findall(cli_candi_pat,' '.join(os.listdir(jsdict['FileDir'])), flags=re.I)) > 1:
                    feat_list.append('f07')
            else:
            # f8 - acronym appear in multiple places of the file
            # f9 - acronym appear in other files under the same directory
            # f17 - entity is a full name with multiple string or single acronym/string
                if len(re.findall(cli_candi_pat,jsdict['ContentText'],flags=re.I)) > 1:
                    feat_list.append('f08')
                if len(re.findall(cli_candi_pat,' '.join(os.listdir(jsdict['FileDir'])), flags=re.I)) > 1:
                    feat_list.append('f09')
                feat_list.append('f17')
            
            # f10 - same string/acronym appear in file name
            if client_candi in jsdict['FileName']:
                feat_list.append('f10')
            # f11 - same string/acronym appear in the file path (not include file name)
            if client_candi in jsdict['FileDir'].split('DCRE')[-1]:
                feat_list.append('f11')
            # f13 - with xxx, to xxx, of xxx
            if re.search(r'\b(to|of|at|by|for|with)\b\s+%s' % cli_candi_pat, jsdict['ContentText'], flags=re.I):
                feat_list.append('f13')
            # f15 - Prepared for: xxx
            if re.search('Prepared for:.{,3}%s' % cli_candi_pat,jsdict['ContentText'],flags=re.I):
                feat_list.append('f15')
            # f16 - between KPMG and xxx; between xxx and KPMG
            if re.search(r'between \b(KPMG and %s|%s and KPMG)\b' % (cli_candi_pat,cli_candi_pat),jsdict['ContentText'],flags=re.I):
                feat_list.append('f16')            

            feat_cnt = len(feat_list)
            feat_score = feat_cnt

            # append file type as special feature
            ftype = ftype[:3]
            if ftype in self.file_types:
                feat_list.append('ft%d' % (self.file_types.index(ftype)+1))
            else:
                feat_list.append('ft%d' % (len(self.file_types)+1))
            
            if feat_cnt>1: 
#                 print client_candi,'<', feat_list, '(%d)' % feat_cnt
                feat_list.sort()
                if self.mdls:
                    feat_score = self._model_pred(feat_list)
                candi_score[client_candi] = feat_score
                print "%s(%.2f)\t%s" % (client_candi,feat_score,'\t'.join(feat_list))                
            feat_list = []

        print 'Ranked by feature score or count:',
        last_idx = 8
        if last_idx > len(candi_list):
            last_idx = len(candi_list)
        client_list = [] 
        for client_candi in sorted(candi_score, key=candi_score.get, reverse=True)[:last_idx]:
            feat_score = candi_score[client_candi]
            if self.mdls:
                if feat_score < 0.1: break
            elif feat_cnt < 2 : break
            client_list.append(client_candi)
        
        if not client_list and self.mdls:
            # 2nd round check for ranking
            print '(2nd scan)',
            if last_idx > 3: last_idx = 3
            for k in candi_score: candi_score[k] = -abs(candi_score[k]) 
            for client_candi in sorted(candi_score, key=candi_score.get)[:last_idx]:
                feat_score = abs(candi_score[client_candi])
                if feat_score < 0.01: break
                client_list.append(client_candi)
        
        print ' | '.join(client_list)
        if self.jsOut:
            self.jsOut.write(json.dumps({'FilePath':jsdict['FilePath'],'ClientCandidates':client_list})+'\n')
#             print >>self.jsOut, json.dumps({'FilePath':jsdict['FilePath'],'ClientCandidates':client_list})+',' 
#             print client_candi,'|',
#         print ''

        if Debug:
            _debug_pause()
            
    def _model_pred(self, feat_list):
        inpX = np.zeros(18, dtype=np.int)
        for feat in feat_list:
            try:
                inpX[int(feat[1:])-1] = 1
            except: # ftxx for file type
                inpX[17] = int(feat[2:])
        # Special consideration for 'f01' name contain common suffix (Inc, LLp, Corp, etc.)
        if inpX[0]:
            pred = True
        else: 
            pred = self.mdls[0].predict(inpX)
        feat_score = self.mdls[0].predict_proba(inpX)[0][1] + self.mdls[1].predict_proba(inpX)[0][1]
        feat_score *= 0.5 
        if pred:
            sys.stdout.write('*')
        else:
            feat_score *= -1
        return feat_score

def _extract_content(efile, jsdict):
    # Obtain decent pure text content
    _printf(efile)
    lineStrs = ''
    lcount = 0
    try:
        for line in Popen('nc 127.0.0.1 3333 < "%s"' % efile, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.readlines():
#             line = filter(lambda x: ord(x)<128,line).encode('utf-8').strip()
            line = ''.join(map(lambda x: x if ord(x)<128 else '-',line))
            line = re.sub('\s{2,}',' ',line).strip()
            if line:
                lineStrs += line+'\n'
                lcount += 1
        # Check if content too long over 16MB - max document limit for MongoDB
        if len(lineStrs) > 16E6: #16777216:
            raise Exception('Text too long over 16MB')
        if not lineStrs:
            _printf(' >> Empty Content\n')
            return False
        _printf(' >> Content OK (%d lines)' % lcount)
        jsdict["ContentText"] = lineStrs
    except Exception as csExp:
        _printf(' >> Content Error: ' + csExp.args[0] + '\n')
        return False
    except:
        e = sys.exc_info()[0]
        _printf(' >> Content Error: ' + repr(e) + '\n')
        _printf(_printf(Popen('tika -t < "%s" ' % efile, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()))    
        return False
    
    # Obtain html content
    try:
        lineStrs = Popen('nc 127.0.0.1 4444 < "%s"' % efile, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()
        soup = bs(lineStrs).body
    except Exception as csExp:
        _printf(' >> Html Content Error: ' + csExp.args[0] + '\n')
        return False
    except:
        e = sys.exc_info()[0]
        _printf(' >> Html Content Error: ' + repr(e) + '\n')
        return False    
    _printf("\n")
    
#     if Debug:
#         print jsdict
    
    return soup
    
"""
Main Program
"""	
if __name__ == '__main__':

    # Global variables
    (para, Debug, jsOut) = [False for i in range(3)]
    ml_models = []

    # Read in additional parameter(s)
    try:
        if re.match('^-', sys.argv[1]):
            para = sys.argv[1][1:]
            inps = sys.argv[2:]
        else:
            inps = sys.argv[1:]
    except:
        inps = ['ml_models/NaiveBayes_nf3.pkl,ml_models/RandomForest_nf20.pkl','sample_files_test.txt']
        para = 'LD'
        
    if para:
        if para.count('L'):
            model_files = inps.pop(0).split(',')
            if len(model_files)<2:
                model_files.append(model_files[0])
            try:
                for idx, mdlf in enumerate(model_files):
                    ml_models.append(joblib.load(mdlf))
                    print 'Loaded model #%d from file: %s' % (idx+1,mdlf)
                jsOut = open('extracted_clients_json.txt','w')
            except:
                e = sys.exc_info()[0]
                print 'Failed to load model:', mdlf, 'Error:', repr(e) 
        if para.count('D'):
            Debug = True
            print '>>Debug mode ON.'
        
    if len(inps) < 1:
        helper()
        exit('Need input file list.')
    
    # Load low frequency client tokens
    clf_file = 'client_tokens_lf150.json'
    cname_lf = json.load(open(clf_file))
    
    fname = inps.pop(0)
    LIN = open(fname,'r')
    print 'Read in file list from:', fname
#     LIN.readline() # skip first line

    # Initialize Apache Tika server for meta & text mode
#     os.system("java -Xmx300m -jar /usr/local/Cellar/tika/1.5/libexec/tika-app-1.5.jar -j --server --port 1111 &")
#     os.system("java -Xmx300m -jar /usr/local/Cellar/tika/1.5/libexec/tika-app-1.5.jar -t --server --port 2222 &")
#     subprocess.call("java -Xmx300m -jar /usr/local/Cellar/tika/1.5/libexec/tika-app-1.5.jar -j --server --port 1111 &", shell=True, stderr=subprocess.PIPE)
    subprocess.call("java -Xmx300m -jar /usr/local/Cellar/tika/1.5/libexec/tika-app-1.5.jar -t --server --port 3333 &", shell=True, stderr=subprocess.PIPE)
    subprocess.call("java -Xmx300m -jar /usr/local/Cellar/tika/1.5/libexec/tika-app-1.5.jar -h --server --port 4444 &", shell=True, stderr=subprocess.PIPE)
    
    # Load up NER server for Stanford Named Entity Recognizer
    os.chdir('stanford-ner/')
    nerCmd = "java -Xmx300m -cp stanford-ner.jar edu.stanford.nlp.ie.NERServer -loadClassifier classifiers/english.all.3class.distsim.crf.ser.gz -port 1234 -outputFormat inlineXML &"
    subprocess.call(nerCmd, shell=True, stderr=subprocess.PIPE)
    nerTagger = ner.SocketNER(host='localhost', port=1234)
    
    feats = Features(ml_models, jsOut)
    
    for line in LIN:
        docf = line.strip()
        ftype = os.path.splitext(docf)[1][1:]
        jsdict = {}
        jsdict['FilePath'] = docf.split('DCRE/')[-1]
        jsdict['FileDir'] = os.path.dirname(docf).upper()
        jsdict['FileName'] = os.path.basename(docf).upper()
        if ftype in 'tmp|zip':
            print docf,'>> Ignore Type'
            if jsOut:
                jsOut.write(json.dumps({'FilePath':jsdict['FilePath'],'ClientCandidates':[]})+',\n')
            continue
        # Extract & normalize file info
        soup = _extract_content(docf, jsdict)
        if not soup: 
            if jsOut:
                jsOut.write(json.dumps({'FilePath':jsdict['FilePath'],'ClientCandidates':[]})+',\n')
            continue
        candi_list, feat_list = feats._extract_entity(jsdict, soup, ftype)
        if candi_list:
            feats._check_feature(jsdict, candi_list, feat_list, ftype)
        elif jsOut:
            jsOut.write(json.dumps({'FilePath':jsdict['FilePath'],'ClientCandidates':[]})+',\n')
    
    if jsOut:
        jsOut.close()
    
    if Debug:        
        try:
            inp = raw_input('Shutdown Tika servers? (y/n): ')
            if inp.count('y'): 
                subprocess.call("pkill -f 'java -Xmx300m'", shell=True)
        except:
            e = sys.exc_info()[0]
            exit('\nAborted: %s' % repr(e))
