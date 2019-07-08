import sys
from collections import Counter
import re
from gurobipy import *
import gzip
from textblob import *
import os
import time
import codecs
import math
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import aspell
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import pylab as pl
from itertools import cycle
from operator import itemgetter

LSIM = 0.7
lmtzr = WordNetLemmatizer()
Tagger_Path = ''
ASPELL = aspell.Speller('lang', 'en')
WORD = re.compile(r'\w+')
cachedstopwords = stopwords.words("english")
AUX = ['be','can','cannot','could','am','has','had','is','are','may','might','dare','do','did','have','must','need','ought','shall','should','will','would','shud','cud','don\'t','didn\'t','shouldn\'t','couldn\'t','wouldn\'t']

NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
              "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
              "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
              "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
              "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
              "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
              "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
              "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

def compute_similarity(ifname,keyterm,placefile,date,Ts):

	PLACE = {}
        fp = open(placefile,'r')
        for l in fp:
                if PLACE.__contains__(l.strip(' \t\n\r').lower())==False:
                	PLACE[l.strip(' \t\n\r').lower()] = 1
        fp.close()

	FTW = []
	fp = open(ifname,'r')
	fo = open('temp.txt','w')
	for l in fp:
		wl = l.split('\t')
		t = (wl[0].strip(' \t\n\r'),float(wl[2]),int(wl[4]),float(wl[5]))
		FTW.append(t)
		fo.write(wl[0].strip(' \t\n\r'))
		fo.write('\n')
	fp.close()
	fo.close()
	
	t0 = time.time()
	command = Tagger_Path + './runTagger.sh --output-format conll temp.txt > tag.txt'
	os.system(command)

	fp = open('tag.txt','r')
	
	T = {}
	TW = {}
	P = []
	index = 0
	temp = set([])
	TAGREJECT = ['#','@','~','U',',','E','G']
	L = 0
	
	for l in fp:
		wl = l.split('\t')
		if len(wl)>1:
			word = wl[0].strip(' #\t\n\r').lower()
                        tag = wl[1].strip(' \t\n\r')
			if tag not in TAGREJECT:
				L+=1
			if PLACE.__contains__(word)==True:
                                s = word
                                temp.add(s)
                        elif tag=='$':
                                s = word
                                try:
                                        Q = s
                                        #print('1',Q)
                                        for x in RPL:
                                                Q = s.replace(x,'')
                                                s = Q
                                        #print('2',s,type(s))
                                        w = str(numToWord(int(s)))
                                        #print('here',w,s,type(w),type(s))
                                        if len(w.split())>1: # like 67
                                                w = s
                                        #print('double',w,s,type(w),type(s))
                                except Exception as e:
                                        w = str(s)
                                word = w.lstrip('0')
                                s = word
                                temp.add(s)
                        elif tag=='^' and ASPELL.check(word)==1 and word not in cachedstopwords:
                                w = lmtzr.lemmatize(word)
                                s = w
                                if len(word)>1:
                                        temp.add(s)
                        elif tag=='N' and ASPELL.check(word)==1 and word not in cachedstopwords:
                                w = lmtzr.lemmatize(word)
                                s = w
                                if len(word)>1:
                                        temp.add(s)
			elif tag=='V' and ASPELL.check(word)==1:
                                try:
                                        w = Word(word)
                                        x = w.lemmatize("v")
                                except Exception as e:
                                        x = word
                                if x not in AUX and x not in cachedstopwords and x not in NEGATE:
                                        s = x
                                        if len(word)>1:
                                                temp.add(s)
                        else:
                        	pass

		else:
			q = FTW[index]
			T[index] = [q[0],q[1],L,q[3],temp]
			P.append(q[3])
			index+=1
			temp = set([])
			L = 0

	fp.close()

	Q = set_weight(P,0,1)
	print(Q[0],Q[len(Q)-1])
	L = len(T.keys())
	################ Select centroid from each cluster ###############################
	tweet_cur_window = {}
        for i in range(0,L,1):
        	temp = T[i]
                tweet_cur_window[i] = [temp[0].strip(' \t\n\r'),int(temp[2]),temp[4],Q[i],float(temp[1])]

	##################### First apply pagerank based cowts ################################
        ofname = keyterm + '_cowabs_' + date + '.txt'
        optimize(tweet_cur_window,ofname,Ts,0.4,0.6)
	t1 = time.time()
        print('Summarization done: ',ofname,' ',t1-t0)

def set_weight(P,L,U):
        min_p = min(P)
        max_p = max(P)

        x = U - L + 4.0 - 4.0
        y = max_p - min_p + 4.0 - 4.0
        factor = round(x/y,4)

        mod_P = []
        for i in range(0,len(P),1):
                val = L + factor * (P[i] - min_p)
                mod_P.append(round(val,4))

        count = 0
        return mod_P

def optimize(tweet,ofname,L,A1,A2):


        ################################ Extract Tweets and Content Words ##############################
        word = {}
        tweet_word = {}
        tweet_index = 1
        for  k,v in tweet.iteritems():
                set_of_words = v[2]
                for x in set_of_words:
                        if word.__contains__(x)==False:
				word[x] = 1

                tweet_word[tweet_index] = [v[1],set_of_words,v[0],v[3],v[4]]  #Length of tweet, set of content words present in the tweet, tweet itself, tweet id, confidence score
                tweet_index+=1

        ############################### Make a List of Tweets ###########################################
        sen = tweet_word.keys()
        sen.sort()
        entities = word.keys()
        print(len(sen),len(entities))

        ################### Define the Model #############################################################

        m = Model("sol1")

        ############ First Add tweet variables ############################################################

        sen_var = []
        for i in range(0,len(sen),1):
                sen_var.append(m.addVar(vtype=GRB.BINARY, name="x%d" % (i+1)))

        ############ Add entities variables ################################################################

        con_var = []
        for i in range(0,len(entities),1):
                con_var.append(m.addVar(vtype=GRB.BINARY, name="y%d" % (i+1)))

        ########### Integrate Variables ####################################################################
        m.update()

	P = LinExpr() # Contains objective function
        C1 = LinExpr()  # Summary Length constraint
        C4 = LinExpr()  # Summary Length constraint
        C2 = [] # If a tweet is selected then the content words are also selected
        counter = -1
        for i in range(0,len(sen),1):
                P += tweet_word[i+1][3] * tweet_word[i+1][4] * sen_var[i]
                C1 += tweet_word[i+1][0] * sen_var[i]
                v = tweet_word[i+1][1] # Entities present in tweet i+1
                C = LinExpr()
                flag = 0
                for j in range(0,len(entities),1):
                        if entities[j] in v:
                                flag+=1
                                C += con_var[j]
                if flag>0:
                        counter+=1
                        m.addConstr(C, GRB.GREATER_EQUAL, flag * sen_var[i], "c%d" % (counter))

        C3 = [] # If a content word is selected then at least one tweet is selected which contains this word
        for i in range(0,len(entities),1):
                P += con_var[i]
                C = LinExpr()
                flag = 0
                for j in range(0,len(sen),1):
                        v = tweet_word[j+1][1]
                        if entities[i] in v:
                                flag = 1
                                C += sen_var[j]
                if flag==1:
                        counter+=1
                        m.addConstr(C,GRB.GREATER_EQUAL,con_var[i], "c%d" % (counter))

	counter+=1
        m.addConstr(C1,GRB.LESS_EQUAL,L, "c%d" % (counter))


        ################ Set Objective Function #################################
        m.setObjective(P, GRB.MAXIMIZE)

        ############### Set Constraints ##########################################

        fo = open(ofname,'w')
        try:
                m.optimize()
                for v in m.getVars():
                        if v.x==1:
                                temp = v.varName.split('x')
                                if len(temp)==2:
                                        fo.write(tweet_word[int(temp[1])][2])
                                        fo.write('\n')
        except GurobiError as e:
                print(e)
                sys.exit(0)

        fo.close()

def numToWord(number):
        word = []
        if number < 0 or number > 999999:
                return number
                # raise ValueError("You must type a number between 0 and 999999")
        ones = ["","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
        if number == 0: return "zero"
        if number > 9 and number < 20:
                return ones[number]
        tens = ["","ten","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
        word.append(ones[int(str(number)[-1])])
        if number >= 10:
                word.append(tens[int(str(number)[-2])])
        if number >= 100:
                word.append("hundred")
                word.append(ones[int(str(number)[-3])])
        if number >= 1000 and number < 1000000:
                word.append("thousand")
                word.append(numToWord(int(str(number)[:-3])))
        for i,value in enumerate(word):
                if value == '':
                        word.pop(i)
        return ' '.join(word[::-1])


def main():
	try:
		_, ifname, keyterm, placefile, date, Ts = sys.argv
	except Exception as e:
		print(e)
		sys.exit(0)
	compute_similarity(ifname,keyterm,placefile,date,int(Ts))
	print('Koustav Done')

if __name__=='__main__':
	main()
