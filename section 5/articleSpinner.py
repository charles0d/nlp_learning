# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:05:01 2022

@author: E030751
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer 
from collections import defaultdict
import numpy as np
import random as r


df = pd.read_csv('bbc_text_cls.csv')
df = df[df['labels'] == 'tech']

corpus = [word_tokenize(text) for text in df['text']]

triplets = {}
for j, doc in enumerate(corpus):
    if j%50 == 0 or j == len(corpus)-1:
        print(f'doc {j}')
    for i, token in enumerate(doc):
        if i == 0 or i == len(doc)-1:
            continue
        
        if (doc[i-1], doc[i+1]) not in triplets:
            triplets[doc[i-1], doc[i+1]] = defaultdict(int)
        
        triplets[doc[i-1], doc[i+1]][token] += 1 
        
for k in triplets:
    tot = sum(triplets[k].values()) 
    triplets[k] = {j: triplets[k][j]/tot for j in triplets[k]}

def sample(d):
    ''' d is a probability dict, return a sample from d'''
    assert(round(sum(d.values()), 8) == 1)
    e = r.random()
    cum = 0
    for k in d:
        cum += d[k]
        if cum >= e:
            return k

# split texts in line and spin lines

def spin_doc(doc, rate = 0.3):
    lines = doc.split('\n')
    res = []
    for line in lines:
        if line:
            new_line = spin_line(line, rate)
        else:
            new_line = line
        res.append(new_line)
    return '\n'.join(res)

detokenizer = TreebankWordDetokenizer()


def spin_line(line, rate):
    previous = False
    tokens = word_tokenize(line)
    res = []
    for i, w in enumerate(tokens):
        res.append(w)
        if i == 0 or i == len(tokens)-1 or rate < r.random() or previous:
            previous = False
            continue
        middle = sample(triplets[(tokens[i-1], tokens[i+1])])
        if middle == w:
            continue
        res.append("<" + middle + '>')
        previous = True
    return detokenizer.detokenize(res)

print(spin_doc(df.iloc[0,0], rate=0.4))