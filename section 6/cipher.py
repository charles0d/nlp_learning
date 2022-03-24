# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:35:16 2022

@author: E030751
"""

import numpy as np
import string
import random
import re

original_message = '''I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.
'''

V = 26
alphabet = list(string.ascii_lowercase)
cipher = [i for i in range(V)]
random.shuffle(cipher)
cipher_dict = {alphabet[i]: alphabet[cipher[i]] for i in range(V)}

encoded_msg = ''.join([cipher_dict[l] if l in alphabet else l 
               for l in original_message.lower()])



pi = np.zeros(V)
P = np.ones((V, V))

offset = ord('a')

def update_pi(c):
    i = ord(c) - offset
    pi[i] += 1

def update_P(c0, c1):
    i0, i1 = ord(c0) - offset, ord(c1) - offset
    P[i0, i1] += 1

regexp = re.compile('[^a-z]')

for line in open('moby_dick.txt', encoding = 'utf8'):
    line = line.rstrip().lower()
    words = [w for w in regexp.split(line) if len(w) > 0]
    if not words:
        continue
    for word in words:
        c0 = word[0]
        update_pi(c0)
        
        for c1 in word[1:]:
            update_P(c0, c1)
            c0 = c1
    
pi = pi/np.sum(pi)
P = P/np.sum(P, axis=1, keepdims=True)

def decode(dna):
    d = {alphabet[i]: dna[i] for i in range(V)}
    return ''.join([d[l] if l in alphabet else l 
                   for l in encoded_msg.lower()])

def fitness(dna):
    fitness = 0
    for word in regexp.split(decode(dna)):
        if len(word) == 0:
            continue
        c0 = word[0]
        i0 = ord(c0) - offset
        fitness += np.log(pi[i0])
        for c1 in word[1:]:
            i1 = ord(c1) - offset
            fitness += np.log(P[i0, i1])
            i0 = i1
    return fitness

def reproduction(dna):
    children = []
    for _ in range(3):
        child = dna.copy()
        for _ in range(random.randint(0, 2)):
            # swap two characters
            i = random.randint(0, 25)
            j = random.randint(0, 25)
            while j ==i:
                j = random.randint(0,25)
            tmp = child[i]
            child[i] = child[j]
            child[j] = tmp
        children.append(child)
        if len(set(child)) != len(child):
            print(dna)
            print(child)
            
    return children


dna_pool = [alphabet.copy() for _ in range(20)]
for dna in dna_pool:
    random.shuffle(dna)
    
epochs = 1000
best_fitness = -1000000000
for epoch in range(epochs):
    if epoch % 50 == 0:
        print(f'{epoch} / {epochs}')
    if epoch > 0:
        children = []
        for parent in dna_pool:
            children += reproduction(parent)
        dna_pool += children
    fitnesses = [fitness(dna) for dna in dna_pool]
    indices = np.argsort(fitnesses)[-5:][::-1]
    dna_pool = [dna_pool[i] for i in indices]
    if max(fitnesses) > best_fitness:
        best_fitness = max(fitnesses)
        print(best_fitness)
        print(decode(dna_pool[0]))
    
    