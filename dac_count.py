
import numpy as np
import sys, fileinput
from tokenizer import tokenize
import collections
import argparse, sys


labels = []
counter = []
vocab=set()
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, help='training data')
args = parser.parse_args()
data1=[]
data=[]

for line in open(args.train):
        a= ' '
        fline, eline = line.split('\t')
        #print(fline)
        fline = ['CLS'] + fline.split()
        a1=a.join(list(fline))
        fwords = tokenize(a1)
        #a.append(fwords)
        ewords = eline.split()
        data1.append((fwords, ewords))
        data.append((fline, ewords))
       
traindata = data
#print(data)
word=[]
label=[]
for line in traindata:
  (wor,lab)=line
  #wordn= tokenize(wor)
  for l in lab:
    if l not in label:
      label.append(l)
  for w in wor:
    if w not in word:
      word.append(w)
      counter.append(1)
    else:
      word_index = word.index(w)
      counter[word_index] = counter[word_index] + 1
 
unknown = []
for i in range(len(counter)):
  if counter[i] == 1:
    unknown.append(word[i])
#print(unknown)
for words, labels in traindata:
    pieces = []
    a= []
    for word in words:
        #print(word)
        #print(labels)
        if word in unknown:
            word = 'unk'
        a.append(word)
    a1= ' '.join(a)
    #print(a1)
    #pieces.append(a1 + '\t' + labels)

    sentence_string = a1 + '\t' + labels[0]
    #sentence= '<CLS> '+sentence_string
    print(sentence_string)


