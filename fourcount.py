import numpy as np
import sys, fileinput
import seqlabel
import collections

labels = []
counter = []
vocab=set()
#for li, line in enumerate(fileinput.input()):

rr = seqlabel.read_labels(fileinput.input())
#print(rr[0])
 
for line in rr:
    (w,label)=line
    for l in label:
      if l not in labels:
          labels.append(l)
          
          
#Total number of unique labels
#print("No. of unique labels: ", len(labels))

word=[]
for line in rr:
  (wor,lab)=line
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
for words, labels in rr:
    pieces = []
    for word, label in zip(words, labels):
        if word in unknown:
            word = '<unk>'
        pieces.append(word + ':' + label)

    sentence_string = ' '.join(pieces)
    sentence= 'S:<BOS> '+sentence_string+' E:<EOS>'
    print(sentence)