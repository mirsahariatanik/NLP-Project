from dateparser.search import search_dates
from dateutil.parser import parse
import spacy
from spacy import displacy
import numpy
import re
import datetime

NER = spacy.load("en_core_web_sm")


def read_data(filename):
    return [list(line.rstrip('\n')) for line in open(filename)]


def read_parallel(filename):
    """Read data from the file named by 'filename.'
    The file should be in the format:
    我 不 喜 欢 沙 子 \t i do n't like sand
    where \t is a tab character.
    Argument: filename
    Returns: list of pairs of lists of strings. <EOS> is appended to all sentences.
    """
    data = []
    for line in open(filename):
        fline, eline = line.split('\t')
        fwords = fline.split() + ['<EOS>']
        ewords = eline.split() + ['<EOS>']
        data.append((fwords, ewords))
    return data
# train,data = read_parallel('train.txt')

for line in open('dev_words.txt'):
    fline = line
    #print(eline(0))
    #fwords = fline.split()
    #ewords = eline.split()
    #print(fline)
    #text1 = NER(fline)
    # d=search_dates(line)
    # print(d)
    sum=0
    total=0
    e = parse(fline, fuzzy_with_tokens=True)
    title = e[1][0]
    
    title1 = e[0]
    #print(d)
    frame= 'schedule' +' '+'('+' '+'title'+' '+'=' + ' '+ str(title)+' '+';'+ ' '+'date'+' '+ '='+' '+str(e[0])+' '+ ';'+' '+'start-time'+' '+'='+' '+str(e[0])+' '+';'+' '+'end-time'+' '+'='+' '+str(e[0])+' '+')'
    print(frame)
    #print(sum)
    #print(total)
    #print('Title:', e[1][0])
    #print('Date Time:', e[0])
    
    # displacy.render(text1,style="ent",jupyter=True)
    # for word in text1.ents:
    # print(word.text, word.label_)
#print(sum/total)
