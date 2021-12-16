import torch
import sys, fileinput
device = 'cuda'

import math, collections.abc, random, copy
import numpy 
import layers 
from seqlabel import *

def progress(iterable):
    import os, sys
    if os.isatty(sys.stderr.fileno()):
        try:
            import tqdm
            return tqdm.tqdm(iterable)
        except ImportError:
            return iterable
    else:
        return iterable

class Model(torch.nn.Module):
    """Neural parsing model."""
    def __init__(self, rules, vocab, dims):
        super().__init__()

        # CFG rules and mapping to numbers
        self.rules = rules
        self.rule_index = {r:i for (i,r) in enumerate(rules)}

        # Vocabulary (terminal alphabet) and mapping to numbers
        self.vocab = list(vocab)
        #print(self.vocab)
        self.vocab_index = {w:i for (i,w) in enumerate(self.vocab)}
        #print(self.vocab_index)
        # Parameters for encoder
        self.emb = layers.Embedding(len(self.rule_index), dims)
        self.rnn1 = layers.RNN(dims)
        self.rnn2 = layers.RNN(dims)

        # Parameters for rule model
        #self.softmax= layers.SoftmaxLayer(dims, len(self.vocab_index))
        self.linear = layers.LinearLayer(dims, len(self.vocab_index))
        self.T= torch.nn.Parameter(torch.randn(len(self.vocab_index), len(self.vocab_index), requires_grad=True))
        
    def encode(self, words):
        
        # numberize words
        #eos = self.rule_index['<EOS>']
        #bos = self.rule_index['<BOS>']
        #unk = self.rule_index['<unk>']
        nums = torch.tensor([self.rule_index.get(w) for w in words])

        # look up word embeddings
        V = self.emb(nums)
        # run RNN
        G= self.rnn1.sequence(V)
        H = self.rnn2.sequence(G)
        O= self.linear(H)
        #print('O size')
        #print(O.size())
        return O

    def score_tree(self, words, labels):
        """Compute the log-weight of a tree.

        Argument:
        - tree (Tree): tree

        Return:
        - float: tree log-weight (sum of rule log-weights)
        """
        for i,word in enumerate(words):
          if word not in self.rules:
            words[i]='<unk>'
            
        O = self.encode(words)
        
        chart = {}
        T= self.T
        n= len(words)
        score=0
        for i in range(n-1):
          fnum= self.vocab_index.get(labels[i])
          fnum1= self.vocab_index.get(labels[i+1])
          score += T[fnum, fnum1]+ O[i, fnum]
        score+=O[n-1,fnum1]
        return score
        

    def labeler(self, words):
    
        for i,word in enumerate(words):
          if word not in self.rules:
            words[i]='<unk>'
            
        O = self.encode(words)
        
        chart = {}
        back={}
        T= self.T
        n= len(words)
        
        for i in range(n):
            chart[i-1,i] = O[i-1]
        #for X in self.vocab:
          #if '<EOS>' !=  X:
        #fnum= self.vocab_index.get(X)
        #chart[n-1,n]= 
        chart[n-1,n] = torch.tensor([- float('inf')]*len(self.vocab_index))
        fnum1= self.vocab_index.get('<EOS>')
        chart[n-1,n][fnum1]=O[n-1,fnum1]
        for i in range(n-2,-1,-1):
              
              chart[i,n]= (T + chart[i,i+1].unsqueeze(-1) + chart[i+1,n].unsqueeze(0)).max(dim=1).values
              back[i,n] = torch.max(T + chart[i,i+1].unsqueeze(-1) + chart[i+1,n].unsqueeze(0), dim=1).indices
        
        snum= self.vocab_index.get('<BOS>')
        #print(snum)
        tree=[]
        current=snum
        sentence=[]
        for i in range(n-2):
          #print(i)
          tree.append(self.vocab[current])
          sentence.append(words[i] + ':' + self.vocab[current])
          current= back[i,n][current]
        tree.append(self.vocab[current])
        sentence.append(words[n-2] + ':' + self.vocab[current])  
        tree.append('<EOS>')
        sentence.append(words[n-1] + ':' + '<EOS>') 
        #sentence.reverse() 
        #tree.reverse()
        return sentence, tree
       
    def CRF(self, words):
        
        for i,word in enumerate(words):
          if word not in self.rules:
            words[i]='<unk>'
            
        O = self.encode(words)
        
        chart = {}
        T= self.T
        n= len(words)
        
        for i in range(n):
            chart[i-1,i] = O[i-1]
        #for X in self.vocab:
          #if '<EOS>' !=  X:
        #fnum= self.vocab_index.get(X)
        #chart[n-1,n]= 
        chart[n-1,n] = torch.tensor([- float('inf')]*len(self.vocab_index))
        fnum1= self.vocab_index.get('<EOS>')
        chart[n-1,n][fnum1]=O[n-1,fnum1]
        
        for i in range(n-2,-1,-1):
              chart[i,n]= torch.logsumexp(T + chart[i,i+1].unsqueeze(-1) + chart[i+1,n].unsqueeze(0), dim=1)
        
        snum= self.vocab_index.get('<BOS>')
        return chart[0,n][snum]
              

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', type=str)
    parser.add_argument('--dev', dest='dev', type=str)
    parser.add_argument('--load', dest='load', type=str)
    parser.add_argument('--save', dest='save', type=str)
    parser.add_argument('infile', nargs='?', type=str, help='test data to parse')
    parser.add_argument('-o', '--outfile', type=str, help='write parses to file')
    args = parser.parse_args()
    
    if args.train:
        # Read training data and create vocabularies
        #traind = read_parallel(args.train)
        #traindata= read_labels(traind)
        traindata = read_labels(fileinput.input(args.train))
        word=set()
        label=set()
        for line in traindata:
          (wor,lab)=line
          for l in lab:
            if l not in label:
                label.add(l)
          for w in wor:
            if w not in word:
                word.add(w)
        #label.add('E')
        #label.add('S')
                
        m = Model(word, label, 200) # try increasing 64 to 128 or 256
        
        if args.dev is None:
            print('error: --dev is required', file=sys.stderr)
            sys.exit()
        devdata = read_labels(fileinput.input(args.dev))
            
    elif args.load:
        if args.save:
            print('error: --save can only be used with --train', file=sys.stderr)
            sys.exit()
        if args.dev:
            print('error: --dev can only be used with --train', file=sys.stderr)
            sys.exit()
        m = torch.load(args.load)

    else:
        print('error: either --train or --load is required', file=sys.stderr)
        sys.exit()

    if args.infile and not args.outfile:
        print('error: -o is required', file=sys.stderr)
        sys.exit()

    if args.train:
        o = torch.optim.Adam(m.parameters(), lr=0.0001)

        prev_dev_loss = best_dev_loss = None

        for epoch in range(10):
            train_loss = 0
            random.shuffle(traindata)
            m.train()
            for words, labels in progress(traindata):
                tree_score = m.score_tree(words, labels)
                z = m.CRF(words)
                #print(z)
                loss = -(tree_score-z)
                
                o.zero_grad()
                loss.backward()
                o.step()
                train_loss += loss.item()
                sen,x=m.labeler(words)
            print(train_loss)
            dev_loss = 0
            dev_failed = 0
            m.eval()
            predict=[]
            correct=[]
            for line_num, (words, labels) in enumerate(devdata):
                tree_score = m.score_tree(words, labels)
                z = m.CRF(words)
                loss = -(tree_score-z)
                correct.append(labels)
                _,label= m.labeler(words)
                predict.append(label)
                dev_loss += loss.item()
                #print(' '.join(sen))
            f1= compute_f1(predict,correct)
            print('F1 Score')
            print(f1)
                
            if best_dev_loss is None or dev_loss < best_dev_loss:
                best_model = copy.deepcopy(m)
                best_dev_loss = dev_loss
                print('saving new best model', file=sys.stderr)
                if args.save:
                    torch.save(m, args.save)
            prev_dev_loss = dev_loss

            #print(f'train_loss={train_loss} dev_loss={dev_loss} dev_failed={dev_failed}', file=sys.stderr)
            
        m = best_model

    if args.infile:
        with open(args.outfile, 'w') as outfile:
            for line in open(args.infile):
                words = line.split()
                t, label = m.labeler(words)
                #print(label)
                key=[]
                value=[]
                #aa=label.split()
                for i,part in enumerate(label):
                  index= part.find('B-')
                  if (index==0):
                    key.append(part[index+2:])
                    value.append(words[i])
                    
                dict1 = dict( zip(key,value) )
                print(m)
                #print(dict1) 
                '''   
                if t is not None:
                    print(' '.join(t), file=outfile)
                else:
                    print(file=outfile)
                '''