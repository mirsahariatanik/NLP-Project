import torch
device = 'cpu'
import math, collections.abc, random, copy
import numpy 
from seqlabel import *
import sys, fileinput
from layers import *
from frames import Frame
#from model2 import Vocab, read_parallel, read_mono, progress
from tokenizer import tokenize

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
'''
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
        self.emb = Embedding(len(self.rule_index), dims)
        self.rnn1 = RNN(dims)
        self.rnn2 = RNN(dims)

        # Parameters for rule model
        #self.softmax= layers.SoftmaxLayer(dims, len(self.vocab_index))
        self.linear = LinearLayer(dims, len(self.vocab_index))
        self.T= torch.nn.Parameter(torch.randn(len(self.vocab_index), len(self.vocab_index), requires_grad=True))
        
    def encode(self, words):
        
        # numberize words
        #eos = self.rule_index['<EOS>']
        #bos = self.rule_index['<BOS>']
        unk = self.rule_index['<unk>']
        nums = torch.tensor([self.rule_index.get(w, unk) for w in words])

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
        

    def label(self, words):
    
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
'''
class Model(torch.nn.Module):
    """Neural parsing model."""
    def __init__(self, rules, vocab, dims):
        super().__init__()

        # Word and mapping to numbers
        self.rules = rules
        self.rule_index = {r:i for (i,r) in enumerate(rules)}

        # Labels and mapping to numbers
        self.vocab = vocab
        #print(self.vocab)
        self.vocab_index = {w:i for (i,w) in enumerate(self.vocab)}
        #print(len(self.vocab_index))
        
        # Parameters for encoder
        self.emb = Embedding(len(self.rule_index), dims)
        self.pos = torch.nn.Parameter(torch.empty(100, dims))
        torch.nn.init.normal_(self.pos, std=0.01)
        self.att1 = SelfAttention(dims)
        self.ffnn1 = TanhLayer(dims, dims, True)
        self.att2 = SelfAttention(dims)
        self.ffnn2 = TanhLayer(dims, dims, True)
        self.out = SoftmaxLayer(dims, len(self.vocab_index))
        self.rnn1 = RNN(dims)
        self.rnn2 = RNN(dims)
        self.linear = LinearLayer(dims, len(self.vocab_index))
        self.T= torch.nn.Parameter(torch.randn(len(self.vocab_index), len(self.vocab_index), requires_grad=True))

    def encode(self, words):
        
        #unk = self.rule_index['unk']
        nums = torch.tensor([self.rule_index.get(w) for w in words])

        e = self.emb(nums) + self.pos[:len(nums)]
        h = self.att1(e)
        h = self.ffnn1(h)
        h = self.att2(h)
        h = self.ffnn2(h)
        y= self.out(h[0])
        return y
        
    def logprob(self, words, labels):
        '''
        for i,word in enumerate(words):
          if word not in self.rules:
            words[i]='unk'
        '''
        y = self.encode(words)
        #print('y size')
        #print(y.size())
        #loss=torch.nn.NLLLoss(reduction='sum')
        fnum= torch.tensor(self.vocab_index.get(labels[0]))
        logprob = y[fnum]
        return logprob
        

    def labeler(self, words):
        '''
        for i,word in enumerate(words):
          if word not in self.rules:
            words[i]='unk'
        '''
        y = self.encode(words)
        #sentence = []
        #label =[]
        key=list(self.vocab_index.keys())
        enum = torch.argmax(y).item()
        wor = key[enum]
        #sentence= words + '\t' + wor
        label= wor
        return label

    def encode1(self, words):
        
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
        '''
        for i,word in enumerate(words):
          if word not in self.rules:
            words[i]='<unk>'
        '''    
        O = self.encode1(words)
        
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
        

    def label(self, words):
        '''
        for i,word in enumerate(words):
          if word not in self.rules:
            words[i]='<unk>'
        '''    
        O = self.encode1(words)
        
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
            
        O = self.encode1(words)
        
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
    import argparse, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='training data')
    parser.add_argument('--data4', type=str, help='training data')
    parser.add_argument('--dev', type=str, help='development data')
    parser.add_argument('infile', nargs='?', type=str, help='test data to translate')
    parser.add_argument('-o', '--outfile', type=str, help='write translations to file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--save', type=str, help='save model in file')
    args = parser.parse_args()


            
    if args.load:
        if args.save:
            print('error: --save can only be used with --train', file=sys.stderr)
            sys.exit()
        if args.dev:
            print('error: --dev can only be used with --train', file=sys.stderr)
            sys.exit()
        m = torch.load(args.load)
        m1= torch.load('./modproject1.pth')
        
        
        #m1=torch.load('model3_hw4.pth')

    else:
        print('error: either --train or --load is required', file=sys.stderr)
        sys.exit()

    if args.infile and not args.outfile:
        print('error: -o is required', file=sys.stderr)
        sys.exit()

    
    if args.infile:
        with open(args.outfile, 'w') as outfile:
            for line in open(args.infile):
                line1= ['CLS'] + line.split()
                line1 = ' '.join(line1)
                words = tokenize(line1)
                word=[]
                #print(m1)
                label = m.labeler(words)
                line2=line.split()
                line2=' '.join(line2)
                lin = 'S'+' '+label + ' '+line2+' E'
                #print(lin)
                wordn= ' '.join(lin)
                #print(wordn)
                lin1=wordn.split()
                bio= lin.split()
                #print('bio', bio)
                #print(m)
                sen, tree = m1.label(bio)
                
                #print(sen)
                #print(tree)
                key=[]
                value=[]
                value1=[]
                #print(len(value))
                #aa=label.split()
                for i,part in enumerate(tree):
                  index= part.find('B-')
                  index1= part.find('I-')
                  if (index==0):
                    key.append(part[index+2:])
                    value.append(bio[i])
                    value1=value
                  if (index1==0):
                    l=len(value)
                    if (l==0):
                      #val=value[l]+' '+words[i]
                      value1.append(bio[i])
                      #value1=value
                      
                    else:
                      val=value[l-1]+' '+bio[i]
                      #print(val)
                      value1[l-1]=val
                  #else: 
                    #value1= value
                    #value[0]=value[0]+' ' 
                dict1 = dict( zip(key,value1) )
                print(Frame(label, dict1), file=outfile)
                
                #print(dict1)
                '''
                if label is not None:
                    print(bio, file=outfile)
                else:
                    print(file=outfile)
                '''
