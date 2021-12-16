
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

        # Word and mapping to numbers
        self.rules = rules
        self.rule_index = {r:i for (i,r) in enumerate(rules)}

        # Labels and mapping to numbers
        self.vocab = vocab
        #print(self.vocab)
        self.vocab_index = {w:i for (i,w) in enumerate(self.vocab)}
        #print(self.vocab_index)
        
        # Parameters for encoder
        self.emb = layers.Embedding(len(self.rule_index), dims)
        self.rnn1 = layers.RNN(dims)
        self.rnn2 = layers.RNN(dims)
        self.softmax= layers.SoftmaxLayer(dims, len(self.vocab_index))

    def encode(self, words):
        
        unk = self.rule_index['<unk>']
        nums = torch.tensor([self.rule_index.get(w, unk) for w in words])

        # look up word embeddings
        V = self.emb(nums)
        # run RNN
        #Vt= torch.transpose(V,0,1)
        #print(Vt.size())
        G = self.rnn1.sequence(V)
        H = self.rnn2.sequence(G)
        y= self.softmax(H)
        #print('y size')
        #print(y.size())
        return y
        
    def logprob(self, words, labels):
        
        for i,word in enumerate(words):
          if word not in self.rules:
            words[i]='<unk>'
        y = self.encode(words)
        loss=torch.nn.NLLLoss(reduction='sum')
        fnum= torch.tensor([self.vocab_index.get(l) for l in labels])
        logprob = loss(y, fnum)
        return logprob
        

    def labeler(self, words):
    
        for i,word in enumerate(words):
          if word not in self.rules:
            words[i]='<unk>'
        y = self.encode(words)
        sentence = []
        label =[]
        key=list(self.vocab_index.keys())
        for i,eword in enumerate(words):
            enum = torch.argmax(y[i]).item()
            #print(enum)
            wor = key[enum]
            #print(ewords)
            sentence.append(eword + ':' + wor)
            label.append(wor)
        return sentence, label
        
if __name__ == "__main__":
    import argparse, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='training data')
    parser.add_argument('--dev', type=str, help='development data')
    parser.add_argument('infile', nargs='?', type=str, help='test data to translate')
    parser.add_argument('-o', '--outfile', type=str, help='write translations to file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--save', type=str, help='save model in file')
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
        #print('length')
        #print(len(word))
        #print(len(label))
                
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
        opt = torch.optim.Adam(m.parameters(), lr=0.0001)

        best_dev_loss = None
        for epoch in range(10):
            random.shuffle(traindata)

            ### Update model on train

            train_loss = 0.
            for fwords, ewords in progress(traindata):
                #print(fwords)
                #print(ewords)
                loss = m.logprob(fwords, ewords)
                #print(loss)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
                

            ### Validate on dev set and print out a few translations
            
            dev_loss = 0.
            dev_ewords = 0
            predict=[]
            correct=[]
            for line_num, (fwords, ewords) in enumerate(devdata):
                dev_loss += m.logprob(fwords, ewords).item()
                #dev_ewords += len(ewords) # includes EOS
                correct.append(ewords)
                sen, label= m.labeler(fwords)
                predict.append(label)
                print(' '.join(sen))
            f1= compute_f1(predict,correct)
            print('F1 Score')
            print(f1)
            if best_dev_loss is None or dev_loss < best_dev_loss:
                best_model = copy.deepcopy(m)
                if args.save:
                    torch.save(m, args.save)
                best_dev_loss = dev_loss

            #print(f'[{epoch+1}] train_loss={train_loss} train_ppl={math.exp(train_loss/train_ewords)} dev_ppl={math.exp(dev_loss/dev_ewords)}', flush=True)
            
        m = best_model

    ### Translate test set

    if args.infile:
        with open(args.outfile, 'w') as outfile:
            for line in open(args.infile):
                words = line.split()
                t, label = m.labeler(words)
                if t is not None:
                    print(' '.join(t), file=outfile)
                else:
                    print(file=outfile)
