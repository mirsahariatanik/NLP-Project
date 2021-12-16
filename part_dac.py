import torch
device = 'cpu'

import math, collections.abc, random, copy

from layers import *
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
        
        for i,word in enumerate(words):
          if word not in self.rules:
            words[i]='unk'
        y = self.encode(words)
        fnum= torch.tensor(self.vocab_index.get(labels[0]))
        logprob = y[fnum]
        return logprob
        

    def labeler(self, words):
    
        for i,word in enumerate(words):
          if word not in self.rules:
            words[i]='unk'
        y = self.encode(words)
        key=list(self.vocab_index.keys())
        enum = torch.argmax(y).item()
        wor = key[enum]
        label= wor
        return label
        
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
        
        data=[]
        for line in open(args.train):
          fline, eline = line.split('\t')
          fwords = tokenize(fline)
          ewords = eline.split()
          data.append((fwords, ewords))
       
        traindata = data
        #print(data)
        word=[]
        label=[]
        for line in traindata:
          (wor,lab)=line
          
          for l in lab:
            if l not in label:
              label.append(l)
          for w in wor:
            if w not in word:
              word.append(w)
                
        m = Model(word, label, 200) # try increasing 64 to 128 or 256
        
        if args.dev is None:
            print('error: --dev is required', file=sys.stderr)
            sys.exit()
        data1=[]
        for line in open(args.dev):
          a= ' '
          fline, eline = line.split('\t')
          fline = ['CLS'] + fline.split()
          a1=a.join(list(fline))
          fwords = tokenize(a1)
          ewords = eline.split()
          data1.append((fwords, ewords))
        devdata=data1
            
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
        for epoch in range(1):
            random.shuffle(traindata)

            ### Update model on train

            train_loss = 0.
            for fwords, ewords in progress(traindata):
                #print(fwords)
                #print(ewords)
                loss = -m.logprob(fwords, ewords)
                #print(loss)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            #print(train_loss)   

            ### Validate on dev set and print out a few translations
            
            dev_loss = 0.
            dev_ewords = 0
            predict=[]
            correct=[]
            total_types = match_types = 0
            for line_num, (fwords, ewords) in enumerate(devdata):
                dev_loss -= m.logprob(fwords, ewords).item()
                #dev_ewords += len(ewords) # includes EOS
                #correct=ewords[0]
                label= m.labeler(fwords)
                
                #predict.append(label)
                if label == ewords[0]:
                    match_types += 1
                total_types += 1
            
            print('accuracy:', match_types/total_types)
            #f1= compute_f1(predict,correct)
            #print('F1 Score')
            #print(f1)
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
                data1=[]
                #for line in open(args.dev):
                a= ' '
                fline = ['CLS'] + line.split()
                a1=a.join(list(fline))
                fwords = tokenize(a1)
                #data1.append(fwords)
                #devdata=data1
                words = fwords
                label = m.labeler(words)
                if label is not None:
                    print(label, file=outfile)
                else:
                    print(file=outfile)
