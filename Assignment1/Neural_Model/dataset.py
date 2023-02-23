import torch
import spacy
import pandas as pd
import numpy as np
from collections import Counter
from Token import Clean
from Token import Tokenise,paddingString
from nltk.util import ngrams
from torch.utils.data import DataLoader
import argparse

filePath = '../DATA/Ulysses - James Joyce.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_data, batch_size, min_freq=5):
        self.data = train_data
        self.max_len = 20
        self.min_freq = min_freq
        self.vocab = []
        self.batch_size = batch_size
        self.ngramList = []
        sents = self.loadingWords()
        self.wordToIndex = {w: i for i, w in enumerate(self.vocab)}
        self.indexToWord = {i: w for i, w in enumerate(self.vocab)}
        self.padIndex = self.wordToIndex['<PAD>']
        self.unKnownIndex = self.wordToIndex['<UNK>']
        self.startIndex = self.wordToIndex['<START>']
        self.endIndex = self.wordToIndex['<END>']
        for sent in sents:
            tokens = sent
            prefix_seqs = []
            try:
                pfx = [tokens[0]]
                for token in tokens[1:]:
                    pfx.append(token)
                    prefix_seqs.append(pfx.copy())
                for i in range(len(prefix_seqs)):
                    currSeq = [self.wordToIndex.get(w,self.unKnownIndex) for w in prefix_seqs[i]]
                    pref_sq = [self.startIndex]+[self.padIndex]*(self.max_len-len(currSeq)) + [w for w in currSeq[1:]]
                    
                    self.ngramList.append(list(pref_sq))
            except IndexError:
                continue

                
    def loadingWords(self):
        text = [line for line in self.data if line.strip()]
        sentences = []
        wordFreq = {}
        mx = 0
        for line in text:
            tokens = Tokenise(line)
            tokens = ['<START>'] + tokens
            sentences.append(tokens)
            self.vocab += tokens
            mx = max(mx, len(tokens))
            for token in tokens:
                if token in wordFreq:
                    wordFreq[token] += 1
                else:
                    wordFreq[token] = 1

        # wordCount = Counter(wordFreq)
        wordCount = {}
        self.vocab = list(filter(lambda w: wordFreq[w] >= self.min_freq, self.vocab))
        self.vocab = ['<PAD>', '<UNK>','<END>'] + self.vocab
        self.vocab = set(self.vocab)
        self.vocabSize = len(self.vocab)
        print(self.vocabSize)
        self.max_len = max(mx,self.max_len)
        # print(sentences)
        return sentences

