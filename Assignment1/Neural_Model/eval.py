import torch
import torch.nn as nn
import torch
import spacy
import pandas as pd
import numpy as np
from collections import Counter
from Token import Clean
from Token import Tokenise, paddingString
from nltk.util import ngrams
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import sys
import math
from Model import LSTMmodel

from dataset import Dataset
filePath = '../DATA/Pride and Prejudice - Jane Austen.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

def splitData(corpus, train_ratio, valid_ratio, test_ratio):
    with open(corpus, 'r') as f:
        text = f.readlines()
    text = [line.strip() for line in text if line.strip()]
    train_size = int(len(text) * train_ratio)
    valid_size = int(len(text) * valid_ratio)
    test_size = int(len(text) * test_ratio)
    train_data = text[:train_size]
    valid_data = text[train_size:train_size + valid_size]
    test_data = text[train_size + valid_size:]
    return train_data, valid_data, test_data


train_data, valid_data, test_data = splitData(filePath, 0.7, 0.15, 0.15)
BATCH_SIZE = 256
TRAINSET = Dataset(train_data, BATCH_SIZE)
VLAIDSET = Dataset(valid_data, BATCH_SIZE)
TESTSET = Dataset(test_data, BATCH_SIZE)


def generateBatch(dataset):
    input_ngram, trg = [], []
    for tg in dataset:
        input_ngram.append(tg[:-1])
        trg.append(tg[-1])
    return torch.tensor(input_ngram, dtype=torch.long), torch.tensor(trg, dtype=torch.long)


train_loader = DataLoader(
    TRAINSET.ngramList, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generateBatch)
valid_loader = DataLoader(
    VLAIDSET.ngramList, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generateBatch)
test_loader = DataLoader(
    TESTSET.ngramList, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generateBatch)


class Evaluation:
    def __init__(self, model: nn.Module, epochs, datasetTrain: torch.utils.data.DataLoader, datasetValid: torch.utils.data.DataLoader, datasetTest: torch.utils.data.DataLoader):
        self.model = model
        self.datasetTrain = datasetTrain
        self.datasetValid = datasetValid
        self.datasetTest = datasetTest
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.clip = 1
        self.patience = 10
        self.learning_rate = 0.005
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=6, gamma=0.1, last_epoch=-1, verbose=False)

    def train(self):
        maxValidLoss = math.inf
        ctr = 0

        def trainModel():
            epochAcc = 0
            epochLoss = 0
            self.model.train()
            # hidden = self.model.init_hidden(24)
            for i, (x, y) in enumerate(tqdm(self.datasetTrain)):
                x = x.to(device)
                y = y.to(device)
                self.optimizer.zero_grad()
                outputs, hidden = self.model(x)
                y = y.view(-1)
                loss = self.criterion(outputs, y)
                loss.backward()

                epochAcc += 100*(outputs.argmax(dim=1) ==
                                 y).sum().item()/y.shape[0]
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip)
                epochLoss += loss.item()
                self.optimizer.step()
                if i % 100 == 0:
                    print(
                        f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")

            print(
                f"Epoch: {epoch}, Loss: {epochLoss/len(self.datasetTrain)}, Accuracy: {epochAcc/len(self.datasetTrain)}")

        def validate():
            self.model.eval()
            epochAcc = 0
            epochLoss = 0
            with torch.no_grad():
                for i, (x, y) in enumerate(tqdm(self.datasetValid)):
                    x = x.to(device)
                    y = y.to(device)
                    outputs, hidden = self.model(x)
                    y = y.view(-1)
                    loss = self.criterion(outputs, y)
                    epochAcc += 100*(outputs.argmax(dim=1) == y).float().mean()
                    epochLoss += loss.item()

            print(
                f"Validation Loss: {epochLoss/len(self.datasetValid)}, Validation Accuracy: {epochAcc/len(self.datasetValid)}")
            return epochLoss/len(self.datasetValid)

        for epoch in range(self.epochs):
            trainModel()
            valid_loss = validate()

            #valid_loss = self.validate()
            self.scheduler.step()
            if valid_loss < maxValidLoss:
                maxValidLoss = valid_loss
                torch.save(self.model.state_dict(), 'model1.pt')
                print("Model saved")
                ctr = 0
            else:
                ctr += 1
                print(f"Validation loss not improved for {ctr} epochs")
            if ctr > self.patience:
                print("Early stopping")
                break

    def test(self):
        self.model.eval()
        epochAcc = 0
        epochLoss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(self.datasetTest)):
                x = x.to(device)
                y = y.to(device)
                outputs, hidden = self.model(x)
                y = y.view(-1)
                loss = self.criterion(outputs, y)
                epochAcc += 100*(outputs.argmax(dim=1) ==
                                 y).sum().item()/y.shape[0]
                epochLoss += loss.item()

        print(
            f"Test Loss: {epochLoss/len(self.datasetTest)}, Test Accuracy: {epochAcc/len(self.datasetTest)}")
        return epochLoss/len(self.datasetTest)


globalVocab = TRAINSET.vocab
globalWordToIndex = TRAINSET.wordToIndex
PAD_INDEX = TRAINSET.padIndex
START_INDEX = TRAINSET.startIndex
UNK_INDEX = TRAINSET.unKnownIndex
MAX_LEN = TRAINSET.max_len


def getProbPerplexity(model, dataset):
    model.eval()
    perplexity_list = []
    with torch.no_grad():
        for line in dataset.data:
            perplexity = perpForSentence(model, line)
            #print(perplexity)
            if perplexity != -1:
                perplexity_list.append(
                    {'line': line, 'perplexity': perplexity})

    # averagePerplexty
    avgPerplexity = sum([line['perplexity']
                        for line in perplexity_list])/len(perplexity_list)
    return perplexity_list, avgPerplexity


def writeToFile(filePath, perplexity_list, avg):
    with open(filePath, 'w') as f:
        f.write(f"Average Perplexity: {avg}\n")
        for line in perplexity_list:
            f.write(f"{line['line']}\t {line['perplexity']}\n")


def perpForSentence(model, sentence):
    model.eval()
    with torch.no_grad():
        prob_gram = 1
        tokens = Tokenise(sentence)
        tokens = ['<START>'] + tokens

        if len(tokens) == 0 or len(tokens) == 1:
            return -1

        elif len(tokens) > 1:
            prefix_seqs = []
            gramList = []
            try:
                pfx = [tokens[0]]
                for token in tokens[1:]:
                    pfx.append(token)
                    prefix_seqs.append(pfx.copy())
                for i in range(len(prefix_seqs)):
                    currSeq = [globalWordToIndex.get(
                        w, UNK_INDEX) for w in prefix_seqs[i]]
                    pref_sq = [START_INDEX] + [PAD_INDEX] * \
                        (MAX_LEN-len(currSeq)) + [w for w in currSeq[1:]]
                    gramList.append(list(pref_sq))
            except IndexError:
                return -1

            if len(gramList) > 0:
                for gram in gramList:
                    input_gram = torch.tensor(
                        gram[:-1], dtype=torch.long).to(device)
                    output_gram = gram[-1]
                    output, hidden = model(input_gram.unsqueeze(dim=0))
                    output = torch.exp(output.view(-1))

                    prob_gram = prob_gram * output[output_gram].cpu().numpy()

                perplexity = (1/prob_gram)**(1/len(gramList))
                return perplexity
            else:
                return -1


###############################################
VOCAB_SIZE = TRAINSET.vocabSize
EMBEDDING_DIM = 512
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROP_OUT = 0.5


lngMOD = LSTMmodel(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, VOCAB_SIZE, DROP_OUT)
eval = Evaluation(lngMOD, 20, train_loader, valid_loader, test_loader)
# eval.train()

# perplexity_list, avgPerplexity = getProbPerplexity(lngMOD, TESTSET)
# path = 'test1_perplexity.txt'
# writeToFile(path, perplexity_list, avgPerplexity)

import sys
args = sys.argv
modelPath = args[1]
lngMOD.load_state_dict(torch.load('../Models/model1.pt'))
inputsentence = input("Enter the sentence: ")
print(perpForSentence(lngMOD, inputsentence))
