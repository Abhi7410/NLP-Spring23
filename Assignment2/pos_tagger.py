from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import conllu


# Parse the data using the CoNLL-U format parser
train_data = conllu.parse(
    open("./Dataset/en_atis-ud-train.conllu", "r", encoding="utf-8").read())
valid_data = conllu.parse(
    open("./Dataset/en_atis-ud-dev.conllu", "r", encoding="utf-8").read())
test_data = conllu.parse(
    open("./Dataset/en_atis-ud-test.conllu", "r", encoding="utf-8").read())


train_data = [[(word['form'], word['upostag'])
               for word in sentence] for sentence in train_data]
valid_data = [[(word['form'], word['upostag'])
               for word in sentence] for sentence in valid_data]
test_data = [[(word['form'], word['upostag'])
              for word in sentence] for sentence in test_data]


# Glove Embedding


def load_glove_embeddings(file_path):
    with open(file_path, 'r') as f:
        embeddings = {}
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


# embeddings = load_glove_embeddings('./glove.6B/glove.6B.100d.txt')
# embeddings_list = [embeddings[word] for word in embeddings]
# embeddings_matrix = np.array(embeddings_list)
# add for PAD
# embeddings_matrix = np.vstack(
    # (np.zeros(100).astype(np.float32), embeddings_matrix))
# add for UNK
# embeddins_matrix = np.vstack(
    # (embeddings_matrix, np.random.rand(100).astype(np.float32)))

# np.save('embeddings_matrix.npy', embeddings_matrix)


# Create a dictionary of words and their corresponding indices
def dictWCT(data):
    wordCount = {}
    tagCount = {}
    wordFreq = {}
    for sentence in data:
        for word in sentence:
            if word[0] not in wordCount:
                wordCount[word[0]] = len(wordCount)+1
                wordFreq[word[0]] = 1
            else:
                wordFreq[word[0]] += 1
            if word[1] not in tagCount:
                tagCount[word[1]] = len(tagCount)+1
    return wordCount, tagCount, wordFreq


wordToIdx, tagToIdx, wordFreq = dictWCT(train_data)

wordToIdx['<PAD>'] = 0
wordFreq = {k: v for k, v in wordFreq.items() if v > 2}
wordToIdx['<UNK>'] = len(wordFreq)+1
tagToIdx['<UNK>'] = len(tagToIdx)+1
numUniqueWords = len(wordFreq)
numUniqueTags = len(tagToIdx)+1
idxToWord = {v: k for k, v in wordToIdx.items()}
idxToTag = {v: k for k, v in tagToIdx.items()}

# print("Number of unique words: ", numUniqueWords)
# print("Number of unique tags: ", numUniqueTags)


def getIndices(data, wordToIdx, tagToIdx, wordFreq, min_freq=2):
    wordIndices = []
    tagIndices = []
    mxTagForUnk = {}
    for sentence in data:
        wordIdx = []
        tagIdx = []
        for word in sentence:
            if word[0] not in wordFreq:
                wordIdx.append(wordToIdx['<UNK>'])
                tagIdx.append(tagToIdx['<UNK>'])
                if word[1] not in mxTagForUnk:
                    mxTagForUnk[word[1]] = 1
                else:
                    mxTagForUnk[word[1]] += 1
            else:
                wordIdx.append(wordToIdx[word[0]])
                tagIdx.append(tagToIdx[word[1]])
        wordIndices.append(wordIdx)
        tagIndices.append(tagIdx)

    mxTagForUnk = max(mxTagForUnk, key=mxTagForUnk.get)
    # tagIndices = [[tagToIdx[mxTagForUnk] if word == wordToIdx['<UNK>'] else tag for word, tag in zip(
    #     wordIndices[i], tagIndices[i])] for i in range(len(wordIndices))]
    return wordIndices, tagIndices


wordIndices, tagIndices = getIndices(train_data, wordToIdx, tagToIdx, wordFreq)
wordIndicesValid, tagIndicesValid = getIndices(
    valid_data, wordToIdx, tagToIdx, wordFreq)
wordIndicesTest, tagIndicesTest = getIndices(
    test_data, wordToIdx, tagToIdx, wordFreq)


def padWCT(wordIndices, tagIndices):
    max_len = 0
    for i in range(len(wordIndices)):
        if len(wordIndices[i]) > max_len:
            max_len = len(wordIndices[i])

    wordIndices = np.array([np.pad(wordIndices[i], (0, max_len -
                           len(wordIndices[i])), 'constant') for i in range(len(wordIndices))])
    tagIndices = np.array([np.pad(tagIndices[i], (0, max_len -
                          len(tagIndices[i])), 'constant') for i in range(len(tagIndices))])

    return wordIndices, tagIndices


wordIndices, tagIndices = padWCT(wordIndices, tagIndices)
wordIndicesValid, tagIndicesValid = padWCT(wordIndicesValid, tagIndicesValid)
wordIndicesTest, tagIndicesTest = padWCT(wordIndicesTest, tagIndicesTest)

# GLOBAL VARIABLES
WORD_EMBEDDING_DIM = 100
LSTM_HIDDEN_DIM_WORD = 256
DROPOUT_RATE = 0.5
BATCH_SIZE = 128
NUM_EPOCHS = 10


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# print(device)


# Create the model


class BiLSTM(nn.Module):
    def __init__(self, numUniqueWords, numUniqueTags, wordEmbeddingDim, lstmHiddenDimWord, dropoutRate, embedding):
        super(BiLSTM, self).__init__()
        self.wordEmbedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding).float(), freeze=False, padding_idx=0)
        self.lstmWord = nn.LSTM(wordEmbeddingDim,
                                lstmHiddenDimWord, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropoutRate)
        self.linear = nn.Linear(2*lstmHiddenDimWord, numUniqueTags)

    def forward(self, word_indices):
        word_embeddings = self.wordEmbedding(word_indices)
        lstm_word_out = self.lstmWord(word_embeddings)[0]
        lstm_word_out = self.dropout(lstm_word_out)
        linear_out = self.linear(lstm_word_out)
        tag_scores = F.log_softmax(linear_out, dim=2)

        return tag_scores

embeddings_matrix = np.load('embeddings_matrix.npy')
model = BiLSTM(numUniqueWords, numUniqueTags, WORD_EMBEDDING_DIM,
               LSTM_HIDDEN_DIM_WORD, DROPOUT_RATE, embeddings_matrix)
if torch.cuda.is_available():
    model.cuda()


# create tensors 
def convertToTensors(wordIndices, tagIndices):
    wordTensor = [torch.LongTensor(word) for word in wordIndices]
    tagTensor = [torch.LongTensor(tag) for tag in tagIndices]
    return wordTensor, tagTensor


wordTensor, tagTensor = convertToTensors(wordIndices, tagIndices)
wordTensorValid, tagTensorValid = convertToTensors(
    wordIndicesValid, tagIndicesValid)
wordTensorTest, tagTensorTest = convertToTensors(
    wordIndicesTest, tagIndicesTest)


# training model
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss(ignore_index=0)
torch.autograd.set_detect_anomaly(True)


def train(model, optimizer, criterion, wordTensor, tagTensor):
    model.train()
    total_loss = 0
    tot_correct = 0
    total_pad = 0
    for i in range(0, len(wordIndices), BATCH_SIZE):
        optimizer.zero_grad()
        word_tensor = wordTensor[i:i+BATCH_SIZE]
        tag_tensor = tagTensor[i:i+BATCH_SIZE]
        word_tensor = torch.stack(word_tensor).to(device)
        tag_tensor = torch.stack(tag_tensor).to(device)
        tag_tensor = tag_tensor.view(-1)
        output = model(word_tensor)
        output = output.view(-1, output.shape[2])
        loss = criterion(output, tag_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        tot_correct += torch.sum((tag_tensor != 0) &
                                 (tag_tensor == pred)).item()

        total_pad += torch.sum(tag_tensor == 0).item()

    return total_loss, tot_correct/(len(wordIndices)*len(wordIndices[0]) - total_pad)


def evaluate(model, wordTensor, tagTensor):
    model.eval()
    total_loss = 0
    tot_correct = 0
    total_pad = 0
    wordTagList = []
    with torch.no_grad():
        for i in range(0, len(wordTensor), BATCH_SIZE):
            word_tensor = wordTensor[i:i+BATCH_SIZE]
            tag_tensor = tagTensor[i:i+BATCH_SIZE]
            word_tensor = torch.stack(word_tensor).to(device)
            tag_tensor = torch.stack(tag_tensor).to(device)
            tag_tensor = tag_tensor.view(-1)
            output = model(word_tensor)
            output = output.view(-1, output.shape[2])
            loss = criterion(output, tag_tensor)
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            tot_correct += torch.sum((tag_tensor != 0) &
                                     (tag_tensor == pred)).item()
            total_pad += torch.sum(tag_tensor == 0).item()
            # take those only which are not padded
            wordTagList.extend(
                list(zip(word_tensor.view(-1).tolist(), pred.view(-1).tolist())))
    return total_loss, tot_correct/(len(wordTensor)*len(wordTensor[0]) - total_pad), wordTagList


def EpochRun(model, optimizer, criterion, wordTensor, tagTensor, wordTensorValid, tagTensorValid):
    loss_list = []
    accuracy_list = []
    validaton_loss_list = []
    validaton_accuracy_list = []
    for epoch in range(NUM_EPOCHS):
        loss, totalCorrect = train(
            model, optimizer, criterion, wordTensor, tagTensor)
        lossValid, totalCorrectValid, _ = evaluate(
            model, wordTensorValid, tagTensorValid)
        print("Epoch: {0} \t Training Loss: {1} \t Training Accuracy: {2} \t Validation Loss: {3} \t Validation Accuracy: {4}".format(
            epoch+1, loss, totalCorrect, lossValid, totalCorrectValid))
        loss_list.append(loss)
        accuracy_list.append(totalCorrect)
        validaton_loss_list.append(lossValid)
        validaton_accuracy_list.append(totalCorrectValid)
    return loss_list, accuracy_list, validaton_loss_list, validaton_accuracy_list


# loss_list, accuracy_list, validaton_loss_list, validaton_accuracy_list = EpochRun(
#     model, optimizer, criterion, wordTensor, tagTensor, wordTensorValid, tagTensorValid)


def genSentence(model, wordFreq, sentence):
    sentence = sentence.lower()
    print(sentence)
    sentence = sentence.split(" ")
    word_indice = []
    for s in sentence:
        if s not in wordFreq or wordFreq[s] <= 2:
            word_indice.append(wordToIdx['<UNK>'])
        else:
            word_indice.append(wordToIdx[s])
    wordTen = torch.LongTensor(word_indice).to(device)
    wordTen = wordTen.view(1, -1)
    # predict the tag
    with torch.no_grad():
        model.eval()
        output = model(wordTen)
        _, predicted = torch.max(output, 2)
        # print(predicted.shape)
        for i in range(len(predicted[0])):
            print(sentence[i],"\t", idxToTag[predicted[0][i].item()])

# input sentence
import sys

input_sentence = input("Enter a sentence: ")

# load the model

model.load_state_dict(torch.load('modelFinal.pt'))
genSentence(model, wordFreq, input_sentence)
