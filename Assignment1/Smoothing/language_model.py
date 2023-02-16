import sys
import os
import math
from collections import defaultdict
import time
import string
import re

punctu = ['.', ',', '!', '?', ';', ':',
          '(', ')', '[', ']', '{', '}', '/', '\\', '|', '-', '_', '=', '+', '*', '&', '^', '%', '$', '#', '@', '~', '`', '\'', '\"']


def wordTokenizer(str):
    wrds = []
    for w in str.split():
        wrds.append(w)
    return wrds


def remove_punctuations(text):
    for p in punctu:

        text = text.replace(p, '')
    return text


def removeURL(text):
    return re.sub(r'https?://\S+|www\.\S+', "<URL>", text)


def removeHashtag(text):
    return re.sub(r'#[a-xA-Z]+', '<HASHTAG>', text)


def removeMentions(text):
    return re.sub(r'@\S+', '<MENTION>', text)


def removeEmail(text):
    return re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '<EMAIL>', text)


def removeNumber(text):
    return re.sub(r'[0-9]+(,([0-9]+))*(\.([0-9]+))?%?\s', '<NUMBER>', text)


def removeExtraSpace(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+', '', text)
    text = re.sub(r'\s+$', '', text)
    return text


def removeExtraLine(text):
    text = re.sub(r'\n+', '', text)
    return text


def Clean(str):
    str = str.lower()
    str = removeURL(str)
    str = removeHashtag(str)
    str = removeMentions(str)
    str = removeEmail(str)
    # str = removeNumber(str)
    str = remove_punctuations(str)
    str = removeExtraSpace(str)
    return str


def Token(str):
    str = str.lower()
    str = removeURL(str)
    str = removeHashtag(str)
    str = removeMentions(str)
    str = removeEmail(str)
    # str = removeNumber(str)
    str = remove_punctuations(str)
    str = removeExtraSpace(str)
    return wordTokenizer(str)


def paddedString(str, n):
    """
    pad the string with <s> and </s> for n gram
    """
    str = (['<s>']*(n-1)) + str + (['</s>']*(n-1))
    return str


def createFreqTable(trainFile, n):
    """
    create a freq dictionary for each n gram
    and create a vocab set for each n gram
    """
    # open the file and read the data
    with open(trainFile, 'r') as f:
        data = f.readlines()
    data = [line.strip() for line in data]
    uniWordsFreq = {}
    allWordsSentences = []
    for sentence in data:
        # print(sentence)
        words = Token(sentence)
        words = list(words)
        allWordsSentences.append(words)
        for word in words:
            if uniWordsFreq.get(word) is None:
                uniWordsFreq[word] = 1
            else:
                uniWordsFreq[word] += 1

    uniWordsVocab = set(uniWordsFreq.keys())
    uniWordsVocab = uniWordsVocab.difference(
        word for word in uniWordsVocab if uniWordsFreq[word] <= 4)
    uniWordsVocab.update({'<UNK>', '<s>', '</s>'})

    regenratedData = []
    for sentence in allWordsSentences:
        newSentence = []
        newSentence += [word if word in uniWordsVocab else '<UNK>' for word in sentence]
        regenratedData += [paddedString(newSentence, n)]

    # print(regenratedData)
    ngramFreq = [defaultdict(lambda: 0) for i in range(n)]
    # ngramFreq = [{} for i in range(n)]
    # for sent in regenratedData:

    for sent in regenratedData:
        for ind in range(n-1, len(sent)):
            ngram = (sent[ind],)
            # print(ngram)
            for nval in range(1, n+1):
                ngramFreq[nval-1][ngram] += 1
                ngram = (sent[ind-nval],) + ngram

    return ngramFreq, uniWordsVocab

# data = open('../DATA/train.txt', 'r')


# a, b = createFreqTable("../DATA/train.txt", 4)
# # print(a)


class KneserNey:
    """
       Formula :
       Absolute discounting
       Backoff
       Interpolation
    """

    def __init__(self, vocab, ngramFreq, n, d=1):
        """
        Initialize the Kneser-Ney language model.

        Parameters:
        - vocab: A set of words in the vocabulary.
        - ngram_frequencies: A list of dictionaries containing the frequencies of each n-gram for n from 1 to n.
        - n: The order of the model (n-gram).
        - d: The discounting factor (default is 0.75).
        """
        self.vocab = vocab
        self.ngram_frequencies = ngramFreq
        ngramItems = [freq.items() for freq in ngramFreq]
        self.ngram_items = ngramItems
        self.n = n
        self.d = d
        self.dpProb = {}

    def absDiscounting(self, count, countTotal):
        """
        Absolute discounting formula
        """
        return max(0, count-self.d)/countTotal

    def lambdaCount(self, countTotal):
        """
        lambda count formula
        """
        return self.d/countTotal

    def getProbability(self, ind, sentence):
        """
        Get the probability of a indth word using (ind -n +1) to ind as context
        Parameters:
        - ind: The index of the word.
        - sentence: The sentence.
        - dpProb: A dictionary containing the probabilities of each n-gram for n from 1 to n.
        """
        currWord = sentence[ind]
        context = sentence[ind-self.n+1:ind]
        currWord = currWord if currWord in self.vocab else '<UNK>'
        context = [word if word in self.vocab else '<UNK>' for word in context]
        try:
            prob = self.dpProb[tuple(context + [currWord])]
        except KeyError:
            uppList = [befContext for befContext, freq in self.ngram_items[1]
                       if freq > 0 and befContext[1] == currWord]
            numerator = len(uppList)
            downList = [befContext for befContext,
                        freq in self.ngram_items[1] if freq > 0]
            denominator = len(downList)
            if(denominator == 0):
                prob = self.d/self.ngram_frequencies[0][('<UNK>',)]
            else:
                prob = numerator/denominator
                
            for i in range(2, self.n+1):
                oldContext = context[-i+1:]
                oldWord = tuple(oldContext+[currWord])
                freqofOldWord = self.ngram_frequencies[i-1][oldWord]
                probArray = [frequency for cont, frequency in self.ngram_items[i-1]
                             if frequency > 0 and list(cont[:-1]) == list(oldContext)]
                deno = sum(probArray)
                num = len(probArray)
                print("num : ", num)
                print("deno : ", deno)
                if deno == 0:
                    prob = self.d/self.ngram_frequencies[0][('<UNK>',)]
                else:
                    prob = self.absDiscounting(
                        freqofOldWord, deno) + self.lambdaCount(deno)*num*prob
            prob = math.log(prob)
            self.dpProb[tuple(context+[currWord])] = prob
        return prob
       

    def perplexity(self, sentence, isTokenized):
        """
        Calculate the perplexity of a sentence.

        Parameters:
        - sentence: The sentence.

        Returns:
        - The perplexity of the sentence.
        """
        if(isTokenized == 'not'):
            prevLength = len(sentence)
            sentence = paddedString(Token(sentence), self.n)
        else:
            prevLength = len(sentence)
            sentence = paddedString(sentence, self.n)
        logp = 0
        # print(len(sentence))
        for index in range(self.n-1, len(sentence)):
            lp = self.getProbability(index, sentence)
            logp += lp
        # print(logp)
        try:
            perpl = math.exp(logp*(-1/len(sentence)))
        except:
            perpl = float("NaN")
        return perpl


# witten bell smoothing
class WittenBell:
    def __init__(self, vocab, ngramFreq, n):
        """
        Initialize the Witten-Bell language model.

        Parameters:
        - vocab: A set of words in the vocabulary.
        - ngram_frequencies: A list of dictionaries containing the frequencies of each n-gram for n from 1 to n.
        - n: The order of the model (n-gram).
        """
        self.vocab = vocab
        self.ngram_frequencies = ngramFreq
        self.ngram_items = [freq.items() for freq in ngramFreq]
        self.n = n
        self.dpProb = {}

    def getProbability(self, ind, sentence):
        """
        Get the probability of a indth word using (ind -n +1) to ind as context
        Parameters:
        - ind: The index of the word.
        - sentence: The sentence.
        - dpProb: A dictionary containing the probabilities of each n-gram for n from 1 to n.

        Returns:
        - The probability of the word.
        """
        currWord = sentence[ind]
        context = sentence[ind-self.n+1:ind]
        currWord = currWord if currWord in self.vocab else '<UNK>'
        context = [word if word in self.vocab else '<UNK>' for word in context]
        try:
            prob = self.dpProb[tuple(context + [currWord])]
        except:
            KeyError
        pair = tuple(context[-1] + currWord)
        freqofPair = self.ngram_frequencies[1][pair]  # count of pair
        # tw = prob of seeing a new bigram starting with w1
        # nw = number of different n-gram (types) starting with w1
        # zw = number of unseen n-gram (tokens) starting with w1
        # w1 =
        tw = len([freq for cont, freq in self.ngram_items[1]
                 if freq > 0 and cont[0] == context[-1]])
        nw = sum(
            freq for cont, freq in self.ngram_items[1] if freq > 0 and cont[0] == context[-1])
        if(tw == 0):
            tw = 2
        if(freqofPair == 0):
            zw = len(self.vocab)-tw
            if(zw == 0):
                zw = 2
            try:
                prob = (tw/(zw*(nw+tw)))
            except:
                return float("NaN")
        else:
            try:
                prob = (freqofPair)/(nw+tw)
            except:
                return float("NaN")

        for i in range(2, self.n + 1):
            oldContext = context[-i+1:]
            oldWord = tuple(oldContext+[currWord])
            freqofOldWord = self.ngram_frequencies[i-1][oldWord]
            probArray = [frequency for cont, frequency in self.ngram_items[i-1]
                         if frequency > 0 and list(cont[:-1]) == list(oldContext)]
            totalFreq = sum(probArray)
            nw = len(probArray)
            if(nw == 0):
                nw = 2
            try:
                prob = ((freqofOldWord + nw*prob)/(totalFreq + nw))
            except:
                return float('NaN')
        prob = math.log(prob)

        self.dpProb[tuple(context+[currWord])] = prob
        return prob

    def perplexity(self, sentence, isTokenized):
        """
        Calculate the perplexity of a sentence.

        Parameters:
        - sentence: The sentence.

        Returns:
        - The perplexity of the sentence.
        """
        if(isTokenized == 'not'):
            prevLength = len(sentence)
            sentence = paddedString(Token(sentence), self.n)
        else:
            prevLength = len(sentence)
            sentence = paddedString(sentence, self.n)
        logp = 0
        # print(len(sentence))
        for index in range(self.n-1, len(sentence)):
            lp = self.getProbability(index, sentence)
            logp += lp

        try:
            perpl = math.exp(logp*(-1/len(sentence)))
        except:
            perpl = float("inf")
        return perpl


def splitTestTrainData(dataFile, trainFile, testFile):
    """
    Split the data into train and test data.

    Parameters:
    - dataFile: The data file.
    - trainFile: The train file.
    - testFile: The test file.
    """
    with open(dataFile, 'r') as f:
        data = f.readlines()
    with open(trainFile, 'w') as f:
        for i in range(2000):
            f.write(data[i])
    with open(testFile, 'w') as f:
        for i in range(2500,3000):
            f.write(data[i])


def generateResult(model, testFile, resultFile):
    """
    Generate the result file.

    Parameters:
    - model: The language model.
    - testFile: The test file.
    - resultFile: The result file.
    """

    resFile = open(resultFile, 'w')
    dataFile = open(testFile, 'r')
    avgPerplexity = 0
    storeResults = []
    totLines = 0
    for sentence in dataFile:
        sentence = sentence.rstrip()
        tokens = Token(sentence)
        perplexity = model.perplexity(tokens, 'tok')
        if(perplexity == float('NaN')):
            continue
        if perplexity != float('nan'):
            avgPerplexity += perplexity
        totLines += 1
        storeResults.append((sentence, perplexity))
        # resFile.write(f"{sentence}\t {perplexity}\n")

    avgPerplexity = avgPerplexity/totLines
    resFile.write(f"Average Perplexity: {avgPerplexity}\n")
    for sentence, perplexity in storeResults:
        resFile.write(f"{sentence}\t {perplexity}\n")
    resFile.close()
    dataFile.close()


def language_model(n, smoothingMethod, corupsPath, trainfileName, testfileName, trainResult, testResult):
    # if files doesn't exist, create them
    if not os.path.exists(trainResult):
        open(trainResult, 'w').close()
    if not os.path.exists(testResult):
        open(testResult, 'w').close()
    if not os.path.exists(trainfileName):
        open(trainfileName, 'w').close()
    if not os.path.exists(testfileName):
        open(testfileName, 'w').close()

    # split the data into train and test data
    splitTestTrainData(corupsPath, trainfileName, testfileName)

    model = None
    ngramFreq, vocab = createFreqTable(trainfileName, n)
    # print(ngramFreq)
    if(smoothingMethod == 'k'):
        model = KneserNey(vocab, ngramFreq, n)
    elif(smoothingMethod == 'w'):
        model = WittenBell(vocab, ngramFreq, n)

    # generate the result file for train data
    # generateResult(model, trainfileName, trainResult)
    # generate the result file for test data
    # print("Test Data: ")
    generateResult(model, testfileName, testResult)

    return model


smoothingMethod = sys.argv[1]
corpus = sys.argv[2]
md = language_model(4, smoothingMethod, corpus, '../DATA/train.txt',
                    '../DATA/test.txt', '../DATA/trainResult.txt', '../DATA/testResult.txt')

input_sentence = str(input("Enter a sentence: "))
# print(input_sentence)
tokens = Token(input_sentence)
# print(tokens)
perpl = md.perplexity(tokens, 'tok')
print(perpl)

   def getProbability(self, ind, sentence):
       """
        Get the probability of a indth word using (ind -n +1) to ind as context
        Parameters:
        - ind: The index of the word.
        - sentence: The sentence.
        - dpProb: A dictionary containing the probabilities of each n-gram for n from 1 to n.

        Returns:
        - The probability of the word.
        """
       currWord = sentence[ind]
       context = sentence[ind-self.n+1:ind]
       currWord = currWord if currWord in self.vocab else '<UNK>'
       context = [word if word in self.vocab else '<UNK>' for word in context]
       try:
            prob = self.dpProb[tuple(context + [currWord])]
        except:
            KeyError
        pair = tuple(context[-1] + currWord)
        freqofPair = self.ngram_frequencies[1][pair]  # count of pair
        # tw = prob of seeing a new bigram starting with w1
        # nw = number of different n-gram (types) starting with w1
        # zw = number of unseen n-gram (tokens) starting with w1
        tw = len([freq for cont, freq in self.ngram_items[1]
                 if freq > 0 and cont[0] == context[-1]])
        nw = sum(
            freq for cont, freq in self.ngram_items[1] if freq > 0 and cont[0] == context[-1])
        if(tw == 0):
            tw = 1
        if(freqofPair == 0):
            zw = len(self.vocab)-tw
            den = zw*(nw+tw)
            if den == 0:
                prob = 0
            else:
                prob = (tw/den)
        else:
            try:
                prob = (freqofPair)/(nw+tw)
            except:
                return 1/self.ngram_frequencies[0][('<UNK>',)]

        for i in range(3, self.n + 1):
            oldContext = context[-i+1:]
            oldWord = tuple(oldContext+[currWord])
            freqofOldWord = self.ngram_frequencies[i-1][oldWord]
            probArray = [frequency for cont, frequency in self.ngram_items[i-1]
                         if frequency > 0 and list(cont[:-1]) == list(oldContext)]
            totalFreq = sum(probArray)
            nw = len(probArray)
            if(nw == 0):
                nw = 2
            try:
                prob = ((freqofOldWord + nw*prob)/(totalFreq + nw))
            except:
                return float('NaN')
        prob = math.log(prob)

        self.dpProb[tuple(context+[currWord])] = prob
        return prob
