import time
import string
import re


punctu = ['.', ',', '!', '?', ';', ':', '—',
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
    return re.sub(r'[0-9]+(,([0-9]+))*(\.([0-9]+))?%?\s', '<NUMBER> ', text)


def removeExtraSpace(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+', '', text)
    text = re.sub(r'\s+$', '', text)
    return text


def removeExtraLine(text):
    text = re.sub(r'\n+', '', text)
    return text


def removeDataTime(text):
    text = re.sub(
        r'\d{2,4}\-\d\d-\d{2,4}|\d{2,4}\/\d\d\/\d{2,4}|\d{2,4}:\d\d:?\d{2,4}', '<DATE>', text)
    return re.sub(r'\d+:\d\d:?\d{0,2}?( am|am| pm|pm)', r'<TIME>', text)


def replaceApostrophe(txt_line):
    t1 = re.sub(r'can\'t', r'can not', txt_line)
    t1 = re.sub(r'won\'t', r'will not', txt_line)
    t1 = re.sub(r'([a-zA-Z]+)n\'t', r'\1 not', txt_line)
    t1 = re.sub(r'([a-zA-Z]+)\'s', r'\1 is', t1)
    t1 = re.sub(r'([iI])\'m', r'\1 am', t1)
    t1 = re.sub(r'([a-zA-Z]+)\'ve', r'\1 have', t1)
    t1 = re.sub(r'([a-zA-Z]+)\'d', r'\1 had', t1)
    t1 = re.sub(r'([a-zA-Z]+)\'ll', r'\1 will', t1)
    t1 = re.sub(r'([a-zA-Z]+)\'re', r'\1 are', t1)
    t1 = re.sub(r'([a-zA-Z]+)in\'', r'\1ing', t1)
    return t1


def Clean(str):
    str = str.lower()
    str = removeURL(str)
    str = removeHashtag(str)
    str = removeMentions(str)
    str = removeEmail(str)
    str = remove_punctuations(str)
    str = removeDataTime(str)
    str = removeNumber(str)
    str = replaceApostrophe(str)
    str = removeExtraSpace(str)
    return str

def Tokenise(str):
    x = Clean(str)
    return wordTokenizer(x)


def paddingString(text):
    ans = "<START> " + Clean(text)
    return ans
