import math
import pickle
import re
from wordfreq import word_frequency
from nltk import RegexpTokenizer

from collections import defaultdict

def log_odds(counts1,counts2):
    prior = counts2

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in prior.keys():
        prior[word] = int(prior[word] + 0.5)

    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    n1  = sum(counts1.values())
    n2  = sum(counts2.values())
    nprior = sum(prior.values())

    for word in prior.keys():
        if prior[word] > 0:
            l1 = float(counts1[word] + prior[word]) / (( n1 + nprior ) - (counts1[word] + prior[word]))
            l2 = float(counts2[word] + prior[word]) / (( n2 + nprior ) - (counts2[word] + prior[word]))
            sigmasquared[word] =  1/(float(counts1[word]) + float(prior[word])) + 1/(float(counts2[word]) + float(prior[word]))
            sigma[word] =  math.sqrt(sigmasquared[word])
            delta[word] = ( math.log(l1) - math.log(l2) ) / sigma[word]

    different_words = []

    for word in sorted(delta, key=delta.get):
        if delta[word]>1.645:
            different_words.append(word)

    return different_words

counter = pickle.load(open("./models/github_words.p","rb"))
our_words = dict([(i,word_frequency(i,"en")*10**9) for i in counter])
different_words = log_odds(defaultdict(int,counter),defaultdict(int,our_words))


def clean_text(text):
    result = []
    words = text.split(" ")
    words = [a.strip(',.!?:; ') for a in words]

    words = list(set(words))
    words = [word for word in words if not word.isalpha() or word.lower() in different_words]

    for word in set(words):
        # Maybe unkify?
        result += [re.sub(r'[^a-zA-Z0-9]' + re.escape(word.lower()) + r'[^a-zA-Z0-9]', ' potato ', " "+text.lower()+" ").strip()]

    tokenizer = RegexpTokenizer(r'\w+')
    all_words = tokenizer.tokenize(text)
    # logging.info("all_words "+str(all_words))
    # Try removing all unknown words
    for word in set(all_words):
        if word.lower() not in counter and word_frequency(word.lower(), "en") == 0 and len(word) > 2:
            text = text.replace(word, '')

    result += [text]
    text_mod =' '.join(result)
    return text_mod