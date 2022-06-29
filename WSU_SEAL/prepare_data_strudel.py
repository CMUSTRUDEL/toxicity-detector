import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("..")

from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import textblob
import PretrainedPolitenessModel

import re
import PPAClient


import nltk

nltk.download('words')
nltk.download('punkt')
nltk.download('vader_lexicon')

url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mention_pattern =re.compile('@\w+')

emoticon_list={':)', ':(', ':/', ':O', ':o', ':-(', '>:)', '<|:O', ':?:', ':-|', '|-O',
               '</3', ':(', ':-)', ':-*', ':D', '<3', ':S', ':P', ';)',';-)',':-o' }


def read_lines_from_model(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
        return lines

LIWC_list = read_lines_from_model('models/anger-words.txt')

def count_anger_words(text):
    count = 0
    words = word_tokenize(text)
    for anger_word in LIWC_list:
        if anger_word in words:
            count = count + 1
    return count

def read_dataframe_from_excel(file):
    dataframe = pd.read_excel(file)
    return dataframe


training_data = read_dataframe_from_excel("models/code-review-dataset-full.xlsx")

vader = SentimentIntensityAnalyzer()

def emoji_counter(text):
    return len(list(filter(lambda x: x in emoticon_list, text.split(' '))))

def count_mention(text):
    return len(re.findall(mention_pattern, text))

def count_url(text):
    return len(re.findall(url_regex, text))


def get_nltk_score(text):
    return vader.polarity_scores(text)['compound']

def get_textblob_polarity(text):
	textblob_scores = textblob.TextBlob(text)
	return  textblob_scores.polarity

def get_textblob_subjectivity(text):
    textblob_scores = textblob.TextBlob(text)
    return textblob_scores.subjectivity

def get_stanford_politeness(text):
    tokenized = nltk.sent_tokenize(text)
    score = PretrainedPolitenessModel.score({'sentences':tokenized,'text':text})['polite']
    return score

print("Counting number of URLS..")
training_data['num_url'] = training_data.text.apply(count_url)

print("Counting number of emoticons..")
training_data['num_emoji'] = training_data.text.apply(emoji_counter)

print("Counting number of references to anger words..")
training_data['num_reference'] = training_data.text.apply(count_anger_words)

if 'perspective_score' not in training_data.columns:
    print("Getting perspective API scores..")
    training_data['perspective_score'] = training_data.text.apply(PPAClient.get_perspective_api_score)

print("Counting number of mentions..")
training_data["num_mention"] = training_data.text.astype(str). \
    apply(count_mention)

print("Counting number of URLS..")
training_data["nltk_score"] = training_data.text.astype(str). \
    apply(get_nltk_score)

print("Computing textblob subjectivity..")
training_data["subjectivity"] = training_data.text.astype(str). \
    apply(get_textblob_subjectivity)

print("Computing textblob polarity..")
training_data["polarity"] = training_data.text.astype(str). \
    apply(get_textblob_polarity)

print("Computing politeness score...")
training_data["stanford_polite"] = training_data.text.astype(str). \
    apply(get_stanford_politeness)

print("Computing VADER sentiment..")
training_data["nltk_score"] = training_data.text.astype(str). \
    apply(get_nltk_score)

print("Saving preprocessed dataset")
training_data.to_excel("models/code_review_preprocessed.xlsx")
print("Finished!")