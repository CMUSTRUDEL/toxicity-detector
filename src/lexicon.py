import pandas as pd
import nltk
import text_modifier

def open_lexicons():
    return pd.read_csv("data/lexicons.txt")

def get_lexicon_list():
    all_specific = open_lexicons()["specific"].unique().tolist()
    return all_specific

def score(lexicon_dataframe,text):
    """Need to do stemming later"""


    all_specific = lexicon_dataframe["specific"].unique()

    text = nltk.word_tokenize(text)
    text = [i.lower() for i in text]
    #text = set([text_modifier.ps.stem(i) for i in text])

    score_dict = {}
    for category in all_specific:
        score_dict[category] = 0
        category_words = set(lexicon_dataframe[lexicon_dataframe["specific"] == category]["word"].tolist())
        score_dict[category] = len(category_words.intersection(text))

    return score_dict

def add_lexicons(df):
    o = open_lexicons()
    all_lexicons = df["total_text"].map(lambda x: score(o,x)).tolist()
    categories = all_lexicons[0].keys()

    for key in categories:
        results = []
        for i in all_lexicons:
            results.append(i[key])

        df[key] = results

    return df

def add_lexicons_comment(df):
    o = open_lexicons()
    all_lexicons = df["text"].map(lambda x: score(o,x)).tolist()
    categories = all_lexicons[0].keys()

    for key in categories:
        results = []
        for i in all_lexicons:
            results.append(i[key])

        df[key] = results

    return df
