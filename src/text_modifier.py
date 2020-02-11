from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.stem import PorterStemmer
import pandas as pd
import spacy
import numpy as np
import nltk
import re
import nltk
from get_data import *
from lexicon import *
import time

nlp = spacy.load("en_core_web_md",disable=["parser","ner"])
ps = PorterStemmer()
words = set(nltk.corpus.words.words())

def is_ascii(s):
    """ Check if a character is ascii """
    return all(ord(c) < 128 for c in s)

def percent_uppercase(text):
    """ Calculate what percent of the letters are uppercase in some text """
    text = text.replace(" ","")
    if len(text) == 0:
        return 0
    return sum([1 for i in text if i.isupper()])/len(text)

def cleanup_text(text):
    """ Remove stop words and stem words"""

    text = nlp(text.lower().strip())
    # Stem Non-Urls/non-Stop Words/Non Punctuation/symbol/numbers
    text = [ps.stem(re.sub(r'^https?:\/\/.*[\r\n]*', '', token.text, flags=re.MULTILINE))
            for token in text
            if not token.is_stop and token.pos_ not in ["PUNCT","SYM","NUM"] and token.text in nlp.vocab]
    # Remove ampersands
    text = [re.sub(r'&[^\w]+','',i) for i in text]
    # Lower case
    text = [w for w in text if w.lower() in text]
    # Remove symbols
    text = [w.replace("#","").replace("&","").replace("  "," ") for w in text if is_ascii(w)]

    return " ".join(text)

def count_vector(text):
    """ Create count vector for text """

    count_vect = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,1))
    count_vect.fit(text)

    return count_vect

def tf_idf(text):
    """ Create TF IDF for text """

    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,1),max_features=5000)
    tfidf_vect.fit(text)

    return tfidf_vect

def add_counts(counter,df,name="count_"):
    """ Add some counter (TFIDF/Count Vector) as a feature to dataframe """

    counts = counter.transform(df["text"].tolist())

    # Create the column names
    num_words = counts.shape[1]
    column_names = [name+str(i) for i in range(num_words)]

    # Add in the count matrices
    counts = pd.DataFrame(counts.toarray(),columns=column_names)
    return pd.concat([df.reset_index(drop=True),counts.reset_index(drop=True)],axis=1)

def get_counts(counter,df):
    """ Counter to list """

    counts = counter.transform(df["total_text"].tolist())
    return counts

def add_word2vec(df):
    """ Add word2vec to a dataframe """

    word_list = [nlp(i) for i in df["text"]]

    vector_representation = []
    for i in range(len(word_list)):
        vector_representation.append(list(word_list[i].vector))

    column_names = ["word2vec_"+str(i) for i in range(len(vector_representation[0]))]
    return pd.concat([df.reset_index(drop=True),
        pd.DataFrame(vector_representation,columns=column_names).reset_index(drop=True)]
        ,axis=1)

def add_linguistic_scores_suite(s):
    """ Add differnet linguistics scores, such as whether it's doing name calling """

    datasets = [s.all_train_data]

    if type(s.test_data) != type(None):
        datasets.append(s.test_data)

    for data in datasets:
        data["insult_product"] = data["text"].map(lambda x: score_comment(x,insult_product))
        data["reaction_to_toxicity"] = data["text"].map(lambda x: score_comment(x,reaction_to_toxicity))
        data["name_calling"] = data["text"].map(lambda x: score_comment(x,score_name_calling))
        data["toxicity"] = data["text"].map(lambda x: score_comment(x,score_toxic))
        data["frustrated"] = data["text"].map(lambda x: score_comment(x,user_frustrated))

    s.features += ["insult_product", "reaction_to_toxicity", "name_calling", "toxicity", "frustrated"]
    s.nice_features += ["insult_product", "reaction_to_toxicity", "name_calling", "toxicity", "frustrated"]

    return s

def add_count_vector_suite(s):
    """ Add count vector to a suite """

    if type(s) != type(None):
        s.counter = count_vector(train_data["text"].tolist() + test_data["text"].tolist())
    else:
        s.counter = count_vector(train_data["text"].tolist())
    num_words = len(s.counter.get_feature_names())

    s.all_train_data = add_counts(s.counter, s.all_train_data)

    if type(s) != type(None):
        s.test_data = add_counts(s.counter, s.test_data)

    s.features+=append_to_str("count_",num_words)
    s.nice_features += ["count"]

    return s

def add_tf_idf_suite(s):
    """ Add TF IDF to a suite """

    start = time.time()
    print("Adding TF_IDF Suite")

    if type(s.test_data) != type(None):
        s.tf_idf_counter = tf_idf(s.all_train_data["text"].tolist() + s.test_data["text"].tolist())
    else:
        s.tf_idf_counter = tf_idf(s.all_train_data["text"].tolist())
    s.tf_idf_words = len(s.tf_idf_counter.get_feature_names())
    s.all_train_data = add_counts(s.tf_idf_counter, s.all_train_data, name="tf_idf_")

    if type(s.test_data) != type(None):
        s.test_data = add_counts(s.tf_idf_counter, s.test_data, name="tf_idf_")

    s.features+=append_to_str("tf_idf_",s.tf_idf_words)
    s.nice_features += ["tf_idf"]

    print("Finished adding TF IDF, it took {} time".format(time.time()-start))

    return s

def add_word2vec_suite(s):
    """ Add word2vec to a suite """

    if type(s.test_data) != type(None):
        s.test_data = add_word2vec(s.test_data)
    s.all_train_data = add_word2vec(s.all_train_data)
    s.features+=append_to_str("word2vec_",300)
    s.nice_features += ["word2vec"]

    return s

def add_lexicons_suite(s):
    """ Add lexicons to a suite """

    s.all_train_data = add_lexicons_comment(s.all_train_data)

    if type(s.test_data) != type(None):
        s.test_data = add_lexicons_comment(s.test_data)

    return s

def add_context_suite(s,window,aggregate=True,past=False):
    for i,row in s.all_train_data.iterrows():
        if aggregate:
            for f in s.features:
                s.all_train_data.ix[i, "{}_{}".format(f, "before")] = 0
                s.all_train_data.ix[i, "{}_{}".format(f, "after")] = 0

        surrounding_comments = util.find_surrounding_comments(s.all_train_data,row["_id"],window=window)
        before = surrounding_comments["before"]
        after = surrounding_comments["after"]

        for j in range(window):
            # Get the before and the after windows
            if j < len(before):
                before_df = before.iloc[[j]][s.features]
            else:
                before_df = pd.DataFrame(np.zeros((1, len(s.features))),columns=s.features)

            if j<len(after):
                after_df = after.iloc[[j]][s.features]
            else:
                after_df = pd.DataFrame(np.zeros((1, len(s.features))),columns=s.features)

            for f in s.features:

                if aggregate:
                    s.all_train_data.ix[i, "{}_{}".format(f, "before")]+= list(before_df[f])[0]/window
                    s.all_train_data.ix[i, "{}_{}".format(f, "after")]+= list(after_df[f])[0]/window
                else:
                    s.all_train_data.ix[i,"{}_{}_{}".format(f,"before",j)]= list(before_df[f])[0]
                    s.all_train_data.ix[i,"{}_{}_{}".format(f,"after",j)] = list(after_df[f])[0]
    new_features = []
    if True:
        if aggregate:
            for f in s.features:
                if "{}_{}".format(f,"before") not in new_features:
                    new_features+=["{}_{}".format(f,"before")]
                if past:
                    for f in s.features:
                        if "{}_{}".format(f,"after") not in new_features:
                            new_features+=["{}_{}".format(f,"after")]
        else:
            for i in range(window):
                for f in s.features:
                    if "{}_{}_{}".format(f,"before",i) not in new_features:
                        new_features+=["{}_{}_{}".format(f,"before",i)]
                if past:
                    for f in s.features:
                        if "{}_{}_{}".format(f,"after",i) not in new_features:
                            new_features+=["{}_{}_{}".format(f,"after",i)]

        s.features+=new_features

        if aggregate:
            if past:
                if "aggregate_context_window_past_{}".format(window) not in s.nice_features:
                    s.nice_features+=["aggregate_context_window_past_{}".format(window)]
            else:
                if "aggregate_context_window_{}".format(window) not in s.nice_features:
                    s.nice_features+=["aggregate_context_window_{}".format(window)]
        else:
            if past:
                if "context_window_past_{}".format(window) not in s.nice_features:
                    s.nice_features+=["context_window_past_{}".format(window)]
            else:
                if "context_window_{}".format(window) not in s.nice_features:
                    s.nice_features+=["context_window_{}".format(window)]

    return s
