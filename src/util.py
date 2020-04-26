import pandas as pd
from text_modifier import *
from get_data import *
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import math

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

def select_ratio(df,ratio):
    """ In a dataframe (df) with comments, select a subset, so that
        The ratio of toxic to non-toxic is 1:ratio """

    all_toxic = df[df["toxic"] == 1]
    non_toxic = df[df["toxic"] == 0]
    non_toxic = non_toxic.sample(min(len(non_toxic),int(ratio*len(all_toxic))))
    return pd.concat([all_toxic,non_toxic])

def find_surrounding_comments(df,comment_id,window=1):
    """ Find the surrounding comments
        In other words, given a comment and a dataframe,
        Find the comments +- window """

    # The comment ID is in the form repo/user/issue_number/comment_number
    # To search the issue dataframe, we need it in the form repo/user/issue_number
    repo = comment_id.split("/")[:-1]
    repo = "/".join(repo)

    # The row number of the comment in the full dataframe
    row_number = df.loc[df['_id'] == comment_id].index[0]

    before = pd.DataFrame()
    after = pd.DataFrame()

    for i in range(window):
        # Search the one before
        if row_number - i - 1 >= 0:
            previous_comment = df.iloc[[row_number-i-1]]

            if repo in previous_comment.iloc[0]["_id"]:
                before = before.append(previous_comment)

        # Search the one after
        if row_number+i+1<len(df):
            after_comment = df.iloc[[row_number+i+1]]

            if repo in after_comment.iloc[0]["_id"]:
                after = after.append(after_comment)

    return {'before':before,'after':after}

def remove_large_comments(train_data,cutoff=5000):
    train_data["length"] = train_data["text"].map(len)
    return train_data[train_data["length"]<cutoff]

def scale_data(train_data,scale_features):
    """ Use a StandardScaler to scale each of the scale_features in train_data """

    train_data = train_data.copy()

    ss = StandardScaler()

    for f in scale_features:
        transformed = ss.fit_transform(np.array(train_data[f]).reshape(-1,1))

        train_data[f] = list(transformed)

    return train_data

def important_features(model):
    """ Get the most important features from some model """

    invalid_features = ["count_","tf_idf_","word2vec_"]
    feat_importances = pd.Series(model.feature_importances_, index=features).to_string()
    feat_importances = [i for i in feat_importances if not any(feat in i for feat in invalid_features)]
    return "\n".join(feat_importances)

def prepare_dataset(train_data,map_toxic=True):
    """ Calculate several things we need for our dataset """

    # For issues, we need to convert total_text to text
    if "total_text" in train_data:
        train_data["text"] = train_data["total_text"]

    train_data["text"] = train_data["text"].map(str)
    train_data["original_text"] = train_data["text"]
    train_data["uppercase"] = train_data["text"].map(percent_uppercase)
    train_data["length"] = train_data["text"].map(len)
    #train_data["text"] = train_data["text"].map(cleanup_text)

    # Change toxicity from y/n to 1/0
    if 'toxic' in train_data:
        train_data = map_toxicity(train_data)
        train_data = train_data[train_data["toxic"] >= 0]

    return train_data

def dist(t1,t2):
    """ Euclidean distance between two lists """
    return sum([(t1[i]-t2[i])**2 for i in range(len(t1))])
