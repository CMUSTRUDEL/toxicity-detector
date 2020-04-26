#from get_data import *
import logging

from text_modifier import *
from util import *
import itertools
import cross_validate
from statistics import *
from classifiers import *
from collections import Counter
import nltk
import time
import pandas as pd
import operator
import textblob
import pickle
from wordfreq import word_frequency
from nltk.corpus import wordnet
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from perspective import get_perspective_score
from copy import copy, deepcopy
from sklearn.model_selection import KFold
from nltk.tokenize import RegexpTokenizer
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
from pathos.multiprocessing import ProcessingPool as Pool
from lexicon import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import sys
sys.path.insert(0, "politeness3")
import politeness3.model
logging.info("import done")

model = 0

def rescore(new_sentence,features,tf_idf_counter):
    new_features_dict = {}

    if "length" in features:
        new_features_dict["length"] = len(new_sentence)

    if "perspective_score" in features:
        persp_score = get_perspective_score(new_sentence)
        new_features_dict["perspective_score"] = persp_score

    if "stanford_polite" in features:
        sentences = nltk.sent_tokenize(new_sentence)
        stanford_polite = politeness3.model.score(
            {"sentences": sentences,
             "text": new_sentence})['polite']

        new_features_dict["stanford_polite"] = stanford_polite

    if "word2vec_0" in features:
        # Calcualte word2vec
        df = pd.DataFrame([{'text': new_sentence}])
        df = add_word2vec(df).iloc[0]
        word2vec_values = [df['word2vec_{}'.format(i)] for i in range(300)]

        for i in range(300):
            new_features_dict['word2vec_{}'.format(i)] = word2vec_values[i]

    if "LIWC_anger" in features:
        s = score(open_lexicons(), new_sentence)
        new_features_dict["LIWC_anger"] = s["LIWC_anger"]

    if "negative_lexicon" in features:
        s = score(open_lexicons(), new_sentence)
        new_features_dict["negative_lexicon"] = s["negative_lexicon"]

    if "nltk_score" in features:
        sid = SentimentIntensityAnalyzer()
        nltk_score = sid.polarity_scores(new_sentence)['compound']
        new_features_dict["nltk_score"] = nltk_score

    if "polarity" in features or "subjectivity" in features:
        textblob_scores = textblob.TextBlob(new_sentence)
        new_features_dict["polarity"] = textblob_scores.polarity
        new_features_dict["subjectivity"] = textblob_scores.subjectivity

    if "tf_idf_0" in features:
        df = pd.DataFrame([{'text': new_sentence}])
        df = add_counts(tf_idf_counter,df,name="tf_idf_").iloc[0]

        for f in features:
            if "tf_idf_" in f:
                new_features_dict[f] = df[f]

    new_features = []
    for f in features:
        new_features.append(new_features_dict[f])

    return new_features

counter = pickle.load(open("pickles/github_words.p","rb"))
our_words = dict([(i,word_frequency(i,"en")*10**9) for i in counter])
different_words = log_odds(defaultdict(int,counter),defaultdict(int,our_words))
# logging.info("different_words"+str(different_words))

# returns [isToxic, perspective_score, stanford_polite_score]
def score_toxicity(text, model): 
    logging.info("get_prediction "+text)
    features = ["perspective_score","stanford_polite"]

    val = rescore(text,features,0)
    predict = model.predict([val])[0]
    logging.info("predicted "+str(predict)+" from "+str(val))

    return [predict, val[0], val[1]]

# postprocessing (usually only done for toxic comments)
# returns list of clean text variants
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
    return result

def get_prediction(text,model):
    logging.info("get_prediction "+text)
    features = ["perspective_score","stanford_polite"]

    val = rescore(text,features,0)
    predict = model.predict([val])[0]
    logging.info("predicted "+str(predict)+" from "+str(val))

    if predict == 0:
        return 0


    t = time.time()
    words = text.split(" ")
    words = [a.strip(',.!?:; ') for a in words]

    words = list(set(words))
    words = [word for word in words if not word.isalpha() or word.lower() in different_words]

    logging.info("words "+str(words))

    for word in set(words):
        # Maybe unkify?
        new_sentence = re.sub(r'[^a-zA-Z0-9]' + re.escape(word.lower()) + r'[^a-zA-Z0-9]', ' potato ', text.lower())
        new_features = rescore(new_sentence,features,0)
        prediction = model.predict([new_features])[0]
        logging.info("try with potato replacement for "+word+": "+new_sentence+" = "+str(prediction))
        
        if prediction == 0:
            return 0

    tokenizer = RegexpTokenizer(r'\w+')
    all_words = tokenizer.tokenize(text)
    # logging.info("all_words "+str(all_words))
    # Try removing all unknown words
    for word in set(all_words):
        if word.lower() not in counter and word_frequency(word.lower(), "en") == 0 and len(word) > 2:
            text = text.replace(word, '')


    new_features = rescore(text,features,0)
    prediction = model.predict([new_features])[0]
    logging.info(text +" = "+str(prediction))
    if prediction == 0:
        return 0

    return 1



def remove_SE_comment(text,model,features,tf_idf_counter):
    t = time.time()
    words = text.split(" ")
    words = [a.strip(',.!?:; ') for a in words]

    words = list(set(words))
    words = [word for word in words if not word.isalpha() or word.lower() in different_words]

    for word in set(words):
        # Maybe unkify?
        new_sentence = re.sub(r'[^a-zA-Z0-9]' + re.escape(word.lower()) + r'[^a-zA-Z0-9]', ' potato ', text.lower())
        new_features = rescore(new_sentence,features,tf_idf_counter)

        if model.predict([new_features])[0] == 0:
            return 1

    tokenizer = RegexpTokenizer(r'\w+')
    all_words = tokenizer.tokenize(text)
    # Try removing all unknown words
    for word in set(all_words):
        if word.lower() not in counter and word_frequency(word.lower(), "en") == 0 and len(word) > 2:
            text = text.replace(word, '')

    if model.predict([new_features])[0] == 0:
        return 1

    return 0


class Suite:
    def __init__(self):
        global different_words
        global counter

        self.features = []
        self.nice_features = []
        self.parameter_names = []
        self.hyper_parameters_lists = []
        self.last_time = time.time()
        self.tf_idf_counter = 0
        self.use_filters = True
        self.counter = pickle.load(open("pickles/github_words.p","rb"))
        counter = self.counter
        self.our_words = dict([(i,word_frequency(i,"en")*10**9) for i in self.counter])
        self.different_words = log_odds(defaultdict(int,self.counter),defaultdict(int,self.our_words))
        different_words = self.different_words
        self.anger_classifier = pickle.load(open("pickles/anger.p","rb"))
        self.all_words = pickle.load(open("pickles/all_words.p","rb"))
        self.m = sum(self.counter.values())
        self.all_false = {word: False for word in self.all_words}

        start_time = time.time()
        self.alpha = 0.1

        self.all_train_data = None
        self.test_data = None
        self.train_data = None
        self.model_function = None

    def set_model(self, model_function):
        self.model_function = model_function

    def add_parameter(self,name,l):
        self.parameter_names.append(name)
        self.hyper_parameters_lists.append(l)

    def matching_pairs(self,ratio):
        assert type(self.all_train_data) != type(None)

        matching_features = ["length"]
        potential_train_list = deepcopy(self.all_train_data)
        for i in range(len(potential_train_list)):
            potential_train_list.loc[i, 'index1'] = i

        potential_train_list = potential_train_list[potential_train_list["toxic"] == 0]
        potential_train_list = potential_train_list[matching_features + ["index1"]]

        potential_train_list = [tuple(x) for x in potential_train_list.values]

        toxic_data = self.all_train_data[self.all_train_data['toxic'] == 1][matching_features]

        indexes_we_want = []
        for i, row in toxic_data.iterrows():
            row_score = tuple(row)

            smallest_index = 0

            for j in range(1, len(potential_train_list)):
                if dist(potential_train_list[j][:-1], row_score) < dist(potential_train_list[smallest_index][:-1],row_score):
                    smallest_index = j

            indexes_we_want.append(potential_train_list[smallest_index][-1])
            potential_train_list.pop(smallest_index)

        non_toxic_random = pd.DataFrame()
        non_toxic_matched = self.all_train_data.iloc[indexes_we_want]

        if ratio-1 > 0:
            non_matched = list(set(range(len(self.all_train_data)) ) - set(indexes_we_want))
            non_toxic_random = self.all_train_data.iloc[non_matched]
            non_toxic_random = non_toxic_random[non_toxic_random["toxic"] == 0]
            non_toxic_random = non_toxic_random.sample(int((ratio-1)*len(toxic_data)))

        toxic_data = self.all_train_data[self.all_train_data["toxic"] == 1]
        total = toxic_data
        total = total.append([non_toxic_random,non_toxic_matched])

        self.train_data = total

        return indexes_we_want

    def set_ratios(self,ratios):
        self.ratios = ratios

    def set_train_set(self, train_collection):
        self.train_collection = train_collection
        self.all_train_data = get_all_comments(self.train_collection)
        self.all_train_data = prepare_dataset(self.all_train_data)
        print("Prepared training dataset, it took {} time".format(time.time()-self.last_time))
        self.last_time = time.time()

    def set_test_set(self, test_collection):
        self.test_collection = test_collection
        self.test_data = get_all_comments(self.test_collection)
        self.test_data = prepare_dataset(self.test_data)
        print("Prepared testing dataset, it took {} time".format(time.time()-self.last_time))

        self.last_time = time.time()

    def select_subset(self,ratio):
        self.train_data = select_ratio(self.all_train_data, ratio)

    def create_counter(self):
        body = random_issues()
        body = body["body"]
        a = []
        for i in body:
            if i != None:
                a += nltk.word_tokenize(i)
        a = [i.lower() for i in a]
        a = Counter(a)

        self.last_time = time.time()

        return a

    def get_anger_classifier(self):
        text = open("data/anger.txt").read().split("\n")
        label = [i.split("\t")[1] for i in text]
        train = [i.split("\t")[-1][1:-1] for i in text]
        train = [(train[i], label[i]) for i in range(len(train))]
        self.all_words = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
        all_words = self.all_words
        train = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]
        classifier = SklearnClassifier(LinearSVC())
        classifier.train(train)

        return classifier

    def convert(self,test_sentence):
        ret = copy(self.all_false)

        for word in word_tokenize(test_sentence.lower()):
            ret[word] = True

        return ret

    def remove_I(self,test_issues):
        test_issues.loc[test_issues.prediction != 1, "self_angry"] = 0

        test_issues.loc[test_issues.prediction == 1,"self_angry"] = test_issues[test_issues["prediction"] == 1]["original_text"].map(lambda x: self.anger_classifier.classify(self.convert(x)))

        test_issues.loc[test_issues.self_angry == "self","prediction"] = 0

        return test_issues


    def remove_SE(self,test_issues):

        features = self.features
        tf_idf_counter = self.tf_idf_counter
        model = self.model

        #p = Pool(8)
        test_issues.loc[test_issues.prediction != 1, "is_SE"] = 0
        original_text = test_issues[test_issues["prediction"] == 1]["original_text"]
        original_text = [remove_SE_comment(x,model,features,tf_idf_counter) for x in original_text]
        test_issues.loc[test_issues.prediction == 1, "is_SE"] = original_text
        #test_issues.loc[test_issues.prediction == 1, "is_SE"] = test_issues[test_issues["prediction"] == 1]["original_text"].map(lambda x: self.remove_SE_comment(x))
        test_issues.loc[test_issues.is_SE == 1,"prediction"] = 0

        return test_issues

    def classify_test(self):
        print(self.features)
        return classify(self.model, self.train_data, self.test_data, self.features)

    def classify_test_statistics(self):
        return classify_statistics(self.model, self.train_data, self.test_data, self.features)

    def cross_validate(self):
        return cross_validate.cross_validate(self.all_train_data,self.features,self.model)

    def cross_validate_classify(self):
        kfold = KFold(10)
        data = self.all_train_data.sample(frac=1)
        for train, test in kfold.split(data):
            train_data = data.iloc[train].copy()
            test_data = data.iloc[test].copy()
            train_data = select_ratio(train_data,self.ratio)

            test_data = classify(self.model, train_data, test_data, self.features)

            for i, row in test_data.iterrows():
                data.loc[data["_id"] == row["_id"], "prediction"] = row["prediction"]

        data = self.remove_I(data)
        data = self.remove_SE(data)

        test_issues = get_issues(get_labeled_collection())
        predicted = []
        for i, row in test_issues.iterrows():
            matching_comments = data[data["_id"].str.contains(row["_id"])]
            if (len(matching_comments) > 0):
                if (len(matching_comments[matching_comments["prediction"] == 1]) > 0):
                    predicted.append(1)
                else:
                    predicted.append(0)
            else:
                predicted.append(0)

        test_issues["prediction"] = predicted
        test_issues = map_toxicity(test_issues)

        print("Score is {}".format(calculate_statistics(test_issues["prediction"].tolist(), test_issues["toxic"].tolist())))

        return data

    def set_parameters(self):
        for ratio in self.ratios:
            for combination in itertools.product(*self.hyper_parameters_lists):
                self.combination_dict = {}
                for i in range(len(combination)):
                    self.combination_dict[self.parameter_names[i]] = combination[i]

                self.model = self.model_function(**self.combination_dict)

                self.select_subset(ratio)

    def issue_classifications_from_comments(self):
        t = time.time()
        test_issues = get_issues(self.test_collection)
        self.test_data = self.classify_test()
        self.test_data = self.remove_I(self.test_data)
        self.test_data = self.remove_SE(self.test_data)

        predicted = []
        values = []

        print("Looping through test_issues")

        predicted_toxic = list(self.test_data[self.test_data["prediction"] == 1]["_id"])
        predicted_toxic = ["/".join(i.split("/")[:-1]) for i in predicted_toxic]

        for i, row in test_issues.iterrows():
            if row["_id"] in predicted_toxic:
                predicted.append(1)
            else:
                predicted.append(0)

        test_issues["prediction"] = predicted
        test_issues = test_issues.sort_values("prediction")

        return test_issues

    def self_issue_classification_from_comments(self):
        test_issues = get_issues(self.train_collection)

        self.train_data = self.cross_validate_classify()
        self.train_data = self.remove_I(self.train_data)
        self.train_data = self.remove_SE(self.train_data)

        predicted = []
        for i, row in test_issues.iterrows():
            matching_comments = self.train_data[self.train_data["_id"].str.contains(row["_id"])]
            if (len(matching_comments) > 0):
                if (len(matching_comments[matching_comments["prediction"] == 1]) > 0):
                    predicted.append(1)
                else:
                    predicted.append(0)
            else:
                predicted.append(0)

        test_issues["prediction"] = predicted
        test_issues = map_toxicity(test_issues)
        test_issues = test_issues.sort_values("prediction")

        return test_issues

    def train_test_validate(self):
        global model

        # This essentially performs nested Cross Validation
        # To test how well a particular set of features is doing

        print("Train test validating")

        # train test split
        kfold = KFold(10)
        data = self.all_train_data.sample(frac=1)

        all_issues = get_issues(get_labeled_collection())
        all_comments = list(data["_id"])

        all_issues = all_issues.sample(frac=1)

        for train, test in kfold.split(all_issues):
            train = all_issues.iloc[train].copy().sample(frac=1 )
            test = all_issues.iloc[test].copy()

            train_issues = list(train["_id"])
            test_issues = list(test["_id"])

            train_comments_list = [i for i in all_comments if "/".join(i.split("/")[:-1]) in train_issues]
            test_comments_list = [i for i in all_comments if "/".join(i.split("/")[:-1]) in test_issues]

            train_comments = data[data["_id"].isin(train_comments_list)]
            test_comments = data[data["_id"].isin(test_comments_list)]

            kfold_validation = KFold(10)
            train_comments = train_comments.sample(frac=1)

            predicted_train_data = {}  # type: Dict[str, Df]

            # Find the best combo
            for ratio in self.ratios:
                for combination in itertools.product(*self.hyper_parameters_lists):
                    self.combination_dict = {}
                    for i in range(len(combination)):
                        self.combination_dict[self.parameter_names[i]] = combination[i]
                    predicted_train_data["{}_{}".format(ratio, self.combination_dict)] = deepcopy(train_comments)

            for real_train,validation in kfold_validation.split(train):
                real_train = train.iloc[real_train].copy()
                validation = train.iloc[validation].copy()

                real_train_issues = list(real_train["_id"])
                validation_issues = list(validation["_id"])


                real_train_comments = [i for i in train_comments_list if "/".join(i.split("/")[:-1]) in real_train_issues]
                validation_comments = [i for i in train_comments_list if "/".join(i.split("/")[:-1]) in validation_issues]

                for ratio in self.ratios:
                    self.ratio = ratio

                    real_train = data[data["_id"].isin(real_train_comments)].copy()
                    validation = data[data["_id"].isin(validation_comments)].copy()

                    real_train = select_ratio(real_train,self.ratio)

                    for combination in itertools.product(*self.hyper_parameters_lists):
                        self.combination_dict = {}
                        for i in range(len(combination)):
                            self.combination_dict[self.parameter_names[i]] = combination[i]

                        self.model = self.model_function(**self.combination_dict)
                        model = self.model

                        parameter_train = deepcopy(predicted_train_data["{}_{}".format(ratio,self.combination_dict)])

                        validation_predicted = classify(self.model, real_train, validation, self.features)
                        # Add our predictions to train
                        for i, row in validation_predicted.iterrows():
                            parameter_train.loc[parameter_train["_id"] == row["_id"], "prediction"] = row["prediction"]

                        predicted_train_data["{}_{}".format(ratio, self.combination_dict)] = deepcopy(parameter_train)

            included_ids = list(set(["/".join(row["_id"].split("/")[:-1]) for i, row in train_comments.iterrows()]))

            test_issues = get_issues(get_labeled_collection())
            test_issues = test_issues[test_issues['_id'].isin(included_ids)]

            score_dict = {}

            # Now we evaluate each of the hyperparameter combinations
            for combo in predicted_train_data:
                real_train = predicted_train_data[combo]
                temp_test_issues = map_toxicity(deepcopy(test_issues))
                predicted = []
                for i, row in temp_test_issues.iterrows():
                    matching_comments = real_train[real_train["_id"].str.contains(row["_id"])]
                    if (len(matching_comments) > 0):
                        if (len(matching_comments[matching_comments["prediction"] == 1]) > 0):
                            predicted.append(1)
                        else:
                            predicted.append(0)
                    else:
                        predicted.append(0)

                temp_test_issues["prediction"] =  predicted
                score = calculate_statistics(temp_test_issues["prediction"].tolist(), temp_test_issues["toxic"].tolist())['f_0.5']

                score_dict[combo] = score

            max_pair = max(score_dict,key=score_dict.get).split("_")
            ratio = float(max_pair[0])

            combo_dict = eval("_".join(max_pair[1:]))

            self.ratio = ratio
            self.model = self.model_function(**combo_dict)

            train_comments = select_ratio(train_comments,ratio)

            test_predicted = classify(self.model, train_comments.copy(), test_comments.copy(), self.features).copy()
            # Add our predictions to train
            for i, row in test_predicted.iterrows():
                data.loc[data["_id"] == row["_id"], "prediction"] = row["prediction"]

        if self.use_filters:
            data = self.remove_SE(data)

        test_issues = get_issues(get_labeled_collection())
        predicted = []

        for i, row in test_issues.iterrows():
            matching_comments = data[data["_id"].str.contains(row["_id"])]
            if (len(matching_comments) > 0):
                if (len(matching_comments[matching_comments["prediction"] == 1]) > 0):
                    predicted.append(1)
                else:
                    predicted.append(0)
            else:
                predicted.append(0)

        test_issues["prediction"] = predicted
        test_issues = map_toxicity(test_issues)
        test_issues = test_issues.sort_values("prediction")

        score = calculate_statistics(test_issues["prediction"].tolist(), test_issues["toxic"].tolist())
        return score

    def all_combinations(self,function,matched_pairs=False):
        global model

        scores = {}
        for ratio in self.ratios:
            if matched_pairs:
                self.matching_pairs(ratio)
                if "matched_pairs" not in self.nice_features:
                    self.nice_features+=["matched_pairs"]
            else:
                self.select_subset(ratio)

            self.ratio = ratio
            for combination in itertools.product(*self.hyper_parameters_lists):
                self.combination_dict = {}
                for i in range(len(combination)):
                    self.combination_dict[self.parameter_names[i]] = combination[i]

                self.model = self.model_function(**self.combination_dict)
                model = self.model

                s = function()
                scores["{},{}".format(ratio,str(self.combination_dict))] = s

        return max(scores.items(),key=operator.itemgetter(1))

    def self_issue_classification_all(self,matched_pairs=False):
        def self_issue_classification_statistics_per():
            test_issues = self.self_issue_classification_from_comments()
            score = calculate_statistics(test_issues["prediction"].tolist(), test_issues["toxic"].tolist())

            print("{}\t{}\t ratio: {}\t precision: {}\t recall: {}\t f0.5: {}".format(",".join(self.nice_features),
                                                  "{} {} ".format(self.model_function.__doc__, self.combination_dict), self.ratio,
                                score['precision'], score['recall'],score['f_0.5']))

            return score['f_0.5']

        best_score = self.all_combinations(self_issue_classification_statistics_per,matched_pairs=matched_pairs)
        print("Best score {}".format(best_score))

    def test_issue_classifications_from_comments_all(self,matched_pairs=False):
        def test_issue_classifications_from_comments_statistics_per():
            t = time.time()
            test_issues = self.issue_classifications_from_comments()
            test_issues = map_toxicity(test_issues)

            for i,row in test_issues.iterrows():
                print(row["_id"],row["prediction"])

        self.all_combinations(test_issue_classifications_from_comments_statistics_per,matched_pairs=matched_pairs)

    def self_comment_classification_all(self,matched_pairs=False):
        def self_comment_classification_statistics_per():
            score = self.cross_validate()

            print("{}\t{}\t{}\t{}\t{}\t{}".format(",".join(self.nice_features), "{} {} ".format(self.model_function.__doc__,self.combination_dict), self.ratio,
                                      score['precision'], score['recall'],score['auc']))
        self.all_combinations(self_comment_classification_statistics_per,matched_pairs=matched_pairs)


    def test_comment_classifications_from_comments_all(self,matched_pairs=False):
        def issue_classifications_from_comments_statistics_per():
            score = self.classify_test_statistics()

            print("{}\t{}\t{}\t{}\t{}".format(",".join(self.nice_features), "{} {} ".format(self.model_function.__doc__,self.combination_dict), self.ratio,score['precision'], score['recall']))
        self.all_combinations(test_comment_classifications_from_comments_statistics,matched_pairs=matched_pairs)
