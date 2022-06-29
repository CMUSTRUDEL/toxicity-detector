import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn_pandas import DataFrameMapper
import algorithms
import  data_cleaner

class STRUDEL_MODEL:
    def __init__(self, X_train, Y_train):
        self.vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,1),max_features=5000)

        self.mapper = None
        self.Y = None
        self.X = None

        self.clf = algorithms.linear_svm_model()
        self.__prepare_data(X_train, Y_train)
        self.model=self.train()


    def __prepare_data(self, X_train, Y_train):
        self.mapper = DataFrameMapper([
            ('text', self.vectorizer),
            ('num_url', None),
            ('num_emoji', None),
            ('num_mention', None),
            ('nltk_score', None),
            ('subjectivity', None),
            ('polarity', None),
            ('perspective_score', None),
            ('stanford_polite', None),
        ])
        self.Y = np.ravel(Y_train)

        self.X = self.mapper.fit_transform(X_train)  # adding the other features with bagofwords

    def train(self):
        print("Training the model with " + str(len(self.Y)) + " instances and " + str(
            self.X.shape[1]) + " features")
        self.clf.fit(self.X, self.Y)
        print("Model training complete ..")
        return self.clf

    def predict(self, X_test):
        X_test_mapped = self.mapper.transform(X_test)
        predictions = self.model.predict(X_test_mapped)
        return np.expand_dims(predictions, 1)


def read_dataframe_from_excel(file):
    dataframe = pd.read_excel(file)
    return dataframe

print("Reading dataset..")
#training_data = read_dataframe_from_excel("models/code_review_preprocessed.xlsx")
training_data = read_dataframe_from_excel("models/STRUDEL-issue-comments-dataset.xlsx")

print("Applying SE domain specific cleaning steps..")
training_data["text"] = training_data.text.astype(str).apply(data_cleaner.clean_text)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=999)

filename = "results/strudel-CV-issue-comments.csv"
#filename = "results/strudel-CV-code-review.csv"
training_log = open(filename, 'w')
training_log.write("Fold,precision_0,recall_0,f-score_0,precision_1,recall_1,f-score_1,accuracy\n")

count =1
results=""
print("Starting 10-fold cross validations..")
for train_index, test_index in kf.split(training_data, training_data["is_toxic"]):

    X_train, X_test = training_data.loc[train_index, ["text", "perspective_score",	"num_url",
                                                      "num_emoji",	"num_mention",	"nltk_score", "num_reference",
                                                      "subjectivity",	"polarity",	"stanford_polite"]], \
                      training_data.loc[test_index, ["text", "perspective_score",	"num_url",
                                                      "num_emoji",	"num_mention",	"nltk_score", "num_reference",
                                                      "subjectivity",	"polarity",	"stanford_polite"]]

    Y_train, Y_test = training_data.loc[train_index, "is_toxic"], training_data.loc[test_index, "is_toxic"]

    print("Fold# "+ str(count))
    classifier_model = STRUDEL_MODEL(X_train, Y_train)

    predictions = classifier_model.predict(X_test)

    precision_1 = precision_score(Y_test, predictions, pos_label=1)
    recall_1 = recall_score(Y_test, predictions, pos_label=1)
    f1score_1 = f1_score(Y_test, predictions, pos_label=1)

    precision_0 = precision_score(Y_test, predictions, pos_label=0)
    recall_0 = recall_score(Y_test, predictions, pos_label=0)
    f1score_0 = f1_score(Y_test, predictions, pos_label=0)
    accuracy = accuracy_score(Y_test, predictions)
    results = results + str(count) + ","

    results = results + str(precision_0) + "," + str(recall_0) + "," + str(f1score_0)
    results = results + "," + str(precision_1) + "," + str(recall_1) + "," + str(f1score_1) + \
                  "," + str(accuracy)  + "\n"

    print(classification_report(Y_test, predictions))

    count += 1
training_log.write(results)
training_log.flush()