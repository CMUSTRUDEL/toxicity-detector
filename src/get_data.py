from pymongo import MongoClient
import pandas as pd
import random
from copy import deepcopy

def connect_to_database():
    global db
    mongo_name = "NAME"
    mongo_password = "PASSWORD"

    # Connect to the Mongo Database
    client = MongoClient()
    db = client.ghtorrent
    db.authenticate(name=mongo_name, password=mongo_password)
    return db

def connect_to_mysql():
    import mysql
    global cursor
    global mydb
    mydb = mysql.connector.connect(user = "NAME", passwd = "PASSWORD",host = "HOST",database="DATABASE")
    cursor = mydb.cursor()

def find_language(repo):
    statement = "SELECT language FROM projects WHERE url='{}' LIMIT 1;".format(repo)
    cursor.execute(statement)

    for x in cursor:
        return x[0]

try:
    connect_to_database()
except:
    print("Couldn't connect to Mongo")
#connect_to_mysql()

def get_labeled_collection():
    a= {'issues':db.naveen_issues,'comments':db.naveen_labeled_stanford_comments,'name':'labeled'}
    return a

def get_unlabeled_collection():
    a = {'issues':db.naveen_unlabeled_stanford_issues,'comments':db.naveen_unlabeled_stanford_comments,'name':'unlabeled'}
    return a

def get_closed_collection():
    a = {'issues':db.naveen_closed_stanford_issues,'comments':db.naveen_closed_stanford_comments,'name':'closed'}
    return a

def get_toxic_collection():
    a = {'issues':db.naveen_toxic_issues,'comments':db.naveen_toxic_comments,'name':'toxic'}
    return a

def logout():
    global db
    db.logout()

def to_dataframe(collection):
    return pd.DataFrame(list(collection.find()))

def get_issues(collection):
    return to_dataframe(collection['issues'])

def get_all_comments(collection):
    return to_dataframe(collection['comments'])

def get_comments(collection,owner,repo,issue_id):
    return pd.DataFrame(list(collection['comments'].find({"repo":repo,"owner":owner,"issue_id":issue_id})))

def add_comment_score(data_original,collection,top_scores):
    """ Get the comments with the highest perspective score """

    data = deepcopy(data_original)
    scores = ["nltk_score","perspective_score","polarity","subjectivity","stanford_polite"]
    score_dict = {}
    for score in scores:
        score_dict[score] = []

    for i, row in data.iterrows():
        temp_dict = {}
        for score in scores:
            temp_dict[score] = [0 for j in range(top_scores)]

        comments = get_comments(collection,row["owner"], row["repo"], row["issue_id"])

        if not comments.empty:
            comments = comments.sort_values(by="perspective_score",ascending=False).to_dict('records')

            for j in range(min(len(comments),top_scores)):
                for score in scores:
                    temp_dict[score][j] = comments[j][score]

        for score in scores:
            score_dict[score].append(temp_dict[score])


    for score in scores:
        for i in range(top_scores):
            data["min_{}_{}".format(score,i)] = [j[i] for j in score_dict[score]]

    return data

# Map toxicity to a 0 or 1 value
def map_toxicity(data):
    data['toxic'] = data['toxic'].map({'y':1,'n':0})
    return data

# Returns a list with something appeneded on n times
def append_to_str(s,n):
    return ["{}{}".format(s,i) for i in range(n)]

# A list of tuples (s,n) each of which are a set of features
# That are combined into a single list
# Like [("nltk_score_",10),("word2vec_",300)]
def append_to_str_multiple(l):
    ret = []
    for s,n in l:
        ret+=append_to_str(s,n)
    return ret

def random_issues():
    db = connect_to_database()
    return pd.DataFrame(list(db.issues.aggregate([{ "$sample": {"size": 10000}}])))

def get_wikipedia_data(nrows):
    """ Read data about wikipedia """

    original_train_data = pd.read_csv("data/wikipedia_classified.txt", sep="\t", nrows=nrows)
    num_y = len(original_train_data[original_train_data["toxic"] == "y"])
    yes = original_train_data[original_train_data["toxic"] == "y"]
    no = original_train_data[original_train_data["toxic"] == "n"].sample(num_y)

    original_train_data = yes
    original_train_data = original_train_data.append(no)

    return original_train_data

