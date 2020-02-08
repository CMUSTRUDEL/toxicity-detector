from suite import *
from copy import copy
from get_data import *
from text_modifier import *
from classifiers import svm_model
from contextlib import redirect_stdout

def remove_no_mutation(l,a):
    return [i for i in l if i!=a]

def find_score(combo,s=None):
    if s == None:
        s = Suite()
        s.set_train_set(get_labeled_collection())

    s.features = combo
    s.nice_features = combo
    s.add_parameter("gamma",[1,2,2.5,3])
    s.add_parameter("C",[0.01,0.05,0.1,0.5,1,10])
    s.set_ratios([1,1.5,1.75,2,2.25])

    s.set_model(svm_model)
    score = s.train_test_validate()
    return score

def remove_features():
    combo = ["stanford_polite","perspective_score"]
    removal_features = copy(combo)
    baseline = find_score(combo)
    print("Baseline: {}".format(baseline))
    for feature in removal_features:
        score = find_score(remove_no_mutation(combo,feature))
        print("W/O {}: {}".format(feature,score))

def remove_filters():
    s = Suite()
    combo = ["stanford_polite","perspective_score"]
    s.use_filters = False
    s.set_train_set(get_labeled_collection())
    score = find_score(combo,s=s)
    print("W/O filters: {}".format(score))

def add_features():
    combo = ["stanford_polite","perspective_score"]

    # NLTK Score is Vader
    #removal_features = ["length","polarity","subjectivity","nltk_score"]
    removal_features = ["nltk_score"]
    for feature in removal_features:
        score = find_score(combo+[feature])
        print("{}: {}".format(feature,score))

def add_lexicons():
    combo = ["perspective_score","stanford_polite"]
    lexicon_features = ["LIWC_anger","negative_lexicon"]

    s = Suite()
    s.set_train_set(get_labeled_collection())
    add_lexicons_suite(s)

    for feature in lexicon_features:
        score = find_score(combo+[feature],copy(s))
        print("{}: {}".format(feature,score))

def add_word2vec():
    combo = ["perspective_score","stanford_polite"]

    s = Suite()
    s.set_train_set(get_labeled_collection())
    add_word2vec_suite(s)
    features = copy(s.features)

    score = find_score(features,copy(s))
    print("Word2vec: {}".format(score))

def add_tfidf():
    combo = ["perspective_score","stanford_polite"]

    s = Suite()
    s.set_train_set(get_labeled_collection())
    add_tf_idf_suite(s)
    features = copy(s.features)

    score = find_score(features,copy(s))
    print("Word2vec: {}".format(score))

def get_popular_parameters():
    s = Suite()
    s.set_train_set(get_labeled_collection())
    combo = ["perspective_score","stanford_polite"]

    find_score(combo,copy(s))
    best_parameters = s.most_popular
    print("Best parameters {}".format(best_parameters))

def add_context():
    s = Suite()
    s.set_train_set(get_labeled_collection())
    combo = ["perspective_score","stanford_polite"]
    s = add_context_suite(s,1)

    score = find_score(combo,copy(s))
    print("Context {}: ".format(score))

def internal_validation():
    add_features()
    add_lexicons()
    add_word2vec()
    add_tfidf()
    remove_features()
    remove_filters()

    get_popular_parameters()

def test_on(issue_collection,comment_collection,output_name):
    s = Suite()
    s.set_train_set(get_labeled_collection())
    s.set_test_set({'issues':db[issue_collection],'comments':db[comment_collection]})
    combo = ["perspective_score", "stanford_polite"]

    s.features = combo
    s.nice_features = combo
    s.add_parameter("gamma",[2])
    s.add_parameter("C",[0.05])
    s.set_ratios([2])

    s.set_model(svm_model)

    with open('{}_results.txt'.format(output_name.lower()), 'w') as f:
        with redirect_stdout(f):
            s.test_issue_classifications_from_comments_all()


def classify_language():
    for lang in ["Haskell","Python","Java","Javascript"]:
        s = Suite()
        s.set_train_set(get_labeled_collection())
<<<<<<< HEAD
        s.set_test_set({'issues':db["{}_issues".format(lang.lower())],'comments':db["{}_comments".format(lang.lower())]})
=======
#        s.set_test_set({'issues':db["{}_issues".format(lang.lower())],'comments':db["{}_comments".format(lang.lower())]})
>>>>>>> d83676441e3ec66c5f2686e1a432547f8254a5fc
        combo = ["perspective_score", "stanford_polite"]

        s.features = combo
        s.nice_features = combo
        s.add_parameter("gamma",[2])
        s.add_parameter("C",[0.05])
        s.set_ratios([2])

        s.set_model(svm_model)

<<<<<<< HEAD
=======
        import pickle
        pickle.dump(s,open("pretrained_model.p","wb"))

        return 0

>>>>>>> d83676441e3ec66c5f2686e1a432547f8254a5fc
        with open('{}_results.txt'.format(lang.lower()), 'w') as f:
            with redirect_stdout(f):
                s.test_issue_classifications_from_comments_all()

def classify_corporate():
    for lang in ["corporate","uncorporate"]:
        s = Suite()
        s.set_train_set(get_labeled_collection())
        s.set_test_set({'issues':db["{}_issues".format(lang.lower())],'comments':db["{}_comments".format(lang.lower())]})
        combo = ["perspective_score", "stanford_polite"]

        s.features = combo
        s.nice_features = combo
        s.add_parameter("gamma",[2])
        s.add_parameter("C",[0.05])
        s.set_ratios([2])

        s.set_model(svm_model)

        with open('{}_results.txt'.format(lang.lower()), 'w') as f:
            with redirect_stdout(f):
                s.test_issue_classifications_from_comments_all()

def classify_dates():
    dates = ['2012_01_09', '2012_02_13', '2012_03_12', '2012_04_09', '2012_05_14', '2012_06_11', '2012_07_09', '2012_08_13', '2012_09_10', '2012_10_08', '2012_11_12', '2012_12_10', '2013_01_14', '2013_02_11', '2013_03_11', '2013_04_08', '2013_05_13', '2013_06_10', '2013_07_08', '2013_08_12', '2013_09_09', '2013_10_14', '2013_11_11', '2013_12_09', '2014_01_13', '2014_02_10', '2014_03_10', '2014_04_14', '2014_05_12', '2014_06_09', '2014_07_14', '2014_08_11', '2014_09_08', '2014_10_13', '2014_11_10', '2014_12_08', '2015_01_12', '2015_02_09', '2015_03_09', '2015_04_13', '2015_05_11', '2015_06_08', '2015_07_13', '2015_08_10', '2015_09_14', '2015_10_12', '2015_11_09', '2015_12_14', '2016_01_11', '2016_02_08', '2016_03_14', '2016_04_11', '2016_05_09', '2016_06_13', '2016_07_11', '2016_08_08', '2016_09_12', '2016_10_10', '2016_11_14', '2016_12_12', '2017_01_09', '2017_02_13', '2017_03_13', '2017_04_10', '2017_05_08', '2017_06_12', '2017_07_10', '2017_08_14', '2017_09_11', '2017_10_09', '2017_11_13', '2017_12_11', '2018_01_08', '2018_02_12', '2018_03_12', '2018_04_09', '2018_05_14', '2018_06_11', '2018_07_09', '2018_08_13', '2018_09_10', '2018_10_08', '2018_11_12', '2018_12_10']
    for lang in dates:
        lang = lang.replace("_","-")
        s = Suite()
        s.set_train_set(get_labeled_collection())
        s.set_test_set({'issues':db["{}_issues".format(lang.lower())],'comments':db["{}_comments".format(lang.lower())]})
        combo = ["perspective_score", "stanford_polite"]

        s.features = combo
        s.nice_features = combo
        s.add_parameter("gamma",[2])
        s.add_parameter("C",[0.05])
        s.set_ratios([2])

        s.set_model(svm_model)

        with open('{}_results.txt'.format(lang.lower()), 'w') as f:
            with redirect_stdout(f):
                s.test_issue_classifications_from_comments_all()


<<<<<<< HEAD
=======
# Test wheter removing features has any effect
>>>>>>> d83676441e3ec66c5f2686e1a432547f8254a5fc
classify_language()
