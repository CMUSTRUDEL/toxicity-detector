from sklearn.model_selection import KFold
from get_data import *
from classifiers import *
from text_modifier import *

def cross_validate(data,features,model,num_folds=5):
    data = data.sample(frac=1)
    kfold = KFold(num_folds)

    all_stats = {}

    for train, test in kfold.split(data):
        train_data = data.iloc[train].copy()
        test_data = data.iloc[test].copy()

        scores = classify_statistics(model,train_data,test_data,features)

        if all_stats == {}:
            all_stats = scores
        else:
            for i in scores:
                all_stats[i]+=scores[i]

    for i in all_stats:
        all_stats[i]/=num_folds

    return all_stats

def cross_validate_classify(data,features,model,num_folds=5):
    kfold = KFold(num_folds)
    data = data.sample(frac=1)

    for train, test in kfold.split(data):
        train_data = data.iloc[train].copy()
        test_data = data.iloc[test].copy()

        test_data = classify(model,train_data,test_data,features)

        for i,row in test_data.iterrows():
            data.loc[data["_id"] == row["_id"],"prediction"] = row["prediction"]

    return data


def cross_validate_multiple(data,features,model,num_folds=5,num_trials=20):
    mean = 0
    for i in range(num_trials):
        mean+=cross_validate(data,features,model,num_folds=num_folds)['accuracy']

    return mean/num_trials


if __name__ == "__main__":
    train_collection = get_labeled_collection()
    train_data = get_issues(train_collection)
    train_data = map_toxicity(train_data)

    top_scores = 10
    features = append_to_str_multiple([("word2vec_",300),
                                       ("min_perspective_score_",top_scores),
                                       ("min_nltk_score_",top_scores),
                                       ("min_polarity_",top_scores)])

    train_data = add_comment_score(train_data,train_collection,top_scores)

    train_data["total_text"] = train_data["total_text"].map(cleanup_text)
    train_data = add_word2vec(train_data)

    model_list = [logistic_model(),bayes_model(),svm_model(),decision_tree_model(),random_forest_model(),knn_model(k=10)]

    for model in model_list:
        print(cross_validate_multiple(train_data,features,model))

