from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from statistics import calculate_statistics
import time

def df_to_list(df):
    return [list(x) for x in df.values]

def fit_model(model,train,test,features_used):
    train_tuples = [tuple(x) for x in train[features_used].values]
    train_label = train['toxic'].tolist()

    model.fit(train_tuples,train_label)

    return model

def classify(model,train,test,features_used):
    t = time.time()

    model = fit_model(model,train,test,features_used)
    t = time.time()

    test_list = df_to_list(test[features_used])

    t = time.time()
    predicted = model.predict(test_list)
    test['prediction'] = predicted

    return test

def classify_proba(model,train,test,features_used):
    model = fit_model(model,train,test,features_used)

    test_list = df_to_list(test[features_used])
    predicted = model.predict_proba(test_list)
    test['probability'] = [i[1] for i in predicted]
    test = test.sort_values('probability',ascending=False)

    return test

def classify_statistics(model,train,test,features):
    predicted = classify(model,train,test,features)['prediction'].tolist()
    test_label = test['toxic'].tolist()
    return calculate_statistics(predicted,test_label)

def bayes_model():
    """Bayes"""
    return GaussianNB()

def linear_svm_model(C=10**1.5):
    """Linear SVM"""
    return svm.LinearSVC(C=C,max_iter=10000)

def svm_model(C=10**1.5,gamma='scale'):
    """SVM"""
    return svm.SVC(gamma=gamma,C=C,probability=True)

def logistic_model(C=1):
    """Logistic"""
    return LogisticRegression(C=C,solver='lbfgs',multi_class='multinomial',max_iter=4000)

def decision_tree_model():
    """Decision Tree"""
    return tree.DecisionTreeClassifier()

def random_forest_model(n_estimators=100,max_features="auto",max_depth=None,min_samples_leaf=1):
    """Random Forest"""
    return RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,min_samples_leaf=min_samples_leaf)

def knn_model(k=5):
    """KNN"""
    return KNeighborsClassifier(n_neighbors=k)

def get_coef(model):
    coef = model.coef_[0]
    return coef
