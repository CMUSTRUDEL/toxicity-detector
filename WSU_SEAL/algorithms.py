from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# to avoid import errors these methods are copied from src.classifiers.
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
