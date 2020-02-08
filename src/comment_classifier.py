import time
start = time.time()
from suite import Suite
from classifiers import *
from get_data import *

print("Finished imports, took {} time".format(time.time()-start))

s = Suite()

s.set_train_set({'issues':db["naveen_issues_backup"],'comments':db["naveen_comments_backup"],'name':'labeled'})
s.set_test_set({'issues':db["naveen_toxic_issues"],'comments':db["naveen_toxic_comments"],'name':'toxic-2'})
s.set_model(svm_model)

s.set_ratios([2])
s.add_parameter("C", [.05])
s.add_parameter("gamma", [2])

s.features = ["perspective_score","stanford_polite"]
s.nice_features = ["perspective_score","stanford_polite"]

s.self_issue_classification_all()
s.test_issue_classifications_from_comments_all()