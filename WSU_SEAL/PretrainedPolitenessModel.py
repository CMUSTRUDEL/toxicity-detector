import sys
import os
import _pickle
import scipy
import numpy as np
from scipy.sparse import csr_matrix

import vectorizer

MODEL_FILENAME = os.path.join(os.path.split(__file__)[0], 'models/wsu-seal-retrain-politeness-svm.p')

# Load model, initialize vectorizer
clf = _pickle.load(open(MODEL_FILENAME, 'rb'), encoding='latin1', fix_imports=True)
vectorizer = vectorizer.PolitenessFeatureVectorizer()

def score(request):
    """
    :param request - The request document to score
    :type request - dict with 'sentences' and 'parses' field
        sample (taken from test_documents.py)--
        {
            'sentences': [
                "Have you found the answer for your question?",
                "If yes would you please share it?"
            ],
            'parses': [
                ["csubj(found-3, Have-1)", "dobj(Have-1, you-2)",
                 "root(ROOT-0, found-3)", "det(answer-5, the-4)",
                 "dobj(found-3, answer-5)", "poss(question-8, your-7)",
                 "prep_for(found-3, question-8)"],
                ["prep_if(would-3, yes-2)", "root(ROOT-0, would-3)",
                 "nsubj(would-3, you-4)", "ccomp(would-3, please-5)",
                 "nsubj(it-7, share-6)", "xcomp(please-5, it-7)"]
            ]
        }

    returns class probabilities as a dict
        { 'polite': float, 'impolite': float }
    """
    # Vectorizer returns {feature-name: value} dict
    features = vectorizer.features(request)
    fv = [features[f] for f in sorted(features.keys())]
    # Single-row sparse matrix
    X = csr_matrix(np.asarray([fv]))
    probs = clf.predict_proba(X)
    # Massage return format
    probs = {"polite": probs[0][1], "impolite": probs[0][0]}
    return probs
