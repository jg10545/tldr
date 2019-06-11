import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model
import holoviews as hv

def value_generator(*series):
    for s in series:
        for k,v in s.iteritems():
            yield(v)
            
            
def lasso_comparison(s1, s2, label1="corpus1", label2="corpus2", stopwords="english"):
    """
    
    """
    # build a labeled dataset
    y = np.concatenate([np.zeros(len(s1)), np.ones(len(s2))])
    alltext = pd.concat([s1,s2])
    
    vectorizer = CountVectorizer(max_df=0.95, min_df=10,
                                   max_features=50000,
                                   stop_words=stopwords,
                                    ngram_range=(1,3))
    # build a term-document matrix
    X = vectorizer.fit_transform(value_generator(alltext))
    
    model = sklearn.linear_model.LogisticRegressionCV(Cs=[1e-4,1e-3,1e-2,1e-1], cv=3, solver="saga", 
                                                  class_weight="balanced", penalty="l1", 
                                                  max_iter=250, tol=1e-3, n_jobs=2)
    model = model.fit(X,y)
    
    counts = np.array(X.sum(axis=0)).ravel()
    tokens = np.array(vectorizer.get_feature_names())
    coefs = model.coef_.ravel()
    nonzero = np.abs(coefs) > 0.00
    
    logres_df = pd.DataFrame({"x":counts[nonzero],
                  "y":coefs[nonzero],
                  "token":tokens[nonzero]})
    
    return hv.Labels(logres_df, ["x", "y"]).opts(hv.opts.Labels(xlabel="counts", 
                                                     ylabel="%s <---> %s"%(label1, label2)))