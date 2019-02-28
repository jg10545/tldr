"""

_decomp.py


Matrix decomposition-related 


"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE


def nmf_topic_model(content, max_features=20000, n_components=25, **kwargs):
    # build a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=max_features,
                                   stop_words='english')
    # build term-document matrix
    tdm = vectorizer.fit_transform(content)
    
    # factorize matrix
    nmf = NMF(n_components=n_components)
    tdm_factorized = nmf.fit_transform(tdm)
    
    # assemble outputs
    components = pd.DataFrame(nmf.components_.T, index=vectorizer.get_feature_names())
    
    return tdm_factorized, components


def add_tsne_topic_embedding(d, vectors, n_iter=1000, **kwargs):
    """
    Embed a vector for each document (for example, from NMF) as an "x" and "y" column.
    """
    d = d.copy()
    embedding = TSNE(n_components=2, n_iter=1000, **kwargs).fit_transform(vectors)
    d["x"] = embedding[:,0]
    d["y"] = embedding[:,1]
    d["topic"] = vectors.argmax(axis=1)
    
    return d
