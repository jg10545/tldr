"""

entities.py

Just some spacy wrappers

"""
import pandas as pd
import spacy
from tqdm import tqdm


def _doc_dict(d, types=["PERSON", "NORP", "FAC", "ORG", "GPE", 
                       "LOC", "PRODUCT", "EVENT", "LAW"]):
    outdict = {t:[] for t in types}
    
    for e in d.ents:
        if e.label_ in types:
            text = e.text
            for s in ['“', "'", "the", "The", '"', '”']:
                text = text.replace(s,"")
            text = text.strip()
            if (text not in outdict[e.label_])&(len(text) >1):
                outdict[e.label_].append(text)
    return outdict



def extract_entities(content, model=None, types=["PERSON", "NORP", "FAC", 
                    "ORG", "GPE", "LOC", "PRODUCT", "EVENT", 
                                       "LAW"]):
    """
    Extract entities and return as a dataframe
    
    :content: iterable of strings containing text to extract
    :model: Spacy NLP model 
    """
    if model is None:
        model = spacy.load("en")
    docslist = [model(c) for c in tqdm(content, desc="extracting entities")]
    output = pd.DataFrame([_doc_dict(x, types) for x in tqdm(docslist, desc="formatting")])
    output.columns = ["ENTITY_%s"%x for x in list(output.columns)]
    return output