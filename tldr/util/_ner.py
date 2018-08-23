# -*- coding: utf-8 -*-
from tldr.load._ner import _sentparse, token_predict_input_fn
from IPython.display import HTML, display

colmap = {1:"green", 2:"green", 3:"blue", 4:"blue", 5:"yellow", 6:"yellow",
          7:"fuchsia", 8:"fuchsia"}

def _build_colorstring(sent, tags):
    tokens = _sentparse(sent)
    outstr = ""
    for i in range(len(tokens)):
        if tags[i] in colmap:
            outstr += "<b><text style=color:%s>"%colmap[tags[i]] + tokens[i] + "</text></b>"
        else:
            outstr += tokens[i] + " "
    for c in ".,;?!":
        outstr = outstr.replace(" "+c, c)
    return outstr.strip()



def tagprint(model, sent):
    """
    Query a model for entities and display results in a Jupyter notebook.
    
    :model: trained NER model
    :sent: string; sentence to pass to model
    """
    inpt = token_predict_input_fn(sent)
    tags = next(model.predict(inpt))["decode_tags"]
    display(HTML(_build_colorstring(sent, tags)))