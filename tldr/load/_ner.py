# -*- coding: utf-8 -*-
"""
                    _ner.py
                    
Code for preparing input functions for NER data.

Based on model in "End-to-end Sequence Labeling via Bi-directional
LSTM-CNN-CRF by Ma and Hovy (2016).

"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib

# my choice of integer label for each CONLL2003
labelmap = {"O":0, "B-PER":1, "I-PER":2, "B-ORG":3, "I-ORG":4, 
            "B-LOC":5, "I-LOC":6, "B-MISC":7, "I-MISC":8, "PAD":9}


def _prep_doc(d):
    """
    Pull out the tokens and NER labels for a single record
    """
    # extract tokens
    tokens = [x.split(" ")[0] for x in d.split("\n")]
    # extract labels
    labels = [labelmap[x.split(" ")[-1]] for x in d.split("\n")]
    return tokens, labels


def _pad_doc(t, l, n=50):
    """
    Pad or truncate a list of tokens and list of labels
    to a fixed length
    """
    if len(t) > n:
        return t[:n], l[:n]
    elif len(t) < n:
        d = n-len(t)
        return t + [" "]*d, l + [9]*d
    else:
        return t,l
    
def _pad_token(t, n=10):
    """
    Pad or truncate a single token to a fixed length
    """
    if len(t) > n:
        return t[:n]
    elif len(t) < n:
        return t + " "*(n-len(t))
    else:
        return t
    
    
def parse_doc(d, num_tokens=50, token_length=15):
    """
    The one-stop shop for parsing CONLL2003 records
    
    :d: string; a single record
    :num_tokens: number of tokens to pad/truncate each sentence to
    :token_length: length to pad/truncate each token to for the
                    character-level representation
    """
    # pull out tokens and labels
    toks, labs = _prep_doc(d)
    # pad each sentence to the same length
    toks, labs = _pad_doc(toks, labs, num_tokens)
    # get padded tokens for character-level representation
    chars = [_pad_token(t, token_length) for t in toks]
    # turn everything into an array. the expand_dims makes sure
    # that embedding tools in tf.feature_columns won't smoosh
    # them all together.
    toks = np.expand_dims(np.array([t.lower() for t in toks]), -1)
    chars = np.expand_dims(np.array([list(c) for c in chars]), -1)
    labs = np.array(labs, dtype=np.int32)
    return {"tokens":toks, "chars":chars, "num_tokens":num_tokens}, labs



def conll_input_fn(docs, num_tokens=50, token_length=15, repeat=1, 
                   shuffle=False, batch_size=10, prefetch=5):
    """
    Train/test input function generator for named entity recognition.
    
    :docs: list of strings each containing one CONLL2003-formatted entry
    :num_tokens: number of tokens to pad/truncate each sentence to
    :token_length: number of characters to pad/truncate each token to 
                    for the character-level representation
    :repeat: number of times to repeat dataset (1 for test)
    :shuffle: whether to shuffle dataset (False for test)
    :batch_size: number of records to include per batch
    """
    def _gen():
        for d in docs:
            yield parse_doc(d, num_tokens, token_length)
    ds = tf.data.Dataset.from_generator(_gen, 
                                        ({"tokens":tf.string, 
                                          "chars":tf.string,
                                         "num_tokens":tf.int32}, tf.int32),
                                       ({"tokens":tf.TensorShape([num_tokens,1]),
                                        "chars":tf.TensorShape([num_tokens, 
                                                                token_length,1]),
                                        "num_tokens":tf.TensorShape([])},
                                       tf.TensorShape(num_tokens)))
    ds = ds.repeat(repeat)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(prefetch)

    def _input_fn():
        return ds.make_one_shot_iterator().get_next()
    return _input_fn

#    return ds.make_one_shot_iterator().get_next()





