# -*- coding: utf-8 -*-
"""
    Code for loading and working with pretrained word embeddings

"""
import numpy as np

def _parseline(l):
    cells = l.strip().split(" ")
    return cells[0], np.array(list(map(float, cells[1:])))


def embedding_loader(filename):
    """
    Load and parse a set of pretrained GloVe embeddings
    
    :filename: string; path to the saved embedding
    
    Returns a list of N tokens and an (N,d) array of embeddings
    """
    parsed = [_parseline(l) for l in open(filename, "r").readlines()]
    return [p[0] for p in parsed], np.array([p[1] for p in parsed]) 
