# -*- coding: utf-8 -*-
"""

                _seqlab.py

Implementation of the sequence-labeling RNN for text from Ma and Hovy's 2016
paper "End-to-end Sequence Labeling via Bi-directional LSTM-CN-CRFs".

"""
import tensorflow as tf

def _prepare_input_tensors(x, char_list, token_list, char_embed_dim=30, 
                           token_embed_dim=100):
    """
    TO DO come up with a slick way to include pretrained glove or word2vec 
    vectors.
    
    Function to handle the setup of the embeddings.
    
    :x: estimator inputs- should be a dictionary containing tensors "chars" 
            and "tokens"
    :char_list: list of strings for every character in the lookup
    :token_list: list of strings for every token in the lookup
    :char_embed_dim: int; dimension of character embedding
    :token_embed_dim: int; dimension of token embedding
    
    Returns 
    -------
    :char_embed_tensor: [None, num_tokens, num_chars, char_embed_dim] tensor
    :tok_embed_tensor: [None, num_tokens, token_embed_dim] tensor
    """
    # infer shape from the input tensors
    dims = x["chars"].get_shape().as_list()
    num_tokens = dims[1]
    token_length = dims[2]
                           
    # CHARACTER LEVEL EMBEDDING
    char_col = tf.feature_column.categorical_column_with_vocabulary_list(
                            "chars", char_list)
    char_embed_col = tf.feature_column.embedding_column(
                            char_col, char_embed_dim, trainable=True)
    char_embed_tensor = tf.reshape(tf.feature_column.input_layer(
                            x, [char_embed_col]),
                            [-1, num_tokens, token_length, char_embed_dim])
    
    # WORD LEVEL EMBEDDING
    tok_col = tf.feature_column.categorical_column_with_vocabulary_list(
                            "tokens", token_list)
    # THIS NEEDS TO CHANGE: trainable to False and 
    # include initializer=tf.constant_initializer()
    tok_embed_col = tf.feature_column.embedding_column(
                            tok_col, token_embed_dim, trainable=True)
    tok_embed_tensor = tf.reshape(tf.feature_column.input_layer(
                            x, [tok_embed_col]),
                            [-1, num_tokens, token_embed_dim])
    return char_embed_tensor, tok_embed_tensor

