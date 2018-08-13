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
    with tf.name_scope("char_embedding"):
        char_col = tf.feature_column.categorical_column_with_vocabulary_list(
                            "chars", char_list)
        char_embed_col = tf.feature_column.embedding_column(
                            char_col, char_embed_dim, trainable=True)
        char_embed_tensor = tf.reshape(tf.feature_column.input_layer(
                            x, [char_embed_col]),
                            [-1, num_tokens, token_length, char_embed_dim])
    
    # WORD LEVEL EMBEDDING
    with tf.name_scope("token_embedding"):
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

def _character_cnn(inpt, window_size=3, num_filters=30, dropout_prob=0.5, training=True):
    """
    Function to build the character-level CNN
    
    :inpt: the character embedding tensor; [batch_size, num_tokens, token_length, token_embed_dim]
    :window_size: size of the convolutional kernel
    :num_filters: number of convolutional kernels
    :dropout_prob: dropout probability. 0.1 would drop out 10% of inputs.
    :training: Boolean; whether this is a training graph (used for dropout)
    
    Returns a [batch_size, num_tokens, num_filters] tensor
    """
    with tf.name_scope("char_cnn"):
        conv = tf.layers.conv2d(inpt, num_filters, [1, window_size], activation=tf.nn.relu)
        conv_size = conv.get_shape().as_list()[2]
        maxpool = tf.layers.max_pooling2d(conv, [1, conv_size], 1)
        # squeeze the extra dimension out
        maxpool = tf.squeeze(maxpool, 2)
        dropout = tf.layers.dropout(maxpool, rate=dropout_prob, training=training)
    return dropout




def _bidirectional_rnn(inpt, state_size=200, dropout_prob=0.5, training=False, num_labels=10):
    """
    Put the Bidirectional LSTM together.
    
    :inpt: input sequence- should be [batch_size, num_tokens, ?]
    :state_size: number of hidden neurons in each LSTM cell
    :dropout_prob: dropout probability. 0.1 would drop out 10% of inputs.
    :training: Boolean; whether this is a training graph (used for dropout)
    :num_labels: number of output labels for sequence classification
    
    Returns a [batch_size, num_tokens, num_labels] tensor of logits
    """
    with tf.name_scope("rnn"):
        # build forward and backward cells
        fw_cell = tf.nn.rnn_cell.LSTMCell(state_size)
        bw_cell = tf.nn.rnn_cell.LSTMCell(state_size)
        # combine into a bidirectional RNN
        rnn_output, rnn_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, rnn_inputs, 
                                                            dtype=tf.float32)
        dropout = tf.layers.dropout(tf.concat(rnn_output, 2), dropout_prob, training=training)
        logits = tf.layers.dense(dropout, num_labels, name="logits")
    return logits




def model_fn(features, labels, mode, params):
    """
    """
    labels_oh = tf.one_hot(labels, params["num_labels"])
    training = mode == tf.estimator.ModeKeys.TRAIN
    
    # assemble the embeddings
    tok_embed_tensor, char_embed_tensor = _prepare_input_tensors(
                                            features, params["char_list"], 
                                            params["token_list"], 
                                            params["char_embed_dim"],
                                            params["token_embed_dim"])
    # build the character CNN
    cnn_output = _character_cnn(char_embed_tensor, params["window_size"], params["num_filters"], 
                                params["dropout_prob"], training)
    # concatenate token embeddings with CNN outputs
    rnn_inputs = tf.concat([tok_embed_tensor, cnn_output], 2)
    # feed all that into the LSTM
    rnn_output = _bidirectional_rnn(rnn_inputs, params["state_size"], 
                                    params["dropout_prob"], 
                                    training, params["num_labels"])
    
    predictions = tf.argmax(rnn_output, axis=2, name="predictions")
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"class":predictions}
        )
    
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(labels_oh, predictions))
    eval_metric_ops = {
        "accuracy":tf.metrics.accuracy(labels, predictions)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops=eval_metric_ops
        )
    
    optimizer = tf.train.MomentumOptimizer(params["learning_rate"], params["momentum"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )
    

def SequenceClassifier(model_dir, token_list, char_list, num_labels=10,
                       char_embed_dim=30, token_embed_dim=100,
                       window_size=3, num_filters=30, dropout_prob=0.5, 
                       state_size=200, learning_rate=1e-3,
                       warm_start_from=None):
    """
    STUFF
    """
    params = {"token_list":token_list, "char_list":char_list, 
              "num_labels":num_labels, "char_embed_dim":char_embed_dim,
              "token_embed_dim":token_embed_dim, "window_size":window_size,
              "num_filters":num_filters, "dropout_prob":dropout_prob,
              "state_size":state_size, "learning_rate":learning_rate}
    return tf.estimator.Estimator(model_fn, model_dir=model_dir, 
                                  params=params, 
                                  warm_start_from=warm_start_from)

