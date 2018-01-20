"""

		sequence.py


Tools for encoding text as a sequence of indices

"""

import numpy as np
import tensorflow as tf




def encode_fixed_length_sequence(docs, labels, shuffle=False, 
				num_epochs=1, batch_size=100, 
				token_list=None, name="x"):
	"""
	Have a list of arrays that are already the same length? Want
	an input function? Use this.
	
	This function is a very thin wrapper around 
	tf.estimator.inputs.numpy_input_fn
	
	:docs: list of arrays of the same length
	:labels: list or array of labels
	:shuffle: whether input function should be shuffled. should be
			false for evaluation or inference.
	:num_epochs: integer; number of epochs. should be 1 for evaluation.
	:batch_size: integer; number of records per batch
	:name: label for feature

	Returns
	:input_fn: a tf.estimator-compatible input function
	:features: a list of feature columns
	"""
	# find length of sequence
	seqlen = len(docs[0])
	# convert to a large dense matrix
	dense_corpus = np.stack(docs)
    
	input_fn = tf.estimator.inputs.numpy_input_fn(
		{name:dense_corpus}, y=np.array(labels),
		shuffle=shuffle, num_epochs=num_epochs,
		batch_size=batch_size
	)
	features = [tf.feature_column.numeric_column(name, shape=(seqlen,))]

	return input_fn, features


