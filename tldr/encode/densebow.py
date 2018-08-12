"""

		densebow.py


Inelegant code for dense bag-of-words encodings.

"""

import numpy as np
import tensorflow as tf

def densify(x, N):
	"""
	input a list of token indices and a vector length; return a dense array
	"""
	dense = np.zeros(N, dtype=int)
	dense[np.array(x)] = 1
	return dense


def encode_dense_bag_of_words(doc_indices, labels, N, shuffle=False, 
				num_epochs=1, name="x"):
	"""
	Encode a corpus of tokenized, indexed documents into a big
	dense term-document matrix.

	:doc_indices: list or array of indices for each document
	:labels: list or array of labels
	:N: total size of corpus
	:shuffle: whether input function should be shuffled. should be
			false for evaluation or inference.
	:num_epochs: number of epochs. should be 1 for evaluation.
	:name: label for feature

	Returns
	:input_fn: a tf.estimator-compatible input function
	:features: a list of feature columns
	"""
	# convert to a large dense matrix
	dense_corpus = np.stack([densify(d, N) for d in doc_indices])
    
	input_fn = tf.estimator.inputs.numpy_input_fn(
		{name:dense_corpus}, y=np.array(labels),
		shuffle=shuffle, num_epochs=num_epochs
	)
	features = [tf.feature_column.numeric_column(name, shape=(N,))]
	return input_fn, features
