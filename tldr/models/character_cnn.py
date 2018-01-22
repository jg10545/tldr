"""

		character_cnn.py


This module contains code to implement the models described in "Character-level
Convolutional Networks for Text Classification" by Zhang, Zhao and Lecun.

The model is a curiously simple approach to text categorization- rather than bother with tokenizing the text, selecting words, applying stemming and stopword operations, and then choosing a vector embedding, it simply treats text as a sequence of characters and attempts to learn the rest from scratch.

"""

import tensorflow as tf


# The paper contains two variants of the model; here are
# all the filters/neurons for each:
conv_features = {
	"small":[
		(256, 7, True),
		(256, 7, True),
		(256, 3, False),
		(256, 3, False),
		(256, 3, False),
		(256, 3, True)
		],
	"large":[
		(1024, 7, True),
		(1024, 7, True),
		(1024, 3, False),
		(1024, 3, False),
		(1024, 3, False),
		(1024, 3, True)
		]   
}

dense_features = {
	"small":[1024, 1024],
	"large":[2048, 2048]
}


def model_fn(features, labels, mode, params):
	"""
	Model function for the LeCun CNN. Expects the 
	following params:

	:size: "small" or "large", which model to build
	:num_tokens: number of distinct character tokens in
		our data (e.g. the number of dimensions to 
		use for one-hot encoding the inputs)
	:num_classes: number of distinct classes
	:learning_rate: initial learning rate for MomentumOptimizer
	:decay_steps:
	:decay_rate:
	:momentum: momentum for MomentumOptimizer
	:dropout_prob: dropout probability for dense layers
	"""
	# pick out which version of the network we're building
	conv = conv_features[params["size"]]
	dense = dense_features[params["size"]]
    
	# first- expand the input data to one-hot encoding
	with tf.name_scope("input"):
		net = tf.one_hot(features["x"], params["num_tokens"], 
				name="one_hot")

	# add the convolutional layers one at a time
	for n, (features, kernel, pool) in enumerate(conv):
		net = tf.layers.conv1d(net, features, kernel,
					activation=tf.nn.relu,
					name="convlayer_%s"%n)
		if pool:
			net = tf.layers.max_pooling1d(net, 3, 3,
					name="maxpool%s"%n)

	# flatten and add dense layers
	net = tf.contrib.layers.flatten(net)
	for n, f in enumerate(dense):
		net = tf.layers.dense(net, f, activation=tf.nn.relu,
					name="denselayer_%s"%n)
		net = tf.layers.dropout(net, rate=params["dropout_prob"],
					name="dropout_%s"%n)

    
	final = tf.layers.dense(net, params["num_classes"], name="final")
	predictions = tf.argmax(final, axis=1, name="predictions")

	# Provide an estimator spec for `ModeKeys.PREDICT`.
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions={"class": predictions})

	# since we're doing categorization, use softmax loss
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final, 
			labels=tf.one_hot(labels, params["num_classes"])))

	# Calculate root mean squared error as additional eval metric
	eval_metric_ops = {
		"accuracy":tf.metrics.accuracy(labels, predictions),
		"auc":tf.metrics.auc(labels, predictions)
		}

	# Set up a momentum optizimer with exponentially decaying learning rate
	lr = tf.train.exponential_decay(params["learning_rate"],
					tf.train.get_global_step(),
					params["decay_steps"],
					params["decay_rate"],
					name="learning_rate")
	optimizer = tf.train.MomentumOptimizer(lr, params["momentum"])
	train_op = optimizer.minimize(
		loss=loss, global_step=tf.train.get_global_step())

	# Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, train_op=train_op,
		eval_metric_ops=eval_metric_ops)


def  LeCunCharacterCNN(size="small", num_tokens=54, num_classes=7,
			learning_rate=1e-3, decay_steps=10000, decay_rate=0.2,
			momentum=0.9, dropout_prob=0.5, model_dir=None):
	"""
	Macro to build the CNN as a TensorFlow Estimator.

	:size: "small" or "large", which model to build
	:num_tokens: number of distinct character tokens in
		our data (e.g. the number of dimensions to 
		use for one-hot encoding the inputs)
	:num_classes: number of distinct classes
	:learning_rate: learning rate for MomentumOptimizer
	:momentum: momentum for MomentumOptimizer
	:dropout_prob: dropout probability for dense layers
	:model_dir: where to save model
	"""
	params = {"size":size, "num_tokens":num_tokens, 
		"num_classes":num_classes, "learning_rate":learning_rate,
		"decay_steps":decay_steps, "decay_rate":decay_rate,
		"momentum":momentum, "dropout_prob":dropout_prob}
	return tf.estimator.Estimator(model_fn, params=params,
					model_dir=model_dir)








