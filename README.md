<<<<<<< HEAD
tl;dr


=======
# tldr

Convenience functions for getting text data into TensorFlow.


## Goals

 * **Short-term:** convenience code to generate input functions compatible with TensorFlow's `estimator` interface, focusing on text datasets that are small enough to fit into memory.
 * **Longer-term:** generating input functions from larger datasets, custom `estimator` models.


## Components

`tldr` is broken into three submodules, each for one step

 1. `tldr.load:` Tools for getting data loaded from raw files (or from generic functions like `pandas.read_csv`)- differences will be in **the format of the raw data**. Functions in this submodule should return a list or numpy array of strings, each representing a single record. if appropriate, returns a list or numpy array of labels as well.

 2. `prepare:` Tools for preprocessing text- differences will be in **language and application-specific needs.** Things like stemming and stopwords would be dealt with here. Functions should take inputs from `tldr.load` functions. Functions should output an object that maps a string to a list or array of tokens orintegers (representing the one-hot index for each token).

 3. `encode:` Tools for reformatting the list of integers into the shape needed for the `estimator` API- differences will be in **the type of model** (i.e. whether it inputs a bag-of-words vector, a sequence, etc). Functions should output a TF input function and list of feature columns (the latter is needed for specifying a `tf.estimator` model; the former is needed for training/evaluation)


>>>>>>> 53742e8242625172fbd41b2fd6993b69739896b8
