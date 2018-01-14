"""

		character.py


"""

import numpy as np

from tldr.prepare import Bagginator



def character_tokenizer(x):
	"""
	Just a generic character tokenizer that doesn't
	filter anything out. Note that the LeCunn paper
	restricts the alphabet.
	"""
	return list(x.lower().strip())



class FixedCharacterSequencer(tldr.prepare.Bagginator):
	"""
	Text preparer class for when we want a fixed-length sequence
	of characters (like for the LeCunn CNN model).

	Note that this will return 1D arrays of length [length], with
	each value an integer between 0 and numtokens-1. A network
	that inputs these will have to start by using a one-hot encoder
	to convert batches of these to a [batchsize, length, numtokens]
	tensor.

	Strings longer than the specified length will be truncated; strings
	shorter than the length will be left-padded.
	"""
    
	def __init__(self, corpus, length, padchar=" ", 
			tokenizer=character_tokenizer):
		"""
		:corpus: list or iterator of strings, each containing 
			a document
		:length: int, length of feature arrays to generate.
		:padchar: string, character used to left-pad short
			documents. assumes that it exists in the corpus.
		:tokenizer: tokenizing function to use.
		"""
		self.tokenize = tokenizer
		self.token_list = self._make_tokenlist(corpus, minlen=1)
		self.token_index = self._make_index(self.token_list)
		self._numtokens = len(self.token_index)
		elf._pad_index = self.token_index[padchar]
		self._length = length
        
	def __call__(self, s):
		"""
		Input a string, get an array of indices
		"""
		outarr = np.array([self._pad_index]*self._length, dtype=int)
		tokens = self.tokenize(s)
		tokens = [self.token_index[t] for t in tokens
				if t in self.token_index][:self._length]
		outarr[-len(tokens):] = np.array(tokens)
		return outarr
