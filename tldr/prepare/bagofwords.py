"""

			bagofwords.py


In this file- some generic tools for preprocessing text bag-of-words style.

"""

import string


def generic_tokenizer(x):
	"""
	Nothing fancy here.
	"""
	x = x.lower().strip()
	for c in string.digits+string.punctuation:
		x = x.replace(c, " ")
	return x.split()
    


class Bagginator(object):
	"""
	Boring prototype class for managing conversion between raw strings, 
	tokens, and indices.
	"""
    
	def __init__(self, corpus, tokenizer=generic_tokenizer, minlen=2):
		"""
		:corpus: list or iterator of strings, each containing a document
		:tokenizer: function that maps a string to a list of tokens
		:minlen: minimum length for tokens
		"""
		self.tokenize = tokenizer
		self.token_list = self._make_tokenlist(corpus, minlen=minlen)
		self.token_index = self._make_index(self.token_list)
		self._numtokens = len(self.token_index)

        
	def _make_tokenlist(self, corpus, minlen=2):
		"""
		Input a list of strings representing the documents in the
		corpus; return a list of all the distinct words in the corpus
		"""
		return list(set([token for doc in corpus 
				for token in self.tokenize(doc)
				if len(token) >= minlen]))

	def _make_index(self, tokenlist):
		"""
		invert the token list to get a dictionary, where each
		key is a token and each value is the token's index
		"""
		return {tokenlist[i]:i for i in range(len(tokenlist))}
 
	def __len__(self):
		return self._numtokens

	def __call__(self, s):
		"""
		Input a string, return a list of indices
		"""
		tokens = self.tokenize(s)
		return [self.token_index[t] for t in tokens 
				if t in self.token_index]
    
	def __getitem__(self, indices):
		"""
		Input a list of indices, return the associated tokens
		"""
		return [self.token_list[i] for i in indices 
				if i < self._numtokens]



