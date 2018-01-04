import unittest
from tldr.prepare.bagofwords import generic_tokenizer, Bagginator


class TestTokenizer(unittest.TestCase):
	
	def setUp(self):
		self.testdoc = "Who's on first?"

	def test_tokenizer_returns_correct_answer(self):
		tokens = generic_tokenizer(self.testdoc)
		self.assertEqual(tokens, ["who", "s", "on", "first"])



class TestBagginator(unittest.TestCase):

	def setUp(self):
		self.testcorpus = [
			"Somebody once told me,",
			"the world is gonna roll me.",
			"(I ain't the sharpest tool in the shed)"
				]
		self.Bagginator = Bagginator(self.testcorpus, minlen=2)

	def test_bagginator_length_returns_correct_answer(self):
		N = len(self.Bagginator)
		print(N)
		print(self.Bagginator.token_list)
		self.assertEqual(N, 14)


	def test_bagginator_returns_right_number_of_indices(self):
		indices = self.Bagginator(self.testcorpus[2])
		self.assertEqual(len(indices), 7)


if __name__=="__main__":
	unittest.main()
