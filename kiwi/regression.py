import numpy as np
from .tools import sigmoid
from tqdm import tqdm


class GradientDescent:

	def __call__(self, a):
		return self.f(a)

	def __init__(self, in_array, out_array):
		self.f = self.train(in_array, out_array)

	def train(self, in_array, out_array):
		in_width = len(in_array[0])
		in_length = len(in_array)
		out_width = len(out_array[0])
		out_length = len(out_array)

		np.random.seed(1)

		syn0 = 2 * np.random.random((in_width, in_length)) - 1
		syn1 = 2 * np.random.random((out_length, out_width)) - 1

		in_array = np.asarray(in_array, dtype=np.float32)

		for j in range(5000):
			l0 = in_array
			l1 = sigmoid(np.dot(l0, syn0))
			l2 = sigmoid(np.dot(l1, syn1))
			l2_error = out_array - l2

			# if (j% 1000) == 0:
			#     print("Error:" + str(np.mean(np.abs(l2_error))))

			l2_delta = l2_error * sigmoid(l2, deriv=True)
			l1_error = l2_delta.dot(syn1.T)
			l1_delta = l1_error * sigmoid(l1, deriv=True)

			syn1 += l1.T.dot(l2_delta)
			syn0 += l0.T.dot(l1_delta)
		
		return lambda a: sigmoid(np.dot(sigmoid(np.dot(a, syn0)), syn1))
