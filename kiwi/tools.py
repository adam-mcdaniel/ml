import numpy as np
import dill as pickle


def sigmoid(x, deriv=False, ceiling=1):
	if(deriv == True):
	    return x * ((1.0 / float(ceiling)) - x)
	return 1 / ((1.0 / float(ceiling)) + np.exp(-x))


def save(n, filename):
	with open(filename, "wb") as f:
		pickle.dump(n, f)
		f.close()


def read(filename):
	with open(filename, "rb") as f:
		n = pickle.load(f)
		f.close()
	return n


def batch(n): return list(map(lambda a: a / n, list(map(float, range(0, n)))))


def round_array(arr): return np.vectorize(lambda a: round(a))(arr)