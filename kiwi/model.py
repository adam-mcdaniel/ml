import numpy as np

class Dataset(np.ndarray):
	def __init__(self, width):
		self.dataset = np.array(list(map(lambda a: 0, list(range(0, width)))))

	def add(self, in_array):
		self.dataset = np.vstack((self.dataset, in_array))

	def __call__(self):
		return self.dataset[1:]

	def __str__(self):
		return np.array_str(self.dataset[1:])

def EmptyDataset(width):
	d = Dataset(width)
	d.add(list(map(lambda a: 0, list(range(width)))))
	return d