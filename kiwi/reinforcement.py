from .tools import *
from .model import Dataset
from .model import EmptyDataset
from .regression import GradientDescent

class Agent:
	def __init__(self, State0):
		self.state = State0

	def update(self, *args):
		pass

	def get(self):
		return self.state


class KiwiLearning:
	def __init__(self, in_array_size, out_array_size, reward_function, **kwargs):
		input_data = EmptyDataset(in_array_size)
		ouput_data = EmptyDataset(out_array_size)

		self.network = GradientDescent(input_data(),
									   ouput_data())

		self.state_dataset = Dataset(in_array_size)
		self.predicted_rewards_dataset = Dataset(out_array_size)

		self.reward_function = reward_function
		self.out_functions = kwargs["actions"]
		self.score = 0

	def train(self, state):
		predicted_rewards = self.network(state)

		max_val_indice = max([(v,i) for i,v in enumerate(predicted_rewards)])[1]
		new_state = self.out_functions[max_val_indice]()
		
		actual_rewards = self.reward_function(new_state)

		self.state_dataset.add(state)
		self.predicted_rewards_dataset.add(actual_rewards)

		self.network = GradientDescent(self.state_dataset(),
									   self.predicted_rewards_dataset())

		self.score += sum(self.reward_function(new_state))

		return new_state, predicted_rewards, actual_rewards

	def update(self, state):
		predicted_rewards = self.network(state)

		max_val_indice = max([(v,i) for i,v in enumerate(predicted_rewards)])[1]
		new_state = self.out_functions[max_val_indice]()

		actual_rewards = self.reward_function(new_state)

		self.score += sum(self.reward_function(new_state))

		return new_state, predicted_rewards, actual_rewards

	def data(self):
		return self.state_dataset.__str__(), self.predicted_rewards_dataset.__str__()

	def __str__(self):
		score = self.score
		self.score = 0
		return "score={}".format(score)