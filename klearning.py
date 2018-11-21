import time
from tqdm import *
from kiwi.tools import *
from kiwi.reinforcement import Agent
from kiwi.reinforcement import KiwiLearning


def reward_function(state):
	move_size = 0.25
	if state[0] >= 1:
		return [1, 0, 0, 0]
	elif state[0] <= 0:
		return [0, 1, 0, 0]
	elif state[1] >= 1:
		return [0, 0, 1, 0]
	elif state[1] <= 0:
		return [0, 0, 0, 1]
	else:
		return [0.5, 0.5, 0.5, 0.5]


class Bee():
	def __init__(self):
		self.state = [0.5, 0.5]

	def moveLeft(self):
		self.state[0] -= 0.25
		return self.state

	def moveRight(self):
		self.state[0] += 0.25
		return self.state

	def moveUp(self):
		self.state[1] -= 0.25
		return self.state

	def moveDown(self):
		self.state[1] += 0.25
		return self.state


def main():
	bee = Bee()
	k = KiwiLearning(2,
					 4,
					 reward_function,
					 actions=[
					 bee.moveLeft,
					 bee.moveRight,
					 bee.moveUp,
					 bee.moveDown
					 ])

	for i in range(10):
		for j in range(0, 25):
			_, p, a = k.train(bee.state)
			print(k.data()[0])
			print(k.data()[1])

		bee.state = [0.5, 0.5]
		print(k)
		print("\n\n")
		time.sleep(5)

	save(k, "Learning")

if __name__ == "__main__":
	main()