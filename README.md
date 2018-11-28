# ml
One of my old attempts at writing a machine learning library.

## How it works

I'll use one of the examples to demonstrate how it works.
```python
# imports all the libraries in the kiwi package
from kiwi import *


# number of iterations of training
batches = 40


# create training input and output datasets each composed of arrays with 8 pieces of data each
in_ = model.Dataset(8)
out = model.Dataset(8)


# a function to convert an int into a 8 digit binary int
toBinary = lambda i: list(map(int, list('{0:08b}'.format(int(i)))))

for i in range(batches):
	# add an example input to the input training dataset
	in_.add(toBinary(i))

	# add the expected output to the training output dataset
	out_.add(toBinary(i+1))

# lambda function of the neural net
nn = regression.GradientDescent(
	in(),
	out_()
)

# pickle the neural net
tools.save(nn, "BinaryAddOne")


# compare the neural net output to the input
print("Neural Network to add one to a binary number")
for i in range(40, 50):
	print("nn={}, real={}".format(tools.round_array(nn(toBinary(i))), toBinary(i)))
```

Although it seems really nice, it's only linear, so there is no deep learning :(
Also, the KLearning example is experimental, I'm not sure if it worked or not; I worked on this months and months ago.
