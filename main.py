from kiwi import *

batches = 40

in_ = model.Dataset(8)
out_ = model.Dataset(8)

toBinary = lambda i: list(map(int, list('{0:08b}'.format(int(i)))))

for i in range(batches):
    in_.add(toBinary(i))
    out_.add(toBinary(i + 1))

nn = regression.GradientDescent(
    in_(),
    out_()
)

tools.save(nn, "BinaryAddOne")

print("Neural Network to add one to a binary number")
for i in range(40, 50):
    print("nn={}, real={}".format(tools.round_array(nn(toBinary(i))), toBinary(i)))
