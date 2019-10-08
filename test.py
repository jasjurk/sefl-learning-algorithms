# Jan Jurkowski
# example of usage of my package for deep learing convolutional neural networks
# here only fully connected layers are used, because it's task is computing xor

import random
from math import exp
from PIL import Image	
from CNN import Network #import network class

def f(x): #logistic activation function
	return 1 / (1 + exp(-x))
	
def fp(x):
	return f(x) * (1 - f(x))
	
random.seed()
n = Network(1, 2) #make network which takes 1 * 2 image as input
n.add_fully_connected_layer(2, f, fp) #add hidden layer with 10 hidden neurons
n.add_fully_connected_layer(1, f, fp) #add hidden layer with 1 neuron, output
im = [Image.new('RGB', (1, 2), color = (0, 0, 0)) for i in range(4)] #preparing learing set
im[0].putpixel((0, 0), (0, 0, 0))
im[0].putpixel((0, 1), (0, 0, 0))
im[1].putpixel((0, 0), (0, 1, 0))
im[1].putpixel((0, 1), (0, 0, 0))
im[2].putpixel((0, 0), (0, 0, 0))
im[2].putpixel((0, 1), (0, 1, 0))
im[3].putpixel((0, 0), (0, 1, 0))
im[3].putpixel((0, 1), (0, 1, 0))
res = [[0], [1], [1], [0]]
for i in range(1000000): #10000 cycles of learing
	x = random.randint(0, 3)
	error = n.learn(im[x], res[x], 0.7)
	if(i % 10000 == 0): #show apparent error every 1000 cycles
		print(error)
for i in range(4):
print(n.compute(im[i])[0]) #check to see how much it learned
