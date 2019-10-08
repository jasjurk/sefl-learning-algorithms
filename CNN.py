import random
from math import exp, floor
from PIL import Image

def full_convolution(input, filter): #full convolution function, used in backpropagation of convolutional layers
	n = len(input)
	m = len(input[0])
	filterx = len(filter)
	filtery = len(filter[0])
	result = [[0 for x in range(m + filtery)] for y in range(n + filterx)]
	for i in range(n + filterx):
		for i2 in range(m + filtery):
			for x in range(i, i + filterx):
				for y in range(i2, i2 + filtery):
					if(x - filterx + 1 < n and y - filtery + 1 < m and x + 1 >= filterx and y + 1 >= filtery):
						result[i][i2] += input[x - filterx + 1][y - filtery + 1] * filter[x - i][y - i2]
	return result

def convolution(input, filter): #convolution, basic operation in convolutional layer
	n = len(input)
	m = len(input[0])
	filterx = len(filter)
	filtery = len(filter[0])
	result = [[0 for x in range(m - filtery + 1)] for y in range(n - filterx + 1)]
	for i in range(n - filterx + 1):
		for i2 in range(m - filtery + 1):
			for x in range(i, i + filterx):
				for y in range(i2, i2 + filtery):
					result[i][i2] += input[x][y] * filter[x - i][y - i2]
	return result
	
def add(mat1, mat2): #adding two matricies
	return [[mat1[x][y] + mat2[x][y] for y in range(len(mat1[0]))] for x in range(len(mat1))]
		
def flip(mat): #flipping matrix by 180 degrees
	n = len(mat)
	m = len(mat[0])
	return [[mat[n - x - 1][m - y - 1] for y in range(m)] for x in range(n)]

class Layer: #convolutional layer definition
	def __init__(self, n, m, layers, filterx, filtery, filtercount):
		self.n = n
		self.m = m
		self.layers = layers
		self.filterx = filterx
		self.filtery = filtery
		self.filtercount = filtercount
		self.filters = [[[[random.uniform(-1, 1) for q in range(filtery)] for k in range(filterx)] for j in range(layers)] for i in range(filtercount)]
		
	def compute(self, inputs):
		self.inputs = inputs
		self.filtererror = []
		self.result = [[[0 for k in range(self.m - self.filtery + 1)]for j in range(self.n - self.filterx + 1)] for i in range(self.filtercount)]
		for fil in range(self.filtercount):
			for lay in range(self.layers):
				self.result[fil] = add(self.result[fil], convolution(inputs[lay], self.filters[fil][lay]))
		return self.result
		
	def backpropagate(self, errors):
		self.filtererror = [[[[0 for q in range(self.filtery)] for k in range(self.filterx)] for j in range(self.layers)] for i in range(self.filtercount)]
		inputerror = [[[0 for k in range(self.m)] for j in range(self.n)] for i in range(self.layers)]
		for fil in range(self.filtercount):
			for lay in range(self.layers):
				inputerror[lay] = add(inputerror[lay], full_convolution(errors[fil], flip(self.filters[fil][lay])))
				self.filtererror[fil][lay] = convolution(self.inputs[lay], errors[fil])
		return inputerror
	
	def cleanup(self):
		del self.filtererror
		del self.inputs
		del self.result
	
	def learning(self, d):
		for fil in range(self.filtercount):
			for lay in range(self.layers):
				for i in range(self.filterx):
					for i2 in range(self.filtery):
						self.filters[fil][lay][i][i2] += d * self.filtererror[fil][lay][i][i2]
		self.cleanup()
	
	def get_output_dims(self):
		return (self.n - self.filterx + 1, self.m - self.filtery + 1, self.layers)
		
class Maxpool: #pooling layer definition
	def __init__(self, n, m, layers, poolingx, poolingy):
		self.n = n
		self.m = m
		self.layers = layers
		self.poolingx = poolingx
		self.poolingy = poolingy

	def compute(self, inputs):
		n = floor(self.n / self.poolingx)
		m = floor(self.m / self.poolingy)
		self.result = [[[(float("-inf"), 0, 0) for k in range(m)]for j in range(n)] for i in range(self.layers)]
		result = [[[0 for k in range(m)]for j in range(n)] for i in range(self.layers)]
		for lay in range(self.layers):
			for x in range(n):
				for y in range(m):
					for i in range(x * self.poolingx, x * self.poolingx + self.poolingx):
						for i2 in range(y * self.poolingy, y * self.poolingy + self.poolingy):
							self.result[lay][x][y] = max(self.result[lay][x][y], (inputs[lay][i][i2], i, i2))
					result[lay][x][y] = self.result[lay][x][y][0]
		return result
		
	def backpropagate(self, errors):
		result = [[[0 for k in range(self.m)]for j in range(self.n)] for i in range(self.layers)]
		for lay in range(self.layers):
			for x in range(floor(self.n / self.poolingx)):
				for y in range(floor(self.m / self.poolingy)):
					result[lay][self.result[lay][x][y][1]][self.result[lay][x][y][2]] = errors[lay][x][y]
		return result
					
	def cleanup(self):
		del self.result
	
	def learning(self, d):
		self.cleanup()
		
	def get_output_dims(self):
		return (floor(self.n / self.poolingx), floor(self.m / self.poolingy), self.layers)
	
class ReLu: #rectified linear unit(ReLu) layer for breaking linearity of the network
	def __init__(self, n, m, layers, f, fp):
		self.n = n
		self.m = m
		self.layers = layers
		self.f = f
		self.fp = fp

	def compute(self, inputs):
		self.inputs = inputs
		result = [[[0 for k in range(self.m)]for j in range(self.n)] for i in range(self.layers)]
		for lay in range(self.layers):
			for x in range(self.n):
				for y in range(self.m):
					result[lay][x][y] = self.f(inputs[lay][x][y])
		return result
		
	def backpropagate(self, errors):
		result = [[[0 for k in range(self.m)]for j in range(self.n)] for i in range(self.layers)]
		for lay in range(self.layers):
			for x in range(self.n):
				for y in range(self.m):
					result[lay][x][y] = errors[lay][x][y] * self.fp(self.inputs[lay][x][y])
		return result
	
	def cleanup(self):
		del self.inputs
	
	def learning(self, d):
		self.cleanup()
		
	def get_output_dims(self):
		return (self.n, self.m, self.layers)

class FullyConnected: #fully connected perceptron layer definition
	def __init__(self, n, m, layers, hidden, f, fp):
		self.n = n
		self.m = m
		self.layers = layers
		self.hidden = hidden
		self.weights =  [[[[random.uniform(-1, 1) for q in range(self.m)] for k in range(self.n)] for j in range(self.layers)]for i in range(self.hidden)]
		self.bias = [random.uniform(-1, 1) for k in range(self.hidden)]
		self.f = f
		self.fp = fp

	def compute(self, inputs):
		self.inputs = inputs
		self.errors = []
		self.result = [[[self.bias[k] for k in range(self.hidden)]for j in range(1)]for i in range(1)]
		result = [[[0 for k in range(self.hidden)]for j in range(1)]for i in range(1)]
		for i in range(self.hidden):
			for lay in range(self.layers):
				for x in range(self.n):
					for y in range(self.m):
						self.result[0][0][i] += inputs[lay][x][y] * self.weights[i][lay][x][y]
			result[0][0][i] = self.f(self.result[0][0][i])	
		return result
		
	def backpropagate(self, errors):
		self.errors = errors
		result = [[[0 for k in range(self.m)]for j in range(self.n)] for i in range(self.layers)]
		for i in range(self.hidden):
			for lay in range(self.layers):
				for x in range(self.n):
					for y in range(self.m):
						result[lay][x][y] += self.fp(self.result[0][0][i]) * errors[0][0][i] * self.weights[i][lay][x][y]
		return result
	
							
	def cleanup(self):
		del self.inputs
		del self.errors
		del self.result
	
	def learning(self, d):
		for i in range(self.hidden):
			for lay in range(self.layers):
				for x in range(self.n):
					for y in range(self.m):
						self.weights[i][lay][x][y] += d * self.inputs[lay][x][y] * self.errors[0][0][i] * self.fp(self.result[0][0][i])
			self.bias[i] += d * self.fp(self.result[0][0][i]) * self.errors[0][0][i]
		self.cleanup()
		
	def get_output_dims(self):
		return (1, self.hidden, 1)
		
class Network: #definition of neural network class, should be imported for usage
	def __init__(self, n, m): #constructor, you should pass dimensions of input image to it
		self.n = n
		self.m = m
		self.list = []
		
	def get_prev_dims(self): #internal function
		if(len(self.list) == 0):
			return (self.n, self.m, 3)
		else:
			return self.list[-1].get_output_dims()
	
	def add_convolutional_layer(self, filterx, filtery, filtercount): #adds a convolutional layer at the end of network, with stride=1 and filters with given dimensions
		(n, m, layers) = self.get_prev_dims()
		self.list.append(Layer(n, m, layers, filterx, filtery, filtercount))
	
	def add_maxpooling_layer(self, poolingx, poolingy): #adds pooling layer for size and noise reduction
		(n, m, layers) = self.get_prev_dims()
		self.list.append(Maxpool(n, m, layers, poolingx, poolingy))
		
	def add_relu_layer(self, f, fp): #adds ReLu layer, using given function f and its derivative fp, used to model non-linear problems
		(n, m, layers) = self.get_prev_dims()
		self.list.append(ReLu(n, m, layers, f, fp))
		
	def add_fully_connected_layer(self, hidden, f, fp): #adds fully connected perceptron layer, used as last of layers of network, last layer should have as many hidden neurons as you want output
		(n, m, layers) = self.get_prev_dims()
		self.list.append(FullyConnected(n, m, layers, hidden, f, fp))
		
	def compute(self, image): #returns result of network (for a given image, which needs to be image from PIL), as a list of floats
		inputs = [[[image.getpixel((j, k))[i] for k in range(self.m)]for j in range(self.n)]for i in range(3)]
		for i in range(len(self.list)):
			inputs = self.list[i].compute(inputs)
			self.list[i].cleanup()
		return inputs[0][0]
	
	def learn(self, image, output, d): #uses backpropagation to learn with supervision: image is the networks input(a PIL image), output is list of floats (desired outputs) and learing coefficient. Returns networks error
		inputs = [[[image.getpixel((j, k))[i] for k in range(self.m)]for j in range(self.n)]for i in range(3)]
		errors = [[[0 for i in range(self.get_prev_dims()[1])]]]
		for i in range(len(self.list)):
			inputs = self.list[i].compute(inputs)
		for i in range(self.get_prev_dims()[1]):
			errors[0][0][i] = output[i] - inputs[0][0][i] #derivative of error function, difference between networks answer and desired answer
		for i in range(len(self.list) - 1, -1, -1):
			#print(self.list[i].get_output_dims())
			errors = self.list[i].backpropagate(errors)
		for i in range(len(self.list)):
			self.list[i].learning(d)
		error = 0
		for i in range(self.get_prev_dims()[1]):
			error += (output[i] - inputs[0][0][i])**2 / 2 #sum of differences squared(divided by 2 for easier integration)
		return error
