import numpy as np
from layers.layer import Layer
from util.activations import sigmoid, sigmoid_derivative

class Dense(Layer):

	def __init__(self, size):
		Layer.__init__(self, size)
		self.W = None
		self.b = np.random.rand(size, 1)
		self.z = None
		self.a = None
		self.x = None

	def forward(self, x):
		self.x = x
		self.z = self.W.dot(x) + self.b
		self.a = sigmoid(self.z)
		return self.a

	def backward(self, gradients, update):
		dz = gradients * sigmoid_derivative(self.z)
		dW = dz.dot(self.x.T)
		dx = self.W.T.dot(dz)
		db = dz
		
		self.W = update(self.W, dW)
		self.b = update(self.b, db)
	
		return (self.x, dx)

	def set_input(self, layer):
		input_size = np.product(layer.get_output_shape())
		self.W = np.random.rand(self.size, input_size)

	def get_output_shape(self):
		return (self.size, 1)
