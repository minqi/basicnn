import numpy as np
from layers.layer import Layer
from util.activations import sigmoid, sigmoid_derivative

class Dense(Layer):

	def __init__(self, size):
		Layer.__init__(self, size)
		self.W = None
		self.b = np.zeros(size)
		self.a = None
		self.x = None

	def forward(self, x):
		self.x = x
		self.a = sigmoid(self.W.dot(x) + self.b)
		return self.a

	def backward(self, activations, gradients, update):
		dz = gradients * sigmoid_derivative(activations)
		dW = dz.dot(self.x.T)
		dx = self.W.T.dot(dz)
		db = dx 
		
		update(self.W, dW)
		update(self.b, db)
	
		return (self.x, dx)

	def set_input(self, layer):
		input_size = np.product(layer.get_output_shape())
		self.W = np.zeros((self.size, input_size))

	def get_output_shape(self):
		return (self.size,)
