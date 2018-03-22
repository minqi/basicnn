import numpy as np
from layers import Layer
from util.activations import sigmoid, sigmoid_derivative

class Dense(Layer):

	def __init__(self, size):
		self.W = None
		self.b = np.zeros(size)
		self.a = None
		self.x = None

	def forward(self, x):
		self.x = x
		self.a = sigmoid(W.dot(x) + b)
		return self.a

	def backward(self, gradients):
		dz = gradients * sigmoid_derivative(a)
		dW = dz.dot(self.x.T)
		db = dx 
		dx = self.W.T.dot(dz)

	def set_input(self, layer):
		input_size = np.product(layer.get_output_shape())
		self.W = np.zeros((size, input_size))

	def get_output_shape(self):
		return (size,)
