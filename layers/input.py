import numpy as np
from layers.layer import Layer

class Input(Layer):

	def __init__(self, shape):
		self.shape = shape
		self.x = None

	def forward(self, x):
		self.x = x
		return x

	def backward(self, gradients, update):
		return self.x, np.zeros(self.x.shape)

	def get_output_shape(self):
		return self.shape