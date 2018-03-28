import numpy as np
from layers.layer import Layer

class Input(Layer):

	def __init__(self, shape):
		self.shape = shape

	def forward(self, x):
		return x

	def backward(self, gradients):
		return gradients

	def get_output_shape(self):
		return self.shape