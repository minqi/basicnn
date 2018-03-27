import numpy as np
from optimizers import SGD

class Model():
	def __init__(self):
		self.layers = []
		self.optimizer = None

	def add(self, layer):
		self.layers.append(layer)

		num_layers = len(self.layers)
		if num_layers > 1:
			prev_layer = self.layers[num_layers - 2]
			layer.set_input(prev_layer)

	def compile(self, cost, optimizer = 'sgd'):
		if optimizer == 'sgd':
			self.optimizer = SGD(self, cost)

	def predict(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def train(data):
		self.optimizer.train(data)