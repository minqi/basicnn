import numpy as np
from optimizers.sgd import SGD

class Model():
	def __init__(self):
		self.layers = []
		self.optimizer = None

	def get_layers(self):
		return self.layers

	def get_output_shape(self):
		return self.layers[-1].get_output_shape()

	def add(self, layer):
		self.layers.append(layer)

		num_layers = len(self.layers)
		if num_layers > 1:
			prev_layer = self.layers[num_layers - 2]
			layer.set_input(prev_layer)

	def compile(self, cost, optimizer = 'sgd', num_epochs=10, batch_size=4, lr=0.15):
		if optimizer == 'sgd':
			self.optimizer = SGD(self, cost=cost, num_epochs=num_epochs, batch_size=batch_size, lr=lr)

	def predict(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def train(self, train_x, train_y):
		self.optimizer.optimize(train_x, train_y)