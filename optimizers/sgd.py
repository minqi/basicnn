import numpy as np
from optimizer import Optimizer
from util.batch import create_batch_generator

class SGD(Optimizer):

	def __init__(self, model, cost, num_epochs = 10, batch_size = 2, lr = 0.15):
		Optimizer.__init__(self, model, cost, num_epochs)
		self.batch_size = batch_size
		self.lr = lr

	def update(self, parameter, gradient):
		return parameter - (self.lr/self.batch_size) * gradient

	def optimize(self, x, y):
		batch_generator = create_batch_generator(x, y, self.batch_size)

		output_shape = self.model.get_output_shape()
		losses = []
		n = len(x)
		for i in xrange(self.num_epochs):
			print 'Training epoch', i + 1
			batch_index = 0
			while (batch_index * self.batch_size < n):
				batch = batch_generator.next()
				gradients = np.zeros(output_shape)
				for bx, by in batch:
					activations = self.model.predict(bx)
					gradients += self.cost.derivative(by, activations)

				for layer in self.model.layers[::-1]:
					activations, gradients = layer.backward(gradients, self.update)

				batch_index += 1

			# compute loss
			y_hat = [self.model.predict(d) for d in x]
			loss = 1./len(y)*np.sum([self.cost.f(a, b) for a, b in zip(y_hat, y)])
			losses.append(loss)
			print 'training loss:', loss

		return losses