import numpy as np
from util.batch import create_batch_generator

class SGD(Optimizer):

	def __init__(self, model, cost, num_epochs, batch_size, lr):
		Optimizer.__init__(self, model, cost, num_epochs)
		self.batch_size = batch_size
		self.lr = lr

	def update(parameter, gradient):
		return parameter + (self.lr/batch_size) * gradient

	def optimize(self, x, y):
		batch_generator = create_batch_generator(x, y, self.batch_size)

		losses = []
		n = len(x)
		for i in xrange(self.num_epochs):
			print 'Training epoch', i + 1
			batch_index = 0
			while (batch_index * self.batch_size < n):
				batch = batch_generator.next()
				for x, y in batch:
					activations = self.model.predict(x)
					gradients = np.ones(self.model.layers[-1].get_size())
					for layer in self.model.layers[:0:-1]:
						activations, gradients = layer.backward(activations, gradients, self.update)

				batch_index += 1

			# compute loss
			y_hat = [self.model.predict(d) for d in x]
			loss = 1./len(y)*np.sum([self.cost.f(a, b) for a, b in zip(y_hat, y)])
			losses.append(loss)
			print 'training loss:', loss

		return losses