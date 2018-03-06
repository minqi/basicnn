import numpy as np

from util.activations import sigmoid, sigmoid_derivative

class MLP():
	"""
	Basic multi-layer network with sigmoid activation and L2 loss.
	"""
	def __init__(self, layer_sizes, args):
		self.num_layers = len(layer_sizes)
		self.args = args
		weight_dims = zip(layer_sizes[1:], layer_sizes)
		self.W = np.array([np.random.rand(*dims) for dims in weight_dims])
		self.b = np.array([np.random.rand(k, 1) for k in layer_sizes[1:]])
		self.Z = np.array([np.zeros(i) for i in layer_sizes])
		self.A = np.array([np.zeros(i) for i in layer_sizes])

	def activation(self, z):
		return sigmoid(z)

	def grad_activation(self, z):
		return sigmoid_derivative(z)

	def loss(self, y, y_hat):
		return (1./2.0) * np.sum((y - y_hat) ** 2, 0)

	def da(self, y, y_hat):
		return np.sum((y_hat - y), 0)

	def predict(self, x):
		a = x
		self.Z[0] = a
		self.A[0] = a
		for i, w in enumerate(self.W):
			z = np.dot(w, a) + self.b[i]
			self.Z[i + 1] = z
			a = self.activation(z)
			self.A[i + 1] = a
		return a

	def backprop(self, x, y):
		y_hat = self.predict(x)
		dW = np.array([np.zeros(w.shape) for w in self.W])
		db = np.array([np.zeros(b.shape) for b in self.b])
		da = self.da(y, y_hat)
		dz = da * self.grad_activation(self.Z[-1])
		dW[-1] = np.dot(dz, self.A[-2].T)
		db[-1] = dz
		for i, w in enumerate(self.W[:0:-1]):
			l = self.num_layers - 1 - (i + 1)
			dz = np.dot(w.T, dz) * self.grad_activation(self.Z[l])
			dW[l - 1] = np.dot(dz, self.A[l - 1].T)
			db[l - 1] = dz
		return (dW, db)

	def get_next_mini_batch(self, x, y, batch_size):
		pairs = zip(x, y)
		assert(batch_size <= len(pairs))

		batch_start = 0
		while (True):
			next_start = batch_start + batch_size
			batch = pairs[batch_start:next_start]
			if next_start >= len(pairs):
				wrap = next_start - len(pairs)
				if wrap > 0:
					batch = np.concatenate((batch, batch[:wrap]), axis=0)
				next_start = wrap
			batch_start = next_start

			yield batch

	def update_mini_batch(self, batch):
		dW_batch = [np.zeros(w.shape) for w in self.W]
		db_batch = [np.zeros(b.shape) for b in self.b]
		for x, y in batch:
			dW, db = self.backprop(x, y)
			dW_batch = [cumsum + addend for cumsum, addend in zip(dW_batch, dW)]
			db_batch = [cumsum + addend for cumsum, addend in zip(db_batch, db)]
		return np.array(dW_batch), np.array(db_batch)

	def sgd(self, x, y, lr=.15, batch_size=4, epochs=300):
		lr = self.args.get('lr', lr)
		batch_size = self.args.get('batchSize', batch_size)
		epochs = self.args.get('numEpochs', epochs)

		losses = []
		mini_batch_generator = self.get_next_mini_batch(x, y, batch_size)
		n = len(x)
		for i in xrange(epochs):
			print 'Training epoch', i + 1
			batch_index = 0
			while (batch_index * batch_size < n):
				batch = mini_batch_generator.next()
				dW, db = self.update_mini_batch(batch)
				self.W = self.W - (lr/batch_size) * dW
				self.b = self.b - (lr/batch_size) * db
				batch_index += 1

			# compute loss
			y_hat = [self.predict(d) for d in x]
			loss = 1./len(y)*np.sum([self.loss(a, b) for a, b in zip(y_hat, y)])
			losses.append(loss)
			print 'training loss:', loss

		return losses
				
	def train(self, train_x, train_y):
		return self.sgd(train_x, train_y)
