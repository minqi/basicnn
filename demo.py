import numpy as np
import matplotlib.pyplot as plt
from models.mlp import MLP

mlp_args = {
	'lr': 0.2,
	'batchSize': 1,
	'numEpochs': 5000,
}

if __name__ == '__main__':
	# demo MLP
	data_x = np.array([1, 2, 3, 5])
	data_y = np.array([0.2, 0.4, 0.6, 1.0])

	train_x = np.reshape(data_x, (len(data_x), 1, 1))
	train_y = np.reshape(data_y, (len(data_y), 1, 1))
	layer_sizes = [1, 16, 1]
	mlp = MLP(layer_sizes, mlp_args)
	losses = mlp.train(train_x, train_y)
	y_hat = np.reshape([mlp.predict(x) for x in train_x], (len(train_x)))
	print y_hat
	x = np.reshape(train_x, (len(train_x)))
	y = np.reshape(train_y, (len(train_y)))
	# plt.plot(x, y, 'bo')
	# plt.plot(x, y_hat, 'ro')
	plt.plot(np.array(range(len(losses))) + 1, losses)
	plt.show()