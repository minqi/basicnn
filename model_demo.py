import numpy as np
from core.model import Model
from layers.input import Input
from layers.dense import Dense
from util.cost_functions import L2

if __name__ == '__main__':
	# demo MLP
	data_x = np.array([1, 2])
	data_y = np.array([0.2, 0.4])

	train_x = np.reshape(data_x, (len(data_x), 1, 1))
	train_y = np.reshape(data_y, (len(data_y), 1, 1))

	model = Model()
	model.add(Input(1))
	model.add(Dense(3))
	model.add(Dense(1))
	model.compile(cost=L2(), optimizer='sgd', num_epochs=30000, batch_size=1, lr=0.1)
	model.train(train_x, train_y)

	test_data_x = np.array([1])
	test_x = np.reshape(test_data_x, (len(test_data_x), 1))
	print model.predict(test_x)