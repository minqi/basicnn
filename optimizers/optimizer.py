class Optimizer():

	def __init__(self, model, cost, num_epochs):
		self.model = model
		self.cost = cost
		self.num_epochs = num_epochs

	def optimize(self, train_x, train_y):
		pass