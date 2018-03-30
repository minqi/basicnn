import numpy as np
from function import function

class L2(function):

	def f(self, a, b):
		return (b - a) ** 2

	def derivative(self, a, b):
		return 2 * (b - a)