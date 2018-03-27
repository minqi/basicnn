import numpy as np
from function import function

def L2(function):

	def f(a, b):
		return (a - b) ** 2

	def derivative(x):
		return 2 * x