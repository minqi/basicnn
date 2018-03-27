import numpy as np
from function import function

def sigmoid(z):
	return 1./(1. + np.exp(-z))

def sigmoid_derivative(z):
	a = sigmoid(z)
	return (1 - a) * a

def Sigmoid(function):
	def f(x):
		return 1./(1. + np.exp(-x))

	def derivative(x):
		a = sigmoid(x)
		return (1 - a) * a