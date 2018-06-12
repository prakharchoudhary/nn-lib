# Add all the activation functions: sigmoid, softmax, reLu etc.

from .layers import Layer
import numpy as np

class Sigmoid(Layer):
	"""
	Apply sigmoid activation:

	f(x) = 1 / (1 + e^-x)
	"""
	def __init__(self):
		"""
		Nothing to initialize
		"""
		pass

	def forward(self, input_data):
		"""
		Apply the activation function:
		f(x) = 1 / (1 + e^-x)
		"""
		return 1/(1+np.exp(-input_data))

	def backward(self, input_data, grad_output):
		"""
		Compute the gradient loss w.r.t Sigmoid input
		"""
		f = 1/(1+np.exp(-input_data))
		sigmoid_grad = f*(1-f)
		return sigmoid_grad*grad_output

class ReLu(Layer):
	"""
	Apply Rectifier Linear Unit(ReLu) activation:

	f(x) = max(0, x)
	"""
	def __init__(self):
		"""
		Nothing to initialize
		"""
		pass

	def forward(self, input_data):
		"""
		Apply element-wise ReLU to input matrix
		"""
		return np.maximum(0, input_data)

	def backward(self, input_data, grad_output):
		"""
		Compute the gradient loss w.r.t ReLu input
		"""
		relu_grad = input_data > 0
		return grad_output * relu_grad