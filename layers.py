import numpy as np

class Layer:
	"""
	Base class for defining all layers.
	------------------------------------

	Each layer is capable of atleast two things:

	- Process input to get output.

	- Propagate gradients through itself.

	Some layers will have learnable parameters which will 
	update during backward() call.
	"""

	def __init__(self, input_data):
		"""
		Initialize layer parameters, if any.
		"""
		pass

	def forward(self, input_data):
		"""
		Takes input data of shape [batch, input_units],
		return output data [batch, output_units]
		"""
		return input_data

	def backward(self, input_data, grad_output):
		"""
		Perform backpropagation step through the layer,
		with respect to given input.
		"""
		num_units = input_data.shape[1]
		d_layer_d_input = np.eye(num_units)

		return np.dot(grad_output, d_layer_d_input)	# chain rule


class Dense(Layer):
	"""
	Dense layer performs a learned affine transformation:
	f(x) = <W*x> + b
	-----------------------------------------------------

	A dense layer is a deep fully-connected layer made of defined input
	and output size.
	"""
	def __init__(self, input_units, output_units, 
		learning_rate=0.1, momentum=0.9, optimizer='sgd',
		ahead_param=None, init=None):
		"""
		Initialize the parameters and weights for Dense Layer.
		"""

		self.learning_rate = learning_rate
		self.momentum = momentum
		self.optimizer = optimizer

		if init == 'xavier':
			self.weights = np.random.randn(input_units, 
				output_units)*np.sqrt(2/(input_units+output_units))

		elif init == None:
			self.weights = np.random.randn(input_units, 
				output_units)*0.01

		else:
			raise "No such weight initialization described."

		self.biases = np.zeros(output_units)

	def forward(self, input_data):
		"""
		Perform the affine transformation:
		f(x) = <W*x> + b

		input_shape = [batch, input_units]
		output_units = [batch, output_units]
		"""
		return np.dot(input_data, self.weights) + self.biases

	def backward(self, input_data, grad_output):
		"""
		Compute the gradient of output against the weights.
		"""
		grad_input = np.dot(grad_output, self.weights.T)

		# compute gradient w.r.t weights and biases
		grad_weights = np.dot(input_data.T, grad_output)
		grad_biases = grad_output.mean(axis=0)

		# test for correct shape
		assert grad_weights.shape == self.weights.shape\
		 and grad_biases.shape == self.biases.shape

		# peform optimization step
		self._optimization(grad_weights, grad_biases)

		return grad_input

	def _optimization(self, grad_weights, grad_biases):
		"""
		Optimization step.(Updates weights and biases)
		------------------
		- 'sgd': Stochiastic gradient descent step
		- 'nesterov': Nesterov Momentum Method
		- 'rmsprop': RMSprop
		- 'adam' - Adam optimization
		"""
		if self.optimization == 'sgd':
			self.weights = self.weights - self.learning_rate * grad_weights
			self.biases = self.biases - self.learning_rate * grad_biases

		# elif self.optimization == 'nesterov':

