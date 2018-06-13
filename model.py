# Carries the network class for sequentially stacking layers.

import numpy as np

class Network:
	"""
	Model for sequentially stacking deep learning layers.
	"""
	def __init__(self, epochs):
		self.network = []
		self.epochs = epochs

	def add(self, layer):
		self.network.append(layer)
		return True

	def forward(self, X):
		"""
		Return a list of activations for each layer. 
		"""
		activations = []

		for idx, layer in enumerate(self.network):
			if idx==0:
				activations.append(layer.forward(X))
			else:
				activations.append(layer.forward(activations[idx-1]))

		assert len(activations) == len(self.network)
		return activations

	def predict(self, X):
		"""
		Compute network predictions.
		"""
		probabs = self.forward(self.network, X)
		return probabs.argmax(axis=-1)

	def train(self, X, y):
		"""
		Train on a given batch of X and y.
		"""
		for epoch in self.epochs:
			layer_activations = self.forward(X)
			layer_inputs = [X]+layer_activations
			logits = layer_activations[-1]

			# Compute the loss and initialize gradient
			# TODO: take the type of loss calculation when initializing
			# the model.
			
			# loss = 
			# loss_grad = 

			grad = loss_grad
			for idx, layer in enumerate(self.network[::-1]):
				grad = layer.backward(layer_inputs[::-1][idx+1], grad)

		return np.mean(loss)

