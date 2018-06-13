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

	def _softmax_crossentropy_with_logits(self, logits, reference_ans):
		"""
		Compute crossentropy from logits[batch,n_classes] and ids of correct answers
		----------------------------------------------------------------------------
		loss = - a(correct) + log(summation(e^ai))

		"""
		logits_for_ans = logits[np.arange(len(logits)),reference_answers]
		xentropy = - logits_for_ans + np.log(np.sum(np.exp(logits), axis=-1))
		return xentropy

	def _grad_softmax_crossentropy_with_logits(self, logits, reference_answers):
		"""
		Compute crossentropy gradients from logits and ids of correct answers.
		"""
		ones_for_answers = np.zeros_like(logits)
		ones_for_answers[np.arange(len(logits)),reference_answers] = 1

		softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

		return (-ones_for_answers + softmax) / logits.shape[0]


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
			# NOTE: currently only softmax cross entropy is available
			
			loss = self._softmax_crossentropy_with_logits(logits, y)
			loss_grad = self._grad_softmax_crossentropy_with_logits(logits, y)

			grad = loss_grad
			for idx, layer in enumerate(self.network[::-1]):
				grad = layer.backward(layer_inputs[::-1][idx+1], grad)

		return np.mean(loss)

