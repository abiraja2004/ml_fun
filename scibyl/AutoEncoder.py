# things to worry about
# linearity of features
# works best w/ non-linear coupling among features
# compound boolean queries
# simplfied fuzzy arch map

# managing uncertainties w/ ML techniques
# tackling confidence intervals w/ input parameter uncertainties and MCMC

import time
import numpy as np
import theano as th
import pickle
import gzip
from theano import tensor as T
from matplotlib import pyplot
from theano.tensor.shared_randomstreams import RandomStreams

class AutoEncoder(object):
	def __init__(self, x, hidden_size, activation_function, output_function, 
		rng=None, theano_rng=None, W=None, bvis=None, bhid=None):
		self.x = x
		self.x = th.shared(name='x', value=np.asarray(self.x, dtype=th.config.floatX), borrow=True)

		self.hidden_size = hidden_size
		self.n = x.shape[1]
		self.m = x.shape[0]

		self.activation_function = activation_function
		self.output_function = output_function

		if not rng:
			rng = np.random.RandomState(123)
		if not theano_rng:
			theano_rng = RandomStreams(rng.randint(2**30))
		self.theano_rng = theano_rng

		if not W:
			initial_W = np.asarray(rng.uniform(
				low=-4*np.sqrt(6/(self.hidden_size+self.n)),
				high=4*np.sqrt(6/(self.hidden_size+self.n)),
				size=(self.n, self.hidden_size)), dtype=th.config.floatX)
			W = th.shared(value=initial_W, name='W', borrow=True)

		if not bvis:
			bvis = th.shared(value=np.zeros(self.n, dtype=th.config.floatX), name='bvis')

		if not bhid:
			bhid = th.shared(value=np.zeros(self.hidden_size, dtype=th.config.floatX), name='bhid')

		self.W = W
		self.b = bhid
		self.bp = bvis
		self.Wp = self.W.T

		self.params = [self.W, self.b, self.bp]

	def get_hidden_values(self, input):
		return self.activation_function(T.dot(input, self.W)+self.b)

	def get_reconstructed_values(self, input):
		return self.output_function(T.dot(input, self.Wp)+self.bp)

	def get_cost_updates(self, input, learning_rate):
		y = self.get_hidden_values(input)
		z = self.get_reconstructed_values(y)
		
		L = -T.sum(input*T.log(z)+(1-input)*T.log(1-z), axis=1)
		cost = T.mean(L)

		gparams = T.grad(cost, self.params)
		updates = [(param, param-learning_rate*gparam) for param, gparam in zip(self.params, gparams)]

		return cost, updates

	def get_weights(self):
		return [self.W.get_value(), self.b.get_value(), self.bp.get_value()]

	def train(self, n_epochs=100, minibatch_size=1, learning_rate=0.1):
		index = T.lscalar()
		x = T.matrix('x')

		cost, updates = self.get_cost_updates(x, learning_rate)
		trainer = th.function(inputs=[index], outputs=[cost], updates=updates,
			givens={x: self.x[index:index+minibatch_size, :]})

		start_time = time.clock()

		for epoch in range(n_epochs):
			print('Epoch: ', epoch)
			loss = []
			for row in range(0, self.m, minibatch_size):
				loss.append(trainer(row))
			print('Loss: ', np.mean(loss))

		end_time = time.clock()
		print('\nAverage time per epoch: ', (end_time-start_time)/n_epochs)
		print('Average loss: ', np.mean(loss))


def plot_features(weights, k, image_name='features'):
	weight_size = weights.shape[0]
	k = min(weight_size, k)
	j = int(round(k/10.))

	fig, ax = pyplot.subplots(j,10)

	for i in range(k):
		w = weights[i, :]
		w = w.reshape(28,28)
		ax[i/10, i%10].imshow(w, cmap=pyplot.cm.gist_yarg, interpolation='nearest', aspect='equal')
		ax[i/10, i%10].axis('off')

	pyplot.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
	pyplot.tick_params(axis='y', which='both', bottom='off', top='off', labelbottom='off')

	pyplot.savefig(image_name)


def load_data(filename):
	with gzip.open(filename, 'rb') as f:
		 train_set, valid_set, test_set = pickle.load(f, encoding='latin-1')
	return train_set, valid_set, test_set


def get_features(data, epochs=20, batch_size=20, hidden_size=100, learning_rate=0.1):
	x = data
	activation_function = T.nnet.sigmoid
	output_function = activation_function
	encoder = AutoEncoder(x, hidden_size, activation_function, output_function)
	encoder.train(n_epochs=epochs, minibatch_size=batch_size, learning_rate=learning_rate)
	weights = np.transpose(encoder.get_weights()[0])

	return weights


def demo():
	dataset = load_data('mnist.pkl.gz')
	print('Hidden Layer = 100')
	weights = get_features(dataset[0][0], epochs=5, hidden_size=100)
	plot_features(weights, 100, image_name='hidden_100')

	print('\n\nHidden Layer = 500')
	weights = get_features(dataset[0][0], epochs=5, hidden_size=500)
	plot_features(weights, 100, image_name='hidden_500')
