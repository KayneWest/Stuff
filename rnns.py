import cPickle
import os
import sys
import time
import socket
import numpy
from collections import OrderedDict
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared



BATCH_SIZE = 100
SAG = False
 
def relu_f(vec):
	""" Wrapper to quickly change the rectified linear unit function """
	return (vec + abs(vec)) / 2.
 
 
def softplus_f(v):
	return T.log(1 + T.exp(v))
 
 
def dropout(rng, x, p=0.5):
	""" Zero-out random values in x with probability p using rng """
	if p > 0. and p < 1.:
		seed = rng.randint(2 ** 30)
		srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
		mask = srng.binomial(n=1, p=1.-p, size=x.shape,
			dtype=theano.config.floatX)
		return x * mask
	return x
 
 
def fast_dropout(rng, x):
	""" Multiply activations by N(1,1) """
	seed = rng.randint(2 ** 30)
	srng = RandomStreams(seed)
	mask = srng.normal(size=x.shape, avg=1., dtype=theano.config.floatX)
	return x * mask


def build_shared_zeros(shape, name):
	""" Builds a theano shared variable filled with a zeros numpy array """
	return shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
	        name=name, borrow=True)


class Linear(object):
	""" Basic linear transformation layer (W.X + b) """
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, fdrop=False):
		if W is None:
			W_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			W_values *= 4  # This works for sigmoid activated networks!
			W = theano.shared(value=W_values, name='W', borrow=True)
		if b is None:
			b = build_shared_zeros((n_out,), 'b')
		self.input = input
		self.W = W
		self.b = b
		self.params = [self.W, self.b]
		self.output = T.dot(self.input, self.W) + self.b
		if fdrop:
			self.output = fast_dropout(rng, self.output)

	def __repr__(self):
		return "Linear"


class SigmoidLayer(Linear):
	""" Sigmoid activation layer (sigmoid(W.X + b)) """
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, fdrop=False):
		super(SigmoidLayer, self).__init__(rng, input, n_in, n_out, W, b)
		self.pre_activation = self.output
		if fdrop:
			self.pre_activation = fast_dropout(rng, self.pre_activation)
		self.output = T.nnet.sigmoid(self.pre_activation)
 
 
class ReLU(Linear):
	""" Rectified Linear Unit activation layer (max(0, W.X + b)) """
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, fdrop=False):
		if b is None:
			b = build_shared_zeros((n_out,), 'b')
		super(ReLU, self).__init__(rng, input, n_in, n_out, W, b)
		self.pre_activation = self.output
		if fdrop:
			self.pre_activation = fast_dropout(rng, self.pre_activation)
		self.output = relu_f(self.pre_activation)
 
 
class SoftPlus(Linear):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, fdrop=0.):
		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)
		super(SoftPlus, self).__init__(rng, input, n_in, n_out, W, b)
		self.pre_activation = self.output
		if fdrop:
			self.pre_activation = fast_dropout(rng, self.pre_activation, fdrop)
		self.output = softplus_f(self.pre_activation)


class RecurrentReLU(object):
	def __init__(self, rng, input, in_stack, n_in, n_in_stack, n_out,
			W=None, Ws=None, b=None,fdrop=False):  
		if W is None:
			W_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			W_values *= 4  
			W = theano.shared(value=W_values, name='W', borrow=True)
		if b is None:
			b = build_shared_zeros((n_out,), 'b')
		if Ws is None:
			Ws_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in_stack + n_out)),
				high=numpy.sqrt(6. / (n_in_stack + n_out)),
				size=(n_in_stack, n_out)), dtype=theano.config.floatX)
			Ws_values *= 4  # TODO check
			Ws = shared(value=Ws_values, name='Ws', borrow=True)
		self.input_stack = in_stack
		self.Ws = Ws  # weights of the reccurrent connection
		self.input = input
		self.W = W
		self.b = b
		self.params = [self.W, self.b, self.Ws] 
		self.output = (T.dot(self.input, self.W) 
				+ T.dot(self.input_stack, self.Ws) + self.b)
		self.pre_activation = self.output
		if fdrop:
			self.pre_activation = fast_dropout(rng, self.pre_activation)
		self.output = relu_f(self.pre_activation)


class LogisticRegression:
	"""Multi-class Logistic Regression 		#TODO CROSS ENTROPY
	"""
	def __init__(self, rng, input, n_in, n_out, W=None, b=None):
		if W != None:
			self.W = W
		else:
			self.W = build_shared_zeros((n_in, n_out), 'W')
		if b != None:
			self.b = b
		else:
			self.b = build_shared_zeros((n_out,), 'b')

		# P(Y|X) = softmax(W.X + b)
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.output = self.y_pred
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def negative_log_likelihood_sum(self, y):
		return -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def training_cost(self, y):
		""" Wrapper for standard name """
		return self.negative_log_likelihood_sum(y)

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError("y should have the same shape as self.y_pred",
				("y", y.type, "y_pred", self.y_pred.type))
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			print("!!! y should be of int type")
			return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))

	def prediction(self, input):
		return self.y_pred

	def word2vec(self):
		#TODO - loss functinon in outer layer unspired by word2vec
		return ''
 
 

class DatasetMiniBatchIterator(object):
	""" Basic mini-batch iterator """
	def __init__(self, x, y, batch_size=BATCH_SIZE, randomize=False):
		self.x = x
		self.y = y
		self.batch_size = batch_size
		self.randomize = randomize
		from sklearn.utils import check_random_state
		self.rng = check_random_state(42)

	def __iter__(self):
		n_samples = self.x.shape[0]
		if self.randomize:
			for _ in xrange(n_samples / BATCH_SIZE):
				if BATCH_SIZE > 1:
					i = int(self.rng.rand(1) * ((n_samples+BATCH_SIZE-1) / BATCH_SIZE))
				else:
					i = int(math.floor(self.rng.rand(1) * n_samples))
				yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],
						self.y[i*self.batch_size:(i+1)*self.batch_size])
		else:
			for i in xrange((n_samples + self.batch_size - 1)
						/ self.batch_size):
				yield (self.x[i*self.batch_size:(i+1)*self.batch_size],
					self.y[i*self.batch_size:(i+1)*self.batch_size])



class RecurrentNeuralNet(object):
	"""Recurrent Neural network  """
	def __init__(self, numpy_rng, theano_rng=None, 
				n_ins=40*3,
				layers_types=[Linear, ReLU, RecurrentReLU, ReLU, LogisticRegression],
				layers_sizes=[1024, 1024, 1024, 1024],
				n_outs=62 * 3,
				rho=0.95, eps=1.E-6,
				max_norm=0.,
				debugprint=False,
				recurrent_connections=[2]):


		self.layers = []
		self.params = []
		self.pre_activations = [] # SAG specific
		self.n_layers = len(layers_types)
		self.layers_types = layers_types
		assert self.n_layers > 0
		self.max_norm = max_norm
		self._rho = rho  # ''momentum'' for adadelta
		self._eps = eps  # epsilon for adadelta
		self._accugrads = []  # for adadelta
		self._accudeltas = []  # for adadelta
		self._old_dxs = []  # for adadelta with Nesterov
		if SAG:
			self._sag_gradient_memory = []  # for SAG

		if theano_rng == None:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		self.x = T.fmatrix('x')
		self.y = T.ivector('y')

		self.layers_ins = [n_ins] + layers_sizes
		self.layers_outs = layers_sizes + [n_outs]

		layer_input = self.x
        

		for i, layer_type, n_in, n_out in zip(range(len(layers_types)),layers_types,self.layers_ins, self.layers_outs):

			if layer_type==RecurrentReLU and i in recurrent_connections:
				previous_output=layer_input
				#get previous layer's output and weight matrix
				this_layer = layer_type(rng=numpy_rng,
						input=layer_input, in_stack=previous_output,
						n_in=n_in, n_in_stack=n_in, #previous_output's output size.
						n_out=n_out)
				assert hasattr(this_layer, 'output')

				#REMINDER: here's how the recurrent matrix works
				#lin_output = (T.dot(self.input, self.W) 	
				#		+ T.dot(self.input_stack, self.Ws) + self.b)


				self.params.extend(this_layer.params)
				#self.pre_activations.extend(this_layer.pre_activation)# SAG specific TODO 
				self._accugrads.extend([build_shared_zeros(t.shape.eval(),
					'accugrad') for t in this_layer.params])
				self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
					'accudelta') for t in this_layer.params])
				self._old_dxs.extend([build_shared_zeros(t.shape.eval(),
					'old_dxs') for t in this_layer.params])

				if SAG:
					self._sag_gradient_memory.extend([build_shared_zeros(tuple([(x_train.shape[0]+BATCH_SIZE-1) / BATCH_SIZE] + list(t.shape.eval())), 'sag_gradient_memory') for t in this_layer.params])
					#self._sag_gradient_memory.extend([[build_shared_zeros(t.shape.eval(), 'sag_gradient_memory') for _ in xrange(x_train.shape[0] / BATCH_SIZE + 1)] for t in this_layer.params])

				self.layers.append(this_layer)
				layer_input = this_layer.output
			else:
				this_layer = layer_type(rng=numpy_rng,
					input=layer_input, n_in=n_in, n_out=n_out)

				assert hasattr(this_layer, 'output')
				self.params.extend(this_layer.params)
				#self.pre_activations.extend(this_layer.pre_activation)# SAG specific TODO 
				self._accugrads.extend([build_shared_zeros(t.shape.eval(),
					'accugrad') for t in this_layer.params])
				self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
					'accudelta') for t in this_layer.params])
				self._old_dxs.extend([build_shared_zeros(t.shape.eval(),
					'old_dxs') for t in this_layer.params])

				if SAG:
					self._sag_gradient_memory.extend([build_shared_zeros(tuple([(x_train.shape[0]+BATCH_SIZE-1) / BATCH_SIZE] + list(t.shape.eval())), 'sag_gradient_memory') for t in this_layer.params])
					#self._sag_gradient_memory.extend([[build_shared_zeros(t.shape.eval(), 'sag_gradient_memory') for _ in xrange(x_train.shape[0] / BATCH_SIZE + 1)] for t in this_layer.params])

				self.layers.append(this_layer)
				layer_input = this_layer.output

		assert hasattr(self.layers[-1], 'training_cost')
		assert hasattr(self.layers[-1], 'errors')
		# TODO standardize cost
		self.mean_cost = self.layers[-1].negative_log_likelihood(self.y)
		self.cost = self.layers[-1].training_cost(self.y)
		if debugprint:
			theano.printing.debugprint(self.cost)

		self.errors = self.layers[-1].errors(self.y)
		self._prediction = self.layers[-1].prediction(self.x)

	def __repr__(self):
		dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
				zip(self.layers_ins, self.layers_outs))
		return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
				zip(self.layers_types, dimensions_layers_str)))


	def get_SGD_trainer(self):
		""" Returns a plain SGD minibatch trainer with learning rate as param.
		"""
		batch_x = T.fmatrix('batch_x')
		batch_y = T.ivector('batch_y')
		learning_rate = T.fscalar('lr')  # learning rate to use
		# compute the gradients with respect to the model parameters
		# using mean_cost so that the learning rate is not too dependent
		# on the batch size
		gparams = T.grad(self.mean_cost, self.params)

		# compute list of weights updates
		updates = OrderedDict()
		for param, gparam in zip(self.params, gparams):
			if self.max_norm:
				W = param - gparam * learning_rate
				col_norms = W.norm(2, axis=0)
				desired_norms = T.clip(col_norms, 0, self.max_norm)
				updates[param] = W * (desired_norms / (1e-6 + col_norms))
			else:
				updates[param] = param - gparam * learning_rate

		train_fn = theano.function(inputs=[theano.Param(batch_x),
										theano.Param(batch_y),
										theano.Param(learning_rate)],
									outputs=self.mean_cost,
									updates=updates,
									givens={self.x: batch_x, self.y: batch_y})

		return train_fn

	def get_SAG_trainer(self, R=1., alpha=0., debug=False):  # alpha for reg. TODO
		batch_x = T.fmatrix('batch_x')
		batch_y = T.ivector('batch_y')
		ind_minibatch = T.iscalar('ind_minibatch')
		n_seen = T.fscalar('n_seen')
		# compute the gradients with respect to the model parameters
		cost = self.mean_cost
		gparams = T.grad(cost, self.params)
		#sparams = T.grad(cost, self.pre_activations)  # SAG specific

		scaling = numpy.float32(1. / (R / 4. + alpha))

		updates = OrderedDict()
		for accugrad, gradient_memory, param, gparam in zip(
				self._accugrads, self._sag_gradient_memory,
				#self._accugrads, self._sag_gradient_memory[ind_minibatch.eval()],
				self.params, gparams):
			new = gparam + alpha * param
			agrad = accugrad + new - gradient_memory[ind_minibatch]
			# updates[gradient_memory[ind_minibatch]] = new
			updates[gradient_memory] = T.set_subtensor(gradient_memory[ind_minibatch], new)

			updates[param] = param - (scaling / n_seen) * agrad
			updates[accugrad] = agrad

		train_fn = theano.function(inputs=[theano.Param(batch_x), 
			theano.Param(batch_y), theano.Param(ind_minibatch),
			theano.Param(n_seen)],
			outputs=cost,
			updates=updates,
			givens={self.x: batch_x, self.y: batch_y})

		return train_fn

	def get_adagrad_trainer(self):
		""" Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
		"""
		batch_x = T.fmatrix('batch_x')
		batch_y = T.ivector('batch_y')
		learning_rate = T.fscalar('lr')  # learning rate to use
		# compute the gradients with respect to the model parameters
		gparams = T.grad(self.mean_cost, self.params)

		# compute list of weights updates
		updates = OrderedDict()
		for accugrad, param, gparam in zip(self._accugrads, self.params, gparams):
			# c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
			agrad = accugrad + gparam * gparam
			dx = - (learning_rate / T.sqrt(agrad + self._eps)) * gparam
			if self.max_norm:
				W = param + dx
				col_norms = W.norm(2, axis=0)
				desired_norms = T.clip(col_norms, 0, self.max_norm)
				updates[param] = W * (desired_norms / (1e-6 + col_norms))
			else:
				updates[param] = param + dx
			updates[accugrad] = agrad

		train_fn = theano.function(inputs=[theano.Param(batch_x), 
			theano.Param(batch_y),
			theano.Param(learning_rate)],
			outputs=self.mean_cost,
			updates=updates,
			givens={self.x: batch_x, self.y: batch_y})

		return train_fn

	def get_adadelta_trainer(self):
		""" Returns an Adadelta (Zeiler 2012) trainer using self._rho and
		self._eps params.
		"""
		batch_x = T.fmatrix('batch_x')
		batch_y = T.ivector('batch_y')
		# compute the gradients with respect to the model parameters
		gparams = T.grad(self.mean_cost, self.params)

		# compute list of weights updates
		updates = OrderedDict()
		for accugrad, accudelta, param, gparam in zip(self._accugrads,
				self._accudeltas, self.params, gparams):
			# c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
			agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
			dx = - T.sqrt((accudelta + self._eps)
						/ (agrad + self._eps)) * gparam
			updates[accudelta] = (self._rho * accudelta
			                      + (1 - self._rho) * dx * dx)
			if self.max_norm:
				W = param + dx
				col_norms = W.norm(2, axis=0)
				desired_norms = T.clip(col_norms, 0, self.max_norm)
				updates[param] = W * (desired_norms / (1e-6 + col_norms))
			else:
				updates[param] = param + dx
			updates[accugrad] = agrad

		train_fn = theano.function(inputs=[theano.Param(batch_x),
											theano.Param(batch_y)],
										outputs=self.mean_cost,
										updates=updates,
										givens={self.x: batch_x, self.y: batch_y})

		return train_fn

	def nesterov_step(self):
		beta = 0.5
		updates = OrderedDict()
		for param, dx in zip(self.params, self._old_dxs):
			updates[param] = param + beta * dx
		nesterov_fn = theano.function(inputs=[],
				outputs=[],
				updates=updates,
				givens={})
		nesterov_fn()

	def get_adadelta_nesterov_trainer(self):
		""" Returns an Adadelta (Zeiler 2012) trainer using self._rho and
		self._eps params.
		"""
		batch_x = T.fmatrix('batch_x')
		batch_y = T.ivector('batch_y')
		# compute the gradients with respect to the model parameters
		gparams = T.grad(self.mean_cost, self.params)
		beta = 0.5

		# compute list of weights updates
		updates = OrderedDict()
		for accugrad, accudelta, old_dx, param, gparam in zip(self._accugrads,
			    self._accudeltas, self._old_dxs, self.params, gparams):
			# c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
			agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
			dx = - T.sqrt((accudelta + self._eps)
					/ (agrad + self._eps)) * gparam
			updates[accudelta] = (self._rho * accudelta
					+ (1 - self._rho) * dx * dx)
			if self.max_norm:
				W = param + dx - beta*old_dx
				#W = param + dx
				col_norms = W.norm(2, axis=0)
				desired_norms = T.clip(col_norms, 0, self.max_norm)
				updates[param] = W * (desired_norms / (1e-6 + col_norms))
			else:
				updates[param] = param + dx - beta*old_dx
				#updates[param] = param + dx
			updates[old_dx] = dx
			updates[accugrad] = agrad

		train_fn = theano.function(inputs=[theano.Param(batch_x),
										theano.Param(batch_y)],
									outputs=self.mean_cost,
									updates=updates,
									givens={self.x: batch_x, self.y: batch_y})

		return train_fn

	def get_adadelta_rprop_trainer(self):
		""" TODO
		"""
		# TODO working on that
		batch_x = T.fmatrix('batch_x')
		batch_y = T.ivector('batch_y')
		# compute the gradients with respect to the model parameters
		gparams = T.grad(self.mean_cost, self.params)

		# compute list of weights updates
		updates = OrderedDict()
		for accugrad, accudelta, param, gparam in zip(self._accugrads,
			    self._accudeltas, self.params, gparams):
			# c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
			agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
			dx = - T.sqrt((accudelta + self._eps)
					/ (agrad + self._eps)) * T.switch(gparam < 0, -1., 1.)
			updates[accudelta] = (self._rho * accudelta
			                      + (1 - self._rho) * dx * dx)
			if self.max_norm:
				W = param + dx
				col_norms = W.norm(2, axis=0)
				desired_norms = T.clip(col_norms, 0, self.max_norm)
				updates[param] = W * (desired_norms / (1e-6 + col_norms))
			else:
				updates[param] = param + dx
			updates[accugrad] = agrad

		train_fn = theano.function(inputs=[theano.Param(batch_x),
											theano.Param(batch_y)],
									outputs=self.mean_cost,
									updates=updates,
									givens={self.x: batch_x, self.y: batch_y})

		return train_fn

	def score_classif(self, given_set):
		""" Returns functions to get current classification errors. """
		batch_x = T.fmatrix('batch_x')
		batch_y = T.ivector('batch_y')
		score = theano.function(inputs=[theano.Param(batch_x),
										theano.Param(batch_y)],
									outputs=self.errors,
									givens={self.x: batch_x, self.y: batch_y})

		def scoref():
			""" returned function that scans the entire set given as input """
			return [score(batch_x, batch_y) for batch_x, batch_y in given_set]

		return scoref

	def predict(self,X):
		""" Returns functions to get current classification errors. """
		batch_x = T.fmatrix('batch_x')#fmatrix
		predictor = theano.function(inputs=[theano.Param(batch_x)],
									outputs=self.prediction_1,
									givens={self.x: batch_x})
		return predictor(X)





class RegularizedNet(RecurrentNeuralNet):
	""" Neural net with L1 and L2 regularization """
	def __init__(self, numpy_rng, theano_rng=None,
				 n_ins=100,
				 layers_types=[Linear, ReLU, RecurrentReLU, ReLU, LogisticRegression],
				 layers_sizes=[1024, 1024, 1024, 1024],
				 n_outs=2,
				 rho=0.9,
				 eps=1.E-6,
				 L1_reg=0.,
				 L2_reg=0.,
				 max_norm=0.,
				 debugprint=False,
				 recurrent_connections=[2]):
		"""
		Feedforward neural network with added L1 and/or L2 regularization.
		"""
		super(RegularizedNet, self).__init__(numpy_rng, theano_rng, n_ins,
		        layers_types, layers_sizes, n_outs, rho, eps, max_norm,
		        debugprint)

		L1 = shared(0.)
		for param in self.params:
			L1 += T.sum(abs(param))
		if L1_reg > 0.:
			self.cost = self.cost + L1_reg * L1
		L2 = shared(0.)
		for param in self.params:
			L2 += T.sum(param ** 2)
		if L2_reg > 0.:
		    self.cost = self.cost + L2_reg * L2



#########

def add_fit_and_score(class_to_chg):
	""" Mutates a class to add the fit() and score() functions to a NeuralNet.
	"""
	from types import MethodType
	def fit(self, x_train, y_train, x_dev=None, y_dev=None,
			max_epochs=20, early_stopping=True, split_ratio=0.1, # TODO 100+ epochs
			method='adadelta', verbose=False, plot=False):
		"""
		TODO
		"""
		import time, copy
		if x_dev == None or y_dev == None:
			from sklearn.cross_validation import train_test_split
			x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
					test_size=split_ratio, random_state=42)
		if method == 'sgd':
			train_fn = self.get_SGD_trainer()
		elif method == 'adagrad':
			train_fn = self.get_adagrad_trainer()
		elif method == 'adadelta':
			train_fn = self.get_adadelta_trainer()
		elif method == 'adadelta_nesterov':
			train_fn = self.get_adadelta_nesterov_trainer()
		elif method == 'adadelta_rprop':
			train_fn = self.get_adadelta_rprop_trainer()
		elif method == 'sag':
			#train_fn = self.get_SAG_trainer(R=1+numpy.max(numpy.sum(x_train**2, axis=1)))
			if BATCH_SIZE > 1:
				line_sums = numpy.sum(x_train**2, axis=1)
				train_fn = self.get_SAG_trainer(R=numpy.max(numpy.mean(
					line_sums[:(line_sums.shape[0]/BATCH_SIZE)*BATCH_SIZE].reshape((line_sums.shape[0]/BATCH_SIZE,
							BATCH_SIZE)), axis=1)),
						alpha=1./x_train.shape[0])
			else:
				train_fn = self.get_SAG_trainer(R=numpy.max(numpy.sum(x_train**2,
					axis=1)), alpha=1./x_train.shape[0])
		train_set_iterator = DatasetMiniBatchIterator(x_train, y_train)
		if method == 'sag':
			sag_train_set_iterator = DatasetMiniBatchIterator(x_train, y_train, randomize=True)
		dev_set_iterator = DatasetMiniBatchIterator(x_dev, y_dev)
		train_scoref = self.score_classif(train_set_iterator)
		dev_scoref = self.score_classif(dev_set_iterator)
		best_dev_loss = numpy.inf
		epoch = 0

		patience = 1000  # look as this many examples regardless TODO
		patience_increase = 2.  # wait this much longer when a new best is # found
		improvement_threshold = 0.995  # a relative improvement of this much is # considered significant

		done_looping = False
		print '... training the model'
		# early-stopping parameters

		test_score = 0.
		start_time = time.clock()

		done_looping = False
		epoch = 0
		timer = None


		# TODO early stopping (not just cross val, also stop training)
		if plot:
			verbose = True
			self._costs = []
			self._train_errors = []
			self._dev_errors = []
			self._updates = []

		seen = numpy.zeros(((x_train.shape[0]+BATCH_SIZE-1) / BATCH_SIZE,), dtype=numpy.bool)
		n_seen = 0

		while (epoch < max_epochs) and (not done_looping):
			if not verbose:
				sys.stdout.write("\r%0.2f%%" % (epoch * 100./ max_epochs))
				sys.stdout.flush()
			avg_costs = []
			timer = time.time()
			if method == 'sag':
				for ind_minibatch, x, y in sag_train_set_iterator:
					if not seen[ind_minibatch]:
						seen[ind_minibatch] = 1
						n_seen += 1
					if 'nesterov' in method:
						self.nesterov_step()
					avg_cost = train_fn(x, y, ind_minibatch, n_seen)
					if type(avg_cost) == list:
						avg_costs.append(avg_cost[0])
					else:
						avg_costs.append(avg_cost)
			else:
				for iteration, (x, y) in enumerate(train_set_iterator):
					if method == 'sgd' or method == 'adagrad':
						avg_cost = train_fn(x, y, lr=1.E-2)
					elif 'adadelta' in method:
						avg_cost = train_fn(x, y)
					if type(avg_cost) == list:
						avg_costs.append(avg_cost[0])
					else:
						avg_costs.append(avg_cost)
			if verbose:
				mean_costs = numpy.mean(avg_costs)
				mean_train_errors = numpy.mean(train_scoref())
				print('  epoch %i took %f seconds' %
					(epoch, time.time() - timer))
				print('  epoch %i, avg costs %f' %
					(epoch, mean_costs))
				print('  method %s, epoch %i, training error %f' %
					(method, epoch, mean_train_errors))
				if plot:
					self._costs.append(mean_costs)
					self._train_errors.append(mean_train_errors)
			dev_errors = numpy.mean(dev_scoref())
			if plot:
				self._dev_errors.append(dev_errors)
			if dev_errors < best_dev_loss:
				best_dev_loss = dev_errors
				best_params = copy.deepcopy(self.params)
				if verbose:
					print('!!!  epoch %i, validation error of best model %f' %
						(epoch, dev_errors))
			epoch += 1
			if patience <= iteration:  # TODO correct that
				done_looping = True
				break
		if not verbose:
			print("")
		for i, param in enumerate(best_params):
			self.params[i] = param

	def score(self, x, y):
		""" error rates """
		iterator = DatasetMiniBatchIterator(x, y)
		scoref = self.score_classif(iterator)
		return numpy.mean(scoref())
	class_to_chg.fit = MethodType(fit, None, class_to_chg)
	class_to_chg.score = MethodType(score, None, class_to_chg)




def News_test():
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    newsgroups_train = datasets.fetch_20newsgroups(subset='train')
    vectorizer = CountVectorizer(encoding='latin-1', max_features=30000)
    #vectorizer = HashingVectorizer(encoding='latin-1')
    x_train = vectorizer.fit_transform(newsgroups_train.data)
    x_train = numpy.asarray(x_train.todense(), dtype='float32')
    y_train = numpy.asarray(newsgroups_train.target, dtype='int32')
    newsgroups_test = datasets.fetch_20newsgroups(subset='test')
    x_test = vectorizer.transform(newsgroups_test.data)
    x_test = numpy.asarray(x_test.todense(), dtype='float32')
    y_test = numpy.asarray(newsgroups_test.target, dtype='int32')
    add_fit_and_score(RegularizedNet)
    dnn=RegularizedNet(numpy_rng=numpy.random.RandomState(123), theano_rng=None, 
            n_ins=x_train.shape[1],
            n_outs=len(set(y_train)),
            rho=0.95, 
            eps=1.E-6,
            max_norm=0.,
            debugprint=False,
            L1_reg=0.,
            L2_reg=1./x_train.shape[0])
    print len(set(y_train))
    dnn.fit(x_train, y_train, max_epochs=30, method='adadelta', verbose=True, plot=True)
    test_error = dnn.score(x_test, y_test)
    print("score: %f" % (1. - test_error))
News_test()
