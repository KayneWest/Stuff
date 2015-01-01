import pandas as pd
import sqlite3
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
import numpy, theano, sys, math
from theano import tensor as T
from theano import shared
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
from sklearn.svm import LinearSVC,SVC
import numpy as np
import time



'''[relu,relu,relu,recurrent[x,x],relu, ouput]
nesterov mommentum
Connectionist temporal classification loss function loss function
#https://blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
regularized

5 layers of hidden units

We use momentum of 0.99 and anneal the learning rate by a constant factor, chosen to yield the fastest
convergence, after each epoch through the data.

Dropout between 5-10%

Such jittering is not common in ASR, however we found it beneficial to
translate the raw audio files by 5ms (half the filter bank step size) to the left and right, then forward
propagate the recomputed features and average the output probabilities

Q(c) = log(P(c|x)) + α log(Plm(c)) + β word count(c)

where α and β are tunable parameters (set by cross-validation) that control the trade-off between
the RNN, the language model constraint and the length of the sentence. The term Plm denotes the
probability of the sequence c according to the N-gram model. We maximize this objective using a
highly optimized beam search algorithm, with a typical beam size in the range 1000-8000—similar
to the approach described by Hannun et al. [16].

'''

BATCH_SIZE=200
#SAG = Stochastic Average Gradient 
SAG = False

def softplus_f(v):
	"""activation for a softplus layer, not here"""
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

def relu_f(vec): #Clipped ReLU from DeepSpeech paper
	""" min{max{0, z}, 20} is the clipped rectified-linear (ReLu) activation function """
	#original return (vec + abs(vec)) / 2.
	#return np.minimum(np.maximum(0,vec),20)
	return T.clip(vec,0,20)

def build_shared_zeros(shape, name):
	""" Builds a theano shared variable filled with a zeros numpy array """
	return shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
		name=name, borrow=True)
 
 #translate the raw audio files by 5ms (half the filter bank step size)
 # to the left and right, then forward propagate the recomputed features 
 #and average the output probabilities
def j_shift(curr_shape, shiftX, shiftY):
    """ Helper to modify the in_shape tuple by jitter amounts """
    return curr_shape[:-2] + (curr_shape[-2] - shiftY, curr_shape[-1] - shiftX)

class ReLU(object):
	""" clipped ReLU layer, ouput min(max(0,x),20)"""
	def __init__(self, rng, input, n_in, n_out, dropout=0.0, W=None, b=None, fdrop=False):
		if W is None:
			W_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			W_values *= 4  
			W = theano.shared(value=W_values, name='W', borrow=True)
		if b is None:
			b = build_shared_zeros((n_out,), 'b')
		self.input = input
		self.W = W
		self.b = b
		self.params = [self.W, self.b]
		self.output = T.dot(self.input, self.W) + self.b
		self.pre_activation = self.output
		if fdrop:
			self.pre_activation = fast_dropout(rng, self.pre_activation)
		self.output = relu_f(self.pre_activation)

	def __repr__(self):
		return "ReLU"

class RecurrentBackForwardReLU(object):
	''' This is bidirectional rnn layer that has 
		one forward layer made up of one time step of size [n_in,n_in] and 
		one backward layer made up of [n_in,n_in]. essentially 4 layers in one.
		'''
	def __init__(self, rng, input, n_in, n_out, W=None, Wf=None, 
						Wb=None, b=None, bf=None, bb=None, 
						U_forward=None, U_backward=None, fdrop=False):  
		####  init of bias/weight parameters  #####
		if W is None:
			W_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			W_values *= 4  
			W = theano.shared(value=W_values, name='W', borrow=True)
		self.W = W
		if Wf is None:
			Wf_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			Wf_values *= 4  # TODO check
			Wf = shared(value=Wf_values, name='Wf', borrow=True)
		self.Wf = Wf  # weights of the reccurrent forwards (normal) connection
		if Wb is None:
			Wb_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			Wb_values *= 4  # TODO check
			Wb = shared(value=Wb_values, name='Wb', borrow=True)
		self.Wb = Wb  # weights of the reccurrent backwards connection
		if b is None:
			b = build_shared_zeros((n_out,), 'b')
		self.b = b
		if bf is None:
			bf = build_shared_zeros((n_out,), 'bf')
		self.bf = bf
		if bb is None:
			bb = build_shared_zeros((n_out,), 'bb')
		self.bb = bb
		if U_forward is None:
			U_forward = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			U_forward *= 4  # TODO check
			U_forward = shared(value=U_forward, name='U_forward', borrow=True)
		self.U_forward = U_forward  # weights of the reccurrent backwards connection
		if U_backward is None:
			U_backward = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			U_backward *= 4  # TODO check
			U_backward = shared(value=U_backward, name='U_backward', borrow=True)
		self.U_backward = U_backward  # weights of the reccurrent backwards connection

		####  init of bias/weight parameters  ##### 
		self.input = input #forwards is self.input. no point in doubling the memory

		self.params = [self.W, self.Wf, self.Wb, self.b, self.bf, self.bb, self.U_backward, self.U_forward] 

		self.h0_forward = theano.shared(value=np.zeros(n_in, dtype=floatX), name='h0_forward', borrow=True)
		self.h0_backward = theano.shared(value=np.zeros(n_in, dtype=floatX), name='h0_backward', borrow=True)

		# Forward and backward representation over time with 1 step. 
		self.h_forward, _ = theano.scan(fn=self.forward_step, sequences=self.input, outputs_info=[self.h0_forward],
											n_steps=1)
		self.h_backward, _ = theano.scan(fn=self.backward_step, sequences=self.input, outputs_info=[self.h0_backward], 
											n_steps=1, go_backwards=True)
		# if you want Averages,  
		#self.h_forward = T.mean(self.h_forwards, axis=0)
		#self.h_backward = T.mean(self.h_backwards, axis=0)
		# Concatenate
		self.concat = T.concatenate([self.h_forward, self.h_backward], axis=0)
		#self.concat = self.h_forward + self.h_backward
		self.output = relu_f(T.dot(self.concat, self.W) + self.b)

	def forward_step(self, x_t, h_tm1):
		h_t = relu_f(T.dot(x_t, self.Wf) + \
								T.dot(h_tm1, self.U_forward) + self.bf)
		return h_t

	def backward_step(self, x_t, h_tm1):
		h_t = relu_f(T.dot(x_t, self.Wb) + \
								T.dot(h_tm1, self.U_backward) + self.bb)
		return h_t

	def __repr__(self):
		return "RecurrentBackForwardReLU"


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
				yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],self.y[i*self.batch_size:(i+1)*self.batch_size])
		else:
			for i in xrange((n_samples + self.batch_size - 1)
							/ self.batch_size):
				yield (self.x[i*self.batch_size:(i+1)*self.batch_size],self.y[i*self.batch_size:(i+1)*self.batch_size])


class Output_Layer:
	"""
	Output_Layer with a CTC loss function 
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
		#this is the prediction. pred
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.output = self.y_pred
		self.params = [self.W, self.b]

		#stuff for CTC outside of function:
		self.n_out=n_out

	#blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
	def recurrence_relation(self, y):
		def sec_diag_i(yt, ytp1, ytp2):
			return T.neq(yt, ytp2) * T.eq(ytp1, self.n_out)

		y_extend = T.concatenate((y, [self.n_out, self.n_out]))
		sec_diag, _ = theano.scan(sec_diag_i,
				sequences={'input':y_extend, 'taps':[0, 1, 2]})

		y_sz = y.shape[0]
		return T.eye(y_sz) + \
			T .eye(y_sz, k=1) + \
			T.eye(y_sz, k=2) * sec_diag.dimshuffle((0, 'x'))

	def forward_path_probabs(self, y):
		pred_y = self.p_y_given_x[:, y]
		rr = self.recurrence_relation(y)#, self.n_out)

		def step(p_curr, p_prev):
			return p_curr * T.dot(p_prev, rr)

		probabilities, _ = theano.scan(step, sequences=[pred_y],
							outputs_info=[T.eye(y.shape[0])[0]])

		return probabilities

	def backword_path_probabs(self, y):
		pred_y = self.p_y_given_x[::-1][:, y[::-1]]
		rr = self.recurrence_relation(y[::-1])#, self.n_out)

		def step(p_curr, p_prev):
			return p_curr * T.dot(p_prev, rr)

		probabilities, _ = theano.scan(step,sequences=[pred_y],
							outputs_info=[T.eye(y[::-1].shape[0])[0]])

		return probabilities[::-1,::-1]

	def connectionist_temporal_classification(self,y):
		forward_probs  = self.forward_path_probabs(y)
		backward_probs = self.backword_path_probabs(y) #backwards prediction
		probs = forward_probs * backward_probs / self.p_y_given_x[:,y]
		total_prob = T.sum(probs)
		print total_prob
		return -T.log(total_prob)

	def ctc_cost(self,y):
		""" wraper for connectionist_temporal_classification"""
		return self.connectionist_temporal_classification(y)

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

	def predict_result(self, thing):
		p_y_given_x = T.nnet.softmax(T.dot(thing, self.W) + self.b)
		output = T.argmax(p_y_given_x, axis=1)
		return output

class DeepSpeech(object):
	"""DeepSpeech http://arxiv.org/pdf/1412.5567v1.pdf Recurrent Neural network 
	6 layered network:
		1-3: first layers are standard ReLU layers with .05-.10 dropout.
		  4: Backward-Forward layer is a small bidirectional network layer. (a network within the network)
		  5: Standard ReLU layer with .05-.10 dropout
		  6: Output layer with Connectionist Temporal Classification loss
	"""
	def __init__(self, numpy_rng, theano_rng=None, 
				n_ins=40*3,
				layers_types=[ReLU, ReLU, ReLU, RecurrentBackForwardReLU, ReLU, Output_Layer],
				layers_sizes=[1024, 1024, 1024, 1024, 1024],
				n_outs=62 * 3,
				rho=0.95, 
				eps=1.E-6,
				max_norm=0.,
				debugprint=False,
				dropout_rates=[0.05, 0.1, 0.05, 0.0, 0.1],
				fast_drop=True):
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
		
		#################################################################
		############ first pass without dropout   #######################
		#################################################################

		for layer_type, n_in, n_out in zip(layers_types,self.layers_ins, self.layers_outs):
			if layer_type==RecurrentBackForwardReLU:
				###########previous_output=layer_input
				#get previous layer's output and weight matrix
				this_layer = layer_type(rng=numpy_rng,
						input=layer_input, n_in=n_in, n_out=n_out)
				#print this_layer
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
			else:
				this_layer = layer_type(rng=numpy_rng,
							input=layer_input, n_in=n_in, 
							n_out=n_out)
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
					self._sag_gradient_memory.extend([build_shared_zeros(tuple([(x_train.shape[0]+BATCH_SIZE-1) \
						/ BATCH_SIZE] + list(t.shape.eval())), 'sag_gradient_memory') for t in this_layer.params])
					#self._sag_gradient_memory.extend([[build_shared_zeros(t.shape.eval(), 'sag_gradient_memory') for _ in xrange(x_train.shape[0] / BATCH_SIZE + 1)] for t in this_layer.params])
				self.layers.append(this_layer)
				layer_input = this_layer.output
		
		#################################################################
		#################    second pass with dropout    ################
		#################################################################
		
		self.dropout_rates = dropout_rates
		if fast_drop:
			if dropout_rates[0]:
				dropout_layer_input = fast_dropout(numpy_rng, self.x)
			else:
				dropout_layer_input = self.x
		else:
			dropout_layer_input = dropout(numpy_rng, self.x, p=dropout_rates[0])
		self.dropout_layers = []

		for layer, layer_type, n_in, n_out, dr in zip(self.layers,
			layers_types, self.layers_ins, self.layers_outs,
			dropout_rates + [0]): #no dropout in last layer
			#print layer, layer_type, n_in, n_out, dr
			if dr:
				if fast_drop:
					this_layer = layer_type(rng=numpy_rng,
						input=dropout_layer_input, n_in=n_in, n_out=n_out,
							W=layer.W, b=layer.b, fdrop=True)
				else:
					this_layer = layer_type(rng=numpy_rng,
						input=dropout_layer_input, n_in=n_in, n_out=n_out,
							W=layer.W * 1. / (1. - dr),
							b=layer.b * 1. / (1. - dr))
					# Dropout with dr==1 does not drop anything
					this_layer.output = dropout(numpy_rng, this_layer.output, dr)
			else:
				if layer_type==RecurrentBackForwardReLU:
					this_layer = layer_type(rng=numpy_rng,
						input=dropout_layer_input, n_in=n_in, n_out=n_out,
						W=layer.W, Wf=layer.Wf, Wb=layer.Wb,
						b=layer.b, bf=layer.bf, bb=layer.bb,
						U_forward=layer.U_forward,U_backward=layer.U_backward)
				else:
					this_layer = layer_type(rng=numpy_rng,
							input=dropout_layer_input, n_in=n_in, n_out=n_out,
							W=layer.W, b=layer.b)
			assert hasattr(this_layer, 'output')
			self.dropout_layers.append(this_layer)
			dropout_layer_input = this_layer.output

		assert hasattr(self.layers[-1], 'ctc_cost')
		assert hasattr(self.layers[-1], 'errors')

		#labeled mean costs for the average that's eventually taken 
		#self.mean_cost could be renamed self.cost, but whatevs
		#these are the dropout costs
		self.mean_cost = self.dropout_layers[-1].ctc_cost(self.y)

		if debugprint:
			theano.printing.debugprint(self.cost)

		# these is the non-dropout errors
		self.errors = self.layers[-1].errors(self.y)
		self._prediction = self.layers[-1].prediction(self.x)

	def __repr__(self):
		dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
				zip(self.layers_ins, self.layers_outs))
		return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
				zip(self.layers_types, dimensions_layers_str)))

	#https://gist.github.com/SnippyHolloW/8a0f820261926e2f41cc 
	#learning methods based on SnippyHollow code
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


####################################################################################
####################################################################################
####################################################################################
####################################################################################

####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

def add_fit_and_score_early_stop(class_to_chg):
	""" Mutates a class to add the fit() and score() functions to a NeuralNet.
	"""
	from types import MethodType
	def fit(self, x_train, y_train, x_dev=None, y_dev=None,
			max_epochs=20, early_stopping=True, split_ratio=0.1, # TODO 100+ epochs
			method='adadelta', verbose=False, plot=False):
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
		# TODO early stopping (not just cross val, also stop training)
		if plot:
			verbose = True
			self._costs = []
			self._train_errors = []
			self._dev_errors = []
			self._updates = []
 
		seen = numpy.zeros(((x_train.shape[0]+BATCH_SIZE-1) / BATCH_SIZE,), dtype=numpy.bool)
		n_seen = 0

		patience = 1000  
		patience_increase = 2.  # wait this much longer when a new best is found
		improvement_threshold = 0.995  # a relative improvement of this much is considered significant
		done_looping = False
		print '... training the model'
		test_score = 0.
		start_time = time.clock()
		done_looping = False
		epoch = 0
		timer = None
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
			if patience <= iteration:
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
