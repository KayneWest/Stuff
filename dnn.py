


#need to make sure to have better weight initialization:
#        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
#        self.weights = [np.random.randn(y, x)/np.sqrt(x) 
#                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
 
import numpy, theano, sys, math
from theano import tensor as T
from theano import shared
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
 
BATCH_SIZE = 20
 
 
def relu_f(vec):
    """ Wrapper to quickly change the rectified linear unit function """
    return (vec + abs(vec)) / 2.
 
 
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
 
 ####   LENET5   ########
##learning_rate = 0.1
##rng = numpy.random.RandomState(23455)
##ishape = (28, 28)  # this is the size of MNIST images
##batch_size = 20  # sized of the minibatch
# allocate symbolic variables for the data
##x = T.matrix('x')  # rasterized images
##y = T.lvector('y')  # the labels are presented as 1D vector of [long int] labels
##############################
# BEGIN BUILDING ACTUAL MODE
##############################
# Reshape matrix of rasterized images of shape (batch_size,28*28)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
##layer0_input = x.reshape((batch_size,1,28,28))
# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
# maxpooling reduces this further to (24/2,24/2) = (12,12)
# 4D output tensor is thus of shape (20,20,12,12)
##layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
        ##image_shape=(batch_size, 1, 28, 28),
        ##filter_shape=(20, 1, 5, 5), poolsize=(2, 2))
# Construct the second convolutional pooling layer
# filtering reduces the image size to (12 - 5 + 1, 12 - 5 + 1)=(8, 8)
# maxpooling reduces this further to (8/2,8/2) = (4, 4)
# 4D output tensor is thus of shape (20,50,4,4)
##layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
        ##image_shape=(batch_size, 20, 12, 12),
        ##filter_shape=(50, 20, 5, 5), poolsize=(2, 2))
# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size,num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (20, 32 * 4 * 4) = (20, 512)
##layer2_input = layer1.output.flatten(2)
# construct a fully-connected sigmoidal layer
##layer2 = HiddenLayer(rng, input=layer2_input,
                     ##n_in=50 * 4 * 4, n_out=500,
                     ##activation=T.tanh    )
# classify the values of the fully-connected sigmoidal layer
##layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
# the cost we minimize during training is the NLL of the model
##cost = layer3.negative_log_likelihood(y)
# create a function to compute the mistakes that are made by the model
##test_model = theano.function([x, y], layer3.errors(y))
# create a list of all model parameters to be fit by gradient descent
##params = layer3.params + layer2.params + layer1.params + layer0.params
# create a list of gradients for all model parameters
##grads = T.grad(cost, params)
# train_model is a function that updates the model parameters by SGD
# Since this model has many parameters, it would be tedious to manually
# create an update rule for each model parameter. We thus create the updates
# dictionary by automatically looping over all (params[i],grads[i])  pairs.
##updates = []
##for param_i, grad_i in zip(params, grads):
    ##updates.append((param_i, param_i - learning_rate * grad_i))
##train_model = theano.function([index], cost, updates = updates,
        ##givens={
            ##x: train_set_x[index * batch_size: (index + 1) * batch_size],
            ##y: train_set_y[index * batch_size: (index + 1) * batch_size]})
            
            
class ConvolutionalLayer(object):
    """Convolutional Layer with pooling"""
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),fdrop=False):
        """
        Allocate a ConvolutionalLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # initialize weight values: the fan-in of each hidden neuron is
        # restricted by the size of the receptive fields.
        fan_in =  numpy.prod(filter_shape[1:])
        W_values = numpy.asarray(rng.uniform(
              low=-numpy.sqrt(3./fan_in),
              high=numpy.sqrt(3./fan_in),
              size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W')

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b')

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input, self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(conv_out, poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will thus
        # be broadcasted across mini-batches and feature map width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
    def __repr__(self):
        return "ConvolutionalLayer" #might have to change this
        
        
class ConvolutionalLayer1(object):
    """Convolutional Layer with pooling"""
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),fdrop=False):
        """
        Allocate a ConvolutionalLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        print "image_shape and filtershape",image_shape,filter_shape
        assert image_shape[1] == filter_shape[1]
        self.input = input
        

        # initialize weight values: the fan-in of each hidden neuron is
        # restricted by the size of the receptive fields.
        fan_in =  numpy.prod(filter_shape[1:])
        W_values = numpy.asarray(rng.uniform(
              low=-numpy.sqrt(3./fan_in),
              high=numpy.sqrt(3./fan_in),
              size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W')

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b')

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input, self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(conv_out, poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will thus
        # be broadcasted across mini-batches and feature map width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
    def __repr__(self):
        return "ConvolutionalLayer" #might have to change this
        
        
class ConvolutionalLayer2(object):
    """Convolutional Layer with pooling"""
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),fdrop=False):
        """
        Allocate a ConvolutionalLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # initialize weight values: the fan-in of each hidden neuron is
        # restricted by the size of the receptive fields.
        fan_in =  numpy.prod(filter_shape[1:])
        W_values = numpy.asarray(rng.uniform(
              low=-numpy.sqrt(3./fan_in),
              high=numpy.sqrt(3./fan_in),
              size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W')

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b')

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input, self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(conv_out, poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will thus
        # be broadcasted across mini-batches and feature map width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
    def __repr__(self):
        return "ConvolutionalLayer" #might have to change this        
        

#ConvolutionalLayer(rng, input=layer0_input, image_shape=(batch_size, 1, 28, 28),filter_shape=(20, 1, 5, 5), poolsize=(2, 2))
#layer2_input = layer1.output.flatten(2)

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
 
 
class LogisticRegression:
    """Multi-class Logistic Regression
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
 

class LogisticRegression_crossentropy:
    """Multi-class Logistic Regression
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
 
    def cross_entropy(self, y): #TODO enure that this isn't categorical cross_entropy
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y)) 
 
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        #return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    def cross_entropy_sum(self, y): #TODO enure that this isn't categorical cross_entropy
        return T.sum(T.nnet.binary_crossentropy(self.p_y_given_x, y)) 
 
    def negative_log_likelihood_sum(self, y):
        return -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
 
    def cross_entropy_training_cost(self, y):
        """ Wrapper for standard name """
        return self.cross_entropy_sum(y)
 
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


 
class NeuralNet(object):
    """ Neural network (not regularized, without dropout) """
    def __init__(self, numpy_rng, theano_rng=None, 
                 n_ins=40*3,
                 layers_types=[Linear, ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=[1024, 1024, 1024, 1024],
                 n_outs=62 * 3,
                 rho=0.9,
                 eps=1.E-6,
                 max_norm=0.,
                 debugprint=False):
        """
        Basic feedforward neural network.
        """
        self.layers = []
        self.params = []
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self.max_norm = max_norm
        self._rho = rho  # "momentum" for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta
 
        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
 
        self.x = T.fmatrix('x')
        self.y = T.ivector('y')
        
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        
        layer_input = self.x
        
        for layer_type, n_in, n_out in zip(layers_types,
                self.layers_ins, self.layers_outs):
            this_layer = layer_type(rng=numpy_rng,
                    input=layer_input, n_in=n_in, n_out=n_out)
            assert hasattr(this_layer, 'output')
            self.params.extend(this_layer.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer.params])
 
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


class AlexNet(object):
    """ Convolutional Neural network (not regularized, without dropout) """
    def __init__(self, numpy_rng, theano_rng=None, 
                 n_ins=40*3,#3
                 # add two conv layers and their paramsConvolutionalLayer,ConvolutionalLayer,
                 layers_types=[ConvolutionalLayer1,ConvolutionalLayer2, ReLU, ReLU, ReLU, LogisticRegression],#LogisticRegression_crossentropy
                 layers_sizes=[1024, 1024, 1024, 1024, 1024], #play with these sizes
                 n_outs=62 * 3, #3
                 rho=0.9,
                 eps=1.E-6,
                 max_norm=0.,
                 debugprint=False):
        """
        Basic feedforward convolutional neural network.
        """
        self.layers = []
        self.params = []
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self.max_norm = max_norm
        self._rho = rho  # "momentum" for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta
 
        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
 
        #self.x = T.fmatrix('x') #fmatrix=float32  TODO NEED TO GET THE CONVOLUTIONS TO USE FLOAT32
        self.x = T.dmatrix('x') #dmatrix=float64
        #self.y = T.ivector('y') #ivector=int32
        self.y = T.lvector('y') #lvector=int64
        
        #might have to change this little buggger irght here to match the convs
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        
        layer_input = self.x
        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        self.batch_size = BATCH_SIZE
        conv_layer_input=layer_input.reshape((self.batch_size,1,28,28)) #change later params
        
        
        #change these for each conv layer, and specify params
        self.poolsize=(2, 2)
        
        ###THIS IS FOR THE MNIST DATASET### 
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (20,20,12,12)
        image_shape1=(self.batch_size, 1, 28, 28)
        self.filter_shape1=(20, 1, 5, 5)
        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12 - 5 + 1, 12 - 5 + 1)=(8, 8)
        # maxpooling reduces this further to (8/2,8/2) = (4, 4)
        # 4D output tensor is thus of shape (20,50,4,4)
        image_shape2=(self.batch_size, 100, 12, 12)
        self.filter_shape2=(50, 100, 5, 5)
        # the non-convolutional layers being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20, 32 * 4 * 4) = (20, 512)
        #relu_input = convolutional.output.flatten(2)
        # n_in=50 * 4 * 4, # 4D output tensor is thus of shape (20,#50,#4,#4)
        
        
        ######NEED TO FIGURE OUT A GOOD WAY TO DO THIS IN ONE FELL SWOOP####
        
        
        #it's all about making n_in and n_out line up so that a weight matrix can be deployed properly
        #right now, the weight mtrx from conv1->conv2 are fine, but from conv2->relu, the sizes are differrnt
        #and thus are bad
        
        
        
        for layer_type, n_in, n_out in zip(layers_types,
                self.layers_ins, self.layers_outs):
           if layer_type==ConvolutionalLayer1: #if convlayer1,convlayer2,etc. then change params with forloop,
               #last layer must have output.flatten(2) as the summation of the layer to be used with the ReLU layers
               this_layer = layer_type(rng=numpy_rng,
                    input=conv_layer_input, filter_shape=(20, 1, 5, 5), image_shape=(self.batch_size, 1, 28, 28), poolsize=self.poolsize,fdrop=False)
               assert hasattr(this_layer, 'output')
               self.params.extend(this_layer.params)
               self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                   'accugrad') for t in this_layer.params])
               self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                   'accudelta') for t in this_layer.params])
               self.layers.append(this_layer)
               layer_input = this_layer.output
               print this_layer,layer_input
           elif layer_type==ConvolutionalLayer2: #if convlayer1,convlayer2,etc. then change params with forloop
               this_layer = layer_type(rng=numpy_rng,
                    input=layer_input, filter_shape=(50, 20, 5, 5), image_shape=(self.batch_size, 20, 12, 12), poolsize=self.poolsize,fdrop=False)
               assert hasattr(this_layer, 'output')
               self.params.extend(this_layer.params)
               self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                   'accugrad') for t in this_layer.params])
               self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                   'accudelta') for t in this_layer.params])
               self.layers.append(this_layer)
               layer_input = this_layer.output.flatten(2) # NECESSARY TO IMPORT TO OTHER FORMAT
               print this_layer,layer_input
           else: 
               this_layer = layer_type(rng=numpy_rng,
                    input=layer_input, n_in=n_in, n_out=n_out)
               assert hasattr(this_layer, 'output')
               self.params.extend(this_layer.params)
               self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                   'accugrad') for t in this_layer.params])
               self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                   'accudelta') for t in this_layer.params])
               self.layers.append(this_layer)
               layer_input = this_layer.output
               print this_layer,layer_input
 
 
 
 
 
        print zip(layers_types,self.layers_ins, self.layers_outs)
        print self.layers[-1]
        #print self.layers[-1].cross_entropy(self.y)
        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        # TODO standardize cost
        
        #self.mean_cost = self.layers[-1].cross_entropy(self.y) #TODO CHANGE THE COST TO CROSS ENTROPY
        #self.cost = self.layers[-1].cross_entropy_training_cost(self.y)
        self.mean_cost = self.layers[-1].negative_log_likelihood(self.y)
        self.cost = self.layers[-1].training_cost(self.y)
        if debugprint:
            theano.printing.debugprint(self.cost)
 
        self.errors = self.layers[-1].errors(self.y)
 
    def __repr__(self):
        dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
                                    zip(self.layers_ins, self.layers_outs))
        return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
                            zip(self.layers_types, dimensions_layers_str)))
 
 
    def get_SGD_trainer(self):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x = T.dmatrix('batch_x')#fmatrix
        batch_y = T.lvector('batch_y')#ivector
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        # using mean_cost so that the learning rate is not too dependent
        # on the batch size
        print self.mean_cost,self.params
        print T.grad(self.mean_cost, self.params[4:])
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



class RegularizedNet(NeuralNet):
    """ Neural net with L1 and L2 regularization """
    def __init__(self, numpy_rng, theano_rng=None,
                 n_ins=100,
                 layers_types=[ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=[1024, 1024, 1024],
                 n_outs=2,
                 rho=0.9,
                 eps=1.E-6,
                 L1_reg=0.,
                 L2_reg=0.,
                 max_norm=0.,
                 debugprint=False):
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


class RegularizedConvNet(AlexNet):
    """ Convolutional Neural net with L1 and L2 regularization """
    def __init__(self, numpy_rng, theano_rng=None,
                 n_ins=100,
                 layers_types=[ConvolutionalLayer1,ConvolutionalLayer2,ReLU, ReLU, ReLU, LogisticRegression_crossentropy],
                 layers_sizes=[1024, 1024, 1024, 1024, 1024],
                 n_outs=2,
                 rho=0.9,
                 eps=1.E-6,
                 L1_reg=0.,
                 L2_reg=0.,
                 max_norm=0.,
                 debugprint=False):
        """
        Feedforward neural network with added L1 and/or L2 regularization.
        """
        super(RegularizedConvNet, self).__init__(numpy_rng, theano_rng, n_ins,
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
 
 
class DropoutNet(NeuralNet):
    """ Neural net with dropout (see Hinton's et al. paper) """
    def __init__(self, numpy_rng, theano_rng=None,
                 n_ins=40*3,
                 layers_types=[ReLU, ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=[4000, 4000, 4000, 4000],
                 dropout_rates=[0.0, 0.5, 0.5, 0.5, 0.5],
                 n_outs=62 * 3,
                 rho=0.9,
                 eps=1.E-6,
                 max_norm=0.,
                 fast_drop=False,
                 debugprint=False):
        """
        Feedforward neural network with dropout regularization.
        """
        super(DropoutNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, rho, eps, max_norm,
                debugprint)
 
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
                dropout_rates[1:] + [0]):  # !!! we do not dropout anything
                                           # from the last layer !!!
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
                    # N.B. dropout with dr==1 does not dropanything!!
                    this_layer.output = dropout(numpy_rng, this_layer.output, dr)
            else:
                this_layer = layer_type(rng=numpy_rng,
                        input=dropout_layer_input, n_in=n_in, n_out=n_out,
                        W=layer.W, b=layer.b)
 
            assert hasattr(this_layer, 'output')
            self.dropout_layers.append(this_layer)
            dropout_layer_input = this_layer.output
 
        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        # these are the dropout costs
        self.mean_cost = self.dropout_layers[-1].negative_log_likelihood(self.y)
        self.cost = self.dropout_layers[-1].training_cost(self.y)
 
        # these is the non-dropout errors
        self.errors = self.layers[-1].errors(self.y)
 
    def __repr__(self):
        return super(DropoutNet, self).__repr__() + "\n"\
                + "dropout rates: " + str(self.dropout_rates)
 

class DropoutAlexNet(AlexNet):
    """ Convolutional Neural net with dropout (see Hinton's et al. paper) """
    def __init__(self, numpy_rng, theano_rng=None,
                 n_ins=40*3,#3
                 layers_types=[ConvolutionalLayer1, ConvolutionalLayer2, ReLU, ReLU, ReLU, LogisticRegression],#LogisticRegression_crossentropy
                 layers_sizes=[4000, 4000, 4000, 4000, 4000],
                 dropout_rates=[0.0, 0.5, 0.5, 0.5],
                 n_outs=62 * 3,#3
                 rho=0.9,
                 eps=1.E-6,
                 max_norm=0.,
                 fast_drop=False,
                 debugprint=False):
        """
        Feedforward convolutional neural network with dropout regularization.
        """
        super(DropoutAlexNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, rho, eps, max_norm,
                debugprint)
 
        self.dropout_rates = dropout_rates
        if fast_drop:
            if dropout_rates[0]:
                dropout_layer_input = fast_dropout(numpy_rng, self.x)
            else:
                dropout_layer_input = self.x
        else:
            dropout_layer_input = dropout(numpy_rng, self.x, p=dropout_rates[0])
        self.dropout_layers = []
        conv_layer_input=dropout_layer_input.reshape((self.batch_size,1,28,28)) #change later param
        for layer, layer_type, n_in, n_out, dr in zip(self.layers,
                layers_types, self.layers_ins, self.layers_outs,
                [0]+[0]+dropout_rates[1:] + [0]):  # !!! we do not dropout anything
                                           # from the last layer OR THE CONV LAYER !!!
            if dr:
                if fast_drop:
                    print this_layer
                    this_layer = layer_type(rng=numpy_rng,
                            input=dropout_layer_input, n_in=n_in, n_out=n_out,
                            W=layer.W, b=layer.b, fdrop=True)
                    assert hasattr(this_layer, 'output')
                    self.dropout_layers.append(this_layer)
                    dropout_layer_input = this_layer.output
                else:
                    this_layer = layer_type(rng=numpy_rng,
                            input=dropout_layer_input, n_in=n_in, n_out=n_out,
                            W=layer.W * 1. / (1. - dr),
                            b=layer.b * 1. / (1. - dr))
                    # N.B. dropout with dr==1 does not dropanything!!
                    this_layer.output = dropout(numpy_rng, this_layer.output, dr)
                    assert hasattr(this_layer, 'output')
                    self.dropout_layers.append(this_layer)
                    dropout_layer_input = this_layer.output
            elif layer_type==ConvolutionalLayer1: #if convlayer1,convlayer2,etc. then change params with forloop,
               #last layer must have output.flatten(2) as the summation of the layer to be used with the ReLU layers
               this_layer = layer_type(rng=numpy_rng,
                     input=conv_layer_input, filter_shape=(20, 1, 5, 5), image_shape=(self.batch_size, 1, 28, 28), poolsize=(2,2),fdrop=False)
               assert hasattr(this_layer, 'output')
               self.dropout_layers.append(this_layer)
               layer_input = this_layer.output
            elif layer_type==ConvolutionalLayer2: #if convlayer1,convlayer2,etc. then change params with forloop
               this_layer = layer_type(rng=numpy_rng,
                    input=layer_input, filter_shape=(50, 20, 5, 5), image_shape=(self.batch_size, 20, 12, 12), poolsize=(2,2),fdrop=False)
               assert hasattr(this_layer, 'output')
               self.dropout_layers.append(this_layer)
               dropout_layer_input = this_layer.output.flatten(2) #necessary flatten layer
            
            
            else:
               this_layer = layer_type(rng=numpy_rng,
                        input=dropout_layer_input, n_in=n_in, n_out=n_out,
                        W=layer.W, b=layer.b)
               assert hasattr(this_layer, 'output')
               self.dropout_layers.append(this_layer)
               dropout_layer_input = this_layer.output
 
        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        # these are the dropout costs
        #self.mean_cost = self.dropout_layers[-1].cross_entropy(self.y)
        #self.cost = self.dropout_layers[-1].cross_entropy_training_cost(self.y)
        self.mean_cost = self.dropout_layers[-1].negative_log_likelihood(self.y)
        self.cost = self.dropout_layers[-1].training_cost(self.y)
        # these is the non-dropout errors
        self.errors = self.layers[-1].errors(self.y)
 
    def __repr__(self):
        return super(DropoutAlexNet, self).__repr__() + "\n"\
                + "dropout rates: " + str(self.dropout_rates)



def add_fit_and_score(class_to_chg):
    """ Mutates a class to add the fit() and score() functions to a NeuralNet.
    """
    from types import MethodType
    def fit(self, x_train, y_train, x_dev=None, y_dev=None,
            max_epochs=100, early_stopping=True, split_ratio=0.1,
            method='adadelta', verbose=False, plot=False):
        """
        Fits the neural network to `x_train` and `y_train`. 
        If x_dev nor y_dev are not given, it will do a `split_ratio` cross-
        validation split on `x_train` and `y_train` (for early stopping).
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
        train_set_iterator = DatasetMiniBatchIterator(x_train, y_train)
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
 
        while epoch < max_epochs:
            if not verbose:
                sys.stdout.write("\r%0.2f%%" % (epoch * 100./ max_epochs))
                sys.stdout.flush()
            avg_costs = []
            timer = time.time()
            for x, y in train_set_iterator:
                if method == 'sgd' or method == 'adagrad':
                    avg_cost = train_fn(x, y, lr=1)  # TODO: you have to
                                                         # play with this
                                                         # learning rate
                                                         # (dataset dependent)
                elif method == 'adadelta':
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
                print('  epoch %i, training error %f' %
                      (epoch, mean_train_errors))
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
 
 
if __name__ == "__main__":
    add_fit_and_score(DropoutAlexNet)
    #add_fit_and_score(RegularizedConvNet)
 
    def nudge_dataset(X, Y):
        """
        This produces a dataset 5 times bigger than the original one,
        by moving the 8x8 images in X around by 1px to left, right, down, up
        """
        from scipy.ndimage import convolve
        direction_vectors = [
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]]]
        shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                      weights=w).ravel()
        X = numpy.concatenate([X] +
                              [numpy.apply_along_axis(shift, 1, X, vector)
                                  for vector in direction_vectors])
        Y = numpy.concatenate([Y for _ in range(5)], axis=0)
        return X, Y
 
    from sklearn import datasets, svm, naive_bayes
    from sklearn import cross_validation, preprocessing
    MNIST = True  # MNIST dataset
    DIGITS = False  # digits dataset
    FACES = False  # faces dataset
    TWENTYNEWSGROUPS = False  # 20 newgroups dataset
    VERBOSE = True  # prints evolution of the loss/accuracy during the fitting
    SCALE = True  # scale the dataset
    PLOT = True  # plot losses and accuracies
 
    def train_models(x_train, y_train, x_test, y_test, n_features, n_outs,
            use_dropout=True, n_epochs=100, numpy_rng=None,
            svms=False, nb=False, deepnn=True, name=''):
        if svms:
            print("Linear SVM")
            classifier = svm.SVC(gamma=0.001)
            print(classifier)
            classifier.fit(x_train, y_train)
            print("score: %f" % classifier.score(x_test, y_test))
 
            print("RBF-kernel SVM")
            classifier = svm.SVC(kernel='rbf', class_weight='auto')
            print(classifier)
            classifier.fit(x_train, y_train)
            print("score: %f" % classifier.score(x_test, y_test))
 
        if nb:
            print("Multinomial Naive Bayes")
            classifier = naive_bayes.MultinomialNB()
            print(classifier)
            classifier.fit(x_train, y_train)
            print("score: %f" % classifier.score(x_test, y_test))
 
        if deepnn:
            import warnings
            warnings.filterwarnings("ignore")  # TODO remove
 
            if use_dropout:
                #n_epochs *= 4  TODO
                pass
 
            #def new_dnn(dropout=False):
                #if dropout:
                    #print("Dropout DNN")
                    #return DropoutNet(numpy_rng=numpy_rng, n_ins=n_features,
                        #layers_types=[ReLU, ReLU, LogisticRegression],
                        #layers_sizes=[200, 200],
                        #dropout_rates=[0., 0.5, 0.5],
                        ## TODO if you have a big enough GPU, use these:
                        ##layers_types=[ReLU, ReLU, ReLU, ReLU, LogisticRegression],
                        ##layers_sizes=[2000, 2000, 2000, 2000],
                        ##dropout_rates=[0., 0.5, 0.5, 0.5, 0.5],
                        #n_outs=n_outs,
                        #max_norm=4.,
                        #fast_drop=True,
                        #debugprint=0)
                #else:
                    #print("Simple (regularized) DNN")
                    #return RegularizedNet(numpy_rng=numpy_rng, n_ins=n_features,
                        #layers_types=[ReLU, ReLU, LogisticRegression],
                        #layers_sizes=[200, 200],
                        #n_outs=n_outs,
                        ##L1_reg=0.001/x_train.shape[0],
                        ##L2_reg=0.001/x_train.shape[0],
                        #L1_reg=0.,
                        #L2_reg=1./x_train.shape[0],
                        #debugprint=0)
 
 
 
 
            def new_dnn(dropout=False):
                if dropout:
                    print("AlexNet Dropout DNN")
                    return DropoutAlexNet(numpy_rng=numpy_rng, n_ins=n_features,
                        layers_types=[ConvolutionalLayer, ConvolutionalLayer, ReLU, ReLU, ReLU, LogisticRegression],#LogisticRegression_crossentropy,
                        layers_sizes=[200, 200, 200, 200, 200],
                        dropout_rates=[0.0, 0.5, 0.5, 0.5],
                        # TODO if you have a big enough GPU, use these:
                        #layers_types=[ReLU, ReLU, ReLU, ReLU, LogisticRegression],
                        #layers_sizes=[2000, 2000, 2000, 2000],
                        #dropout_rates=[0., 0.5, 0.5, 0.5, 0.5],
                        n_outs=n_outs,
                        max_norm=4.,
                        fast_drop=True,
                        debugprint=0)
                else:
                    print("Simple (regularized) DNN")
                    return RegularizedNet(numpy_rng=numpy_rng, n_ins=n_features,
                        layers_types=[ReLU, ReLU, LogisticRegression],
                        layers_sizes=[200, 200],
                        n_outs=n_outs,
                        #L1_reg=0.001/x_train.shape[0],
                        #L2_reg=0.001/x_train.shape[0],
                        L1_reg=0.,
                        L2_reg=1./x_train.shape[0],
                        debugprint=0)
 
            import matplotlib.pyplot as plt
            plt.figure()
            ax1 = plt.subplot(221)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(223)
            ax4 = plt.subplot(224)  # TODO plot the updates of the weights
            methods = ['sgd', 'adagrad', 'adadelta']
            #methods = ['adadelta'] TODO if you want "good" results asap
            for method in methods:
                dnn = new_dnn(use_dropout)
                print dnn, "using", method
                dnn.fit(x_train, y_train, max_epochs=n_epochs, method=method, verbose=VERBOSE, plot=PLOT)
                test_error = dnn.score(x_test, y_test)
                print("score: %f" % (1. - test_error))
                ax1.plot(numpy.log10(dnn._costs), label=method)
                ax2.plot(numpy.log10(dnn._train_errors), label=method)
                ax3.plot(numpy.log10(dnn._dev_errors), label=method)
                #ax2.plot(dnn._train_errors, label=method)
                #ax3.plot(dnn._dev_errors, label=method)
                ax4.plot([test_error for _ in range(10)], label=method)
            ax1.set_xlabel('epoch')
            ax1.set_ylabel('cost (log10)')
            ax2.set_xlabel('epoch')
            ax2.set_ylabel('train error')
            ax3.set_xlabel('epoch')
            ax3.set_ylabel('dev error')
            ax4.set_ylabel('test error')
            plt.legend()
            plt.savefig('training_' + name + '.png')
 
 
    if MNIST:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
        #X = numpy.asarray(mnist.data, dtype='float32')
        X = numpy.asarray(mnist.data, dtype='float64')
        if SCALE:
            #X = preprocessing.scale(X)
            X /= 255.
        #y = numpy.asarray(mnist.target, dtype='int32')
        y = numpy.asarray(mnist.target, dtype='int64')
        print("Total dataset size:")
        print("n samples: %d" % X.shape[0])
        print("n features: %d" % X.shape[1])
        print("n classes: %d" % len(set(y)))
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                    X, y, test_size=0.2, random_state=42)
 
        train_models(x_train, y_train, x_test, y_test, X.shape[1],
                     len(set(y)), numpy_rng=numpy.random.RandomState(123),
                     name='MNIST')
 
 
    if DIGITS:
        digits = datasets.load_digits()
        data = numpy.asarray(digits.data, dtype='float32')
        target = numpy.asarray(digits.target, dtype='int32')
        nudged_x, nudged_y = nudge_dataset(data, target)
        if SCALE:
            nudged_x = preprocessing.scale(nudged_x)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                nudged_x, nudged_y, test_size=0.2, random_state=42)
        train_models(x_train, y_train, x_test, y_test, nudged_x.shape[1],
                     len(set(target)), numpy_rng=numpy.random.RandomState(123),
                     name='digits')
 
    if FACES:
        import logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s')
        lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70,
                                               resize=0.4)
        X = numpy.asarray(lfw_people.data, dtype='float32')
        if SCALE:
            X = preprocessing.scale(X)
        y = numpy.asarray(lfw_people.target, dtype='int32')
        target_names = lfw_people.target_names
        print("Total dataset size:")
        print("n samples: %d" % X.shape[0])
        print("n features: %d" % X.shape[1])
        print("n classes: %d" % target_names.shape[0])
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                    X, y, test_size=0.2, random_state=42)
 
        train_models(x_train, y_train, x_test, y_test, X.shape[1],
                     len(set(y)), numpy_rng=numpy.random.RandomState(123),
                     name='faces')
 
    if TWENTYNEWSGROUPS:
        from sklearn.feature_extraction.text import TfidfVectorizer
        newsgroups_train = datasets.fetch_20newsgroups(subset='train')
        vectorizer = TfidfVectorizer(encoding='latin-1', max_features=10000)
        #vectorizer = HashingVectorizer(encoding='latin-1')
        x_train = vectorizer.fit_transform(newsgroups_train.data)
        x_train = numpy.asarray(x_train.todense(), dtype='float32')
        y_train = numpy.asarray(newsgroups_train.target, dtype='int32')
        newsgroups_test = datasets.fetch_20newsgroups(subset='test')
        x_test = vectorizer.transform(newsgroups_test.data)
        x_test = numpy.asarray(x_test.todense(), dtype='float32')
        y_test = numpy.asarray(newsgroups_test.target, dtype='int32')
        train_models(x_train, y_train, x_test, y_test, x_train.shape[1],
                     len(set(y_train)),
                     numpy_rng=numpy.random.RandomState(123),
                     svms=False, nb=True, deepnn=True,
                     name='20newsgroups')
