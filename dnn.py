"""
A deep neural network with or w/o dropout in one file.
"""
 
import numpy, theano, sys, math
from theano import tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
 
BATCH_SIZE = 100
 
 
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
 
 
class Convolutional_Pool_Layer(object):
     def __init__(self, rng, input, n_in, n_out, W=None, b=None, fdrop=False, patch_dim, pool_dim):):
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
        self.patch_dim = patch_dim
        self.pool_dim  = pool_dim
    #######################################################################################
    """ Returns elementwise sigmoid output of input array """
    def sigmoid(self, x):
        return (1 / (1 + numpy.exp(-x)))
    def ReLU(self,x):
        return max(0,x)
    #######################################################################################
    """ Returns the convolved features of the input images """
    def convolve(self, input_images, num_features):
        """ Extract useful values """
        image_dim      = input_images.shape[0]
        image_channels = input_images.shape[2]
        num_images     = input_images.shape[3]
        """ Assign memory for the convolved features """
        conv_dim           = image_dim - self.patch_dim + 1
        convolved_features = numpy.zeros((num_features, num_images, conv_dim, conv_dim));
        for image_num in range(num_images):
            for feature_num in range(num_features):
                """ Initialize convolved image as array of zeros """
                convolved_image = numpy.zeros((conv_dim, conv_dim))
                for channel in range(image_channels):
                    """ Extract feature corresponding to the indices """
                    limit0  = self.patch_dim * self.patch_dim * channel
                    limit1  = limit0 + self.patch_dim * self.patch_dim
                    feature = self.W[feature_num, limit0 : limit1].reshape(self.patch_dim, self.patch_dim)
                    """ Image to be convolved """
                    image = input_images[:, :, channel, image_num]
                    """ Convolve image with the feature and add to existing matrix """
                    convolved_image = convolved_image + scipy.signal.convolve2d(image, feature, 'valid');
                """ Take sigmoid transform and store """
                convolved_image = self.ReLU(convolved_image + self.b[feature_num, 0])
                convolved_features[feature_num, image_num, :, :] = convolved_image
        return convolved_features
    #######################################################################################
    """ Pools the given convolved features """
    def pool(self, convolved_features):
        """ Extract useful values """
        num_features = convolved_features.shape[0]
        num_images   = convolved_features.shape[1]
        conv_dim     = convolved_features.shape[2]
        res_dim      = conv_dim / self.pool_dim
        """ Initialize pooled features as array of zeros """
        pooled_features = numpy.zeros((num_features, num_images, res_dim, res_dim))
        for image_num in range(num_images):
            for feature_num in range(num_features):
                for pool_row in range(res_dim):
                    row_start = pool_row * self.pool_dim
                    row_end   = row_start + self.pool_dim
                    for pool_col in range(res_dim):
                        col_start = pool_col * self.pool_dim
                        col_end   = col_start + self.pool_dim
                        """ Extract image patch and calculate mean pool """
                        patch = convolved_features[feature_num, image_num, row_start : row_end,
                                                   col_start : col_end]
                        pooled_features[feature_num, image_num, pool_row, pool_col] = numpy.mean(patch)
        return pooled_features 
        
        
 def getPooledFeatures(network, images, num_features, res_dim, step_size):
    num_images = images.shape[3]
    """ Initialize pooled features as array of zeros """
    pooled_features_data = numpy.zeros((num_features, num_images, res_dim, res_dim))
    for step in range(num_images / step_size):
        """ Limits to access batch of images """
        limit0 = step_size * step
        limit1 = step_size * (step+1)
        image_batch = images[:, :, :, limit0 : limit1]
        """ Calculate pooled features for the image batch """
        convolved_features = network.convolve(image_batch, num_features)
        pooled_features    = network.pool(convolved_features)
        pooled_features_data[:, limit0 : limit1, :, :] = pooled_features
        """ Avoid memory overflow """
        del(image_batch)
        del(convolved_features)
        del(pooled_features)
    """ Reshape data for training / testing """
    input_size = pooled_features_data.size / num_images
    pooled_features_data = numpy.transpose(pooled_features_data, (0, 2, 3, 1))
    pooled_features_data = pooled_features_data.reshape(input_size, num_images)
    return pooled_features_data

 
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
                    avg_cost = train_fn(x, y, lr=0.02)  # TODO: you have to
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
    add_fit_and_score(DropoutNet)
    add_fit_and_score(RegularizedNet)
 
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
    FACES = True  # faces dataset
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
 
            def new_dnn(dropout=False):
                if dropout:
                    print("Dropout DNN")
                    return DropoutNet(numpy_rng=numpy_rng, n_ins=n_features,
                        layers_types=[ReLU, ReLU, LogisticRegression],
                        layers_sizes=[200, 200],
                        dropout_rates=[0., 0.5, 0.5],
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
        X = numpy.asarray(mnist.data, dtype='float32')
        if SCALE:
            #X = preprocessing.scale(X)
            X /= 255.
        y = numpy.asarray(mnist.target, dtype='int32')
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
