olve = scipy.signal.convolve


class Convolutional(RBM):
    '''
    '''

    def __init__(self, num_filters, filter_shape, pool_shape, binary=True, scale=0.001):
        '''Initialize a convolutional restricted boltzmann machine.

        num_filters: The number of convolution filters.
        filter_shape: An ordered pair describing the shape of the filters.
        pool_shape: An ordered pair describing the shape of the pooling groups.
        binary: True if the visible units are binary, False if the visible units
          are normally distributed.
        scale: Scale initial values by this parameter.
        '''
        self.num_filters = num_filters
        self.weights = scale * rng.randn(num_filters, *filter_shape)
        self.vis_bias = scale * rng.randn()
        self.hid_bias = 2 * scale * rng.randn(num_filters)

        self._visible = binary and sigmoid or identity
        self._pool_shape = pool_shape

    def _pool(self, hidden):
        '''Given activity in the hidden units, pool it into groups.'''
        _, r, c = hidden.shape
        rows, cols = self._pool_shape
        active = numpy.exp(hidden.T)
        pool = numpy.zeros(active.shape, float)
        for j in range(int(numpy.ceil(float(c) / cols))):
            cslice = slice(j * cols, (j + 1) * cols)
            for i in range(int(numpy.ceil(float(r) / rows))):
                mask = (cslice, slice(i * rows, (i + 1) * rows))
                pool[mask] = active[mask].sum(axis=0).sum(axis=0)
        return pool.T

    def pooled_expectation(self, visible, bias=0.):
        '''Given visible data, return the expected pooling unit values.'''
        activation = numpy.exp(numpy.array([
            convolve(visible, self.weights[k, ::-1, ::-1], 'valid')
            for k in range(self.num_filters)]).T + self.hid_bias + bias).T
        return 1. - 1. / (1. + self._pool(activation))

    def hidden_expectation(self, visible, bias=0.):
        '''Given visible data, return the expected hidden unit values.'''
        activation = numpy.exp(numpy.array([
            convolve(visible, self.weights[k, ::-1, ::-1], 'valid')
            for k in range(self.num_filters)]).T + self.hid_bias + bias).T
        return activation / (1. + self._pool(activation))

    def visible_expectation(self, hidden, bias=0.):
        '''Given hidden states, return the expected visible unit values.'''
        activation = sum(
            convolve(hidden[k], self.weights[k], 'full')
            for k in range(self.num_filters)) + self.vis_bias + bias
        return self._visible(activation)

    def calculate_gradients(self, visible):
        '''Calculate gradients for an instance of visible data.

        Returns a 3-tuple of gradients: weights, visible bias, hidden bias.

        visible: A single array of visible data.
        '''
        v0 = visible
        h0 = self.hidden_expectation(v0)
        v1 = self.visible_expectation(bernoulli(h0))
        h1 = self.hidden_expectation(v1)

        # h0.shape == h1.shape == (num_filters, visible_rows - filter_rows + 1, visible_columns - filter_columns + 1)
        # v0.shape == v1.shape == (visible_rows, visible_columns)

        gw = numpy.array([convolve(v0, h0[k, ::-1, ::-1], 'valid') - convolve(v1, h1[k, ::-1, ::-1], 'valid')
                               for k in range(self.num_filters)])
        gv = (v0 - v1).sum()
        gh = (h0 - h1).sum(axis=-1).sum(axis=-1)

        logging.debug('error: %.3g, hidden acts: %.3g',
                      numpy.linalg.norm(gv), h0.mean(axis=-1).mean(axis=-1).std())

        return gw, gv, gh

    class Trainer:
        '''
        '''

        def __init__(self, rbm, momentum, target_sparsity=None):
            '''
            '''
            self.rbm = rbm
            self.momentum = momentum
            self.target_sparsity = target_sparsity

            self.grad_weights = numpy.zeros(rbm.weights.shape, float)
            self.grad_vis = 0.
            self.grad_hid = numpy.zeros(rbm.hid_bias.shape, float)

        def learn(self, visible, l2_reg=0, alpha=0.2):
            '''
            '''
            w, v, h = self.rbm.calculate_gradients(visible)
            if self.target_sparsity is not None:
                h = self.target_sparsity - self.rbm.hidden_expectation(visible).mean(axis=-1).mean(axis=-1)

            kwargs = dict(alpha=alpha, momentum=self.momentum)
            self.grad_vis = self.rbm.apply_gradient('vis_bias', v, self.grad_vis, **kwargs)
            self.grad_hid = self.rbm.apply_gradient('hid_bias', h, self.grad_hid, **kwargs)

            kwargs['l2_reg'] = l2_reg
            self.grad_weights = self.rbm.apply_gradient('weights', w, self.grad_weights, **kwargs)
