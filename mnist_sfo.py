import logging

import numpy as np
from pylearn2.models.mlp import MLP, Linear, Sigmoid
from pylearn2.datasets.mnist import MNIST
import scipy as sp
import theano
from theano import tensor

from sfo import SFO


def init_params(model):
    """
    Here we collect all the network's parameters into one large
    parameter, so that we can easily calculate the Hessian and such.

    This only works as long as all layers are children of Linear

    Returns
    -------
    num_params : int
        The total number of parameters in the model
    params : theano shared variable
        A Theano shared variable containing all the parameters
    """
    # Collect the parameter names, shapes and sizes
    param_values, param_sizes, param_names, param_shapes = [], [], [], []
    for param in model.get_params():
        param_values.append(param.get_value())
        param_sizes.append(param_values[-1].size)
        param_shapes.append(param_values[-1].shape)
        param_names.append(param.name)
    num_params = np.sum(param_sizes)
    params_values = np.concatenate([param.flatten() for param in param_values])

    # Combine everything into a large 1D array
    params = tensor.vector('params')

    # Replace all of pylearn2 parameters with slices of this variable
    new_params = {name: params[i:j].reshape(shape)
                  for name, shape, i, j in
                  zip(param_names,
                      param_shapes,
                      np.cumsum([0] + param_sizes[:-1]),
                      np.cumsum(param_sizes))}
    for layer in model.layers:
        assert isinstance(layer, Linear)
        layer.transformer._W = new_params[layer.layer_name + '_W']
        layer.b = new_params[layer.layer_name + '_b']

    return num_params, params, params_values


def create_autoencoder(layer_sizes, input_size=784, irange=None,
                       sparse_stdev=1., sparse_init=None):
    """
    Creates an autoencoder consisting of only sigmoids
    with the given layer sizes

    Returns
    -------
    mlp : MLP
        A Pylearn2 MLP instance
    """
    layers = []
    for i, layer_size in enumerate(layer_sizes + layer_sizes[-2::-1]):
        layers.append(Sigmoid(layer_name='h' + str(i),
                              dim=layer_size, irange=irange,
                              sparse_stdev=sparse_stdev,
                              sparse_init=sparse_init))
    layers.append(Sigmoid(layer_name='y', dim=input_size,
                          irange=irange, sparse_stdev=sparse_stdev,
                          sparse_init=sparse_init))
    return MLP(layers=layers, nvis=input_size)


def load_mnist(mnist_size):
    """
    Loads MNIST and reshapes it if needed

    Parameters
    ----------
    mnist_size : int
        The desired height/width in pixels

    Returns
    -------
    Two Dataset instances for the training and test set
    """
    mnist_train = MNIST(which_set='train')
    mnist_test = MNIST(which_set='test')
    if mnist_size != 28:
        for dataset in [mnist_train, mnist_test]:
            dataset.X = np.array([
                sp.misc.imresize(X.reshape(28, 28),
                                 (mnist_size, mnist_size)).flatten()
                for X in dataset.X
            ])
    return mnist_train, mnist_test


if __name__ == "__main__":
    # Show info messages
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S')

    # Do not allow NaN to be produced
    np.seterr(invalid='raise')

    # Parameters
    mnist_size = 28
    layer_sizes = [1000, 500, 250, 30]
    num_epochs = 10000
    num_subfuncs = 100  # Must be a divisor of 60000; ideally ~240

    # Create autoencoder
    logging.info("Constructing model")
    mlp = create_autoencoder(layer_sizes, input_size=mnist_size ** 2,
                             sparse_init=15)
    num_params, params, params_values = init_params(mlp)

    # Load MNIST
    logging.info("Loading MNIST")
    mnist_train, mnist_test = load_mnist(mnist_size)
    sub_refs = np.split(mnist_train.X, num_subfuncs)

    # Cost and gradient
    logging.info("Compiling Theano functions")

    X = tensor.matrix()
    cost = mlp.cost(X, mlp.fprop(X))
    cost_func = theano.function([params, X], tensor.as_tensor(cost))
    f_df = theano.function([params, X],
                           [cost, tensor.as_tensor(tensor.grad(cost, params))],
                           allow_input_downcast=True)

    # Initialize the optimizer
    optimizer = SFO(f_df, params_values, sub_refs)

    # Start process
    params = optimizer.optimize(num_passes=num_epochs)
