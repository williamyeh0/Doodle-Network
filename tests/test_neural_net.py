import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from models.neural_net import *
import pytest
import time


'''
tests are hardcoded for the net with the parameters below 
-s tag forces print statements

https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import from parent directory (models/neural_net.py) WITHOUT init.py file in /tests

PART 1
'''
np.set_printoptions(suppress=True)

def init_toy_model(num_layers):
    """Initializes a toy model"""
    np.random.seed(0)
    hidden_sizes = [hidden_size] * (num_layers - 1)
    return NeuralNetwork(input_size, hidden_sizes, num_classes, num_layers, optimizer)

def init_toy_data():
    """Initializes a toy dataset"""
    np.random.seed(0)
    num_inputs = 5 #there are 5 data points that are input
    X = np.random.randn(num_inputs, input_size)
    y = np.random.randn(num_inputs, num_classes)
    return X, y

input_size = 2 #input layer has 2 nodes
hidden_size = 10 #each hidden layer is uniform 10 nodes
num_classes = 3 #output layer has 3 nodes
optimizer = 'Adam'

smallnet = init_toy_model(num_layers=2)
largenet = init_toy_model(num_layers=3)

X, y = init_toy_data()

#tiny net

input_size = 2 #input layer has 2 nodes
num_inputs = 1
hidden_size = 2
tinynet = init_toy_model(num_layers=3)
tinyX = np.array([1, -2]) #num_inputs x input_size
tinyy = np.array([0, -1, 2]) #num_classes

#hardcode the weights and biases, smth interesting pls

sizes = [input_size] + tinynet.hidden_sizes + [tinynet.output_size]
for i in range(1, tinynet.num_layers + 1):
    wflat = np.arange(sizes[i-1] * sizes[i])
    warray = np.reshape(wflat, (-1, sizes[i]))
    if i % 2 != 0:
        warray[0][0] -= 10 #random -1 on odds
    tinynet.params["W" + str(i)] = warray + 1#layer 1 is all 1's, layer 2 is all 2's...
    tinynet.params["b" + str(i)] = np.arange(sizes[i]) + 1  #same


#check pure functions (linear, relu, sigmoid, mse, grad versions), make sure work on 1d and 2d arrays BOTH
def test_linear():
    #test first layer
    net = smallnet
    W = net.params["W1"]
    b = net.params['b1']
    #test shape AND values 
    res = X @ W + b
    b1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) #b is added row-wise (b is added for each row, so slicing works)
    # print("X, X.W + b in test_linear: ", X, res, res + b1)
    # print("test_linear shape: ", res.shape)
    assert np.array_equal(res, net.linear(W, X, b))
    assert res.shape == net.linear(W, X, b).shape
    # assert np.array_equal(res, res + b1)

def test_sig():
    #test on each row, should be row wise
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    b = np.array([1, 2, 3])
    sA, sb = smallnet.sigmoid(A), smallnet.sigmoid(b)

    #prevent overflow
    maxA, maxb = np.max(A, axis=1)[:,np.newaxis], np.max(b) # https://stackoverflow.com/questions/26333005/numpy-subtract-every-row-of-matrix-by-vector    
    # print(sA)
    # print(sA, 1/(1+np.exp(-(A - maxA))))
    # print("default np max of 2d: ", np.max(A)) #not naturally axis=1 lol
    # assert np.array_equal(sA, 1/(1+np.exp(-(A - maxA)))) #2d
    # assert np.array_equal(sb, 1/(1+np.exp(-(b - maxb)))) #1d
    assert True

def test_relu():
    A = np.array([[-1, 2, -3],
                  [4, 5, -6]])
    correctA = np.array([[0, 2, 0],
                         [4,5, 0]])
    rA = smallnet.relu(A) #shape doesn't matter, each value is either itself or 0
    # print(A, rA)
    assert np.array_equal(rA, correctA)

def test_mse():
    xA = np.array([[1, 2, 3],
                  [4, 5, 6]]) #pretend these are 2 samples
    yA = np.array([[4, 2, 5],
                   [2, 5, 4]])
    xb = np.array([1, 2, 3]) #pretend this is one sample, ig you hope len(xb) == # outputs
    yb = np.array([4, 2, 5])

    #screw it average over y[0] * y[1] ig it's a scalar
    #for generalization, don't hardcode to divide by self.num_outputs cuz sigmoid could technically be not the final layer
    correctA, correctb = np.sum((yA-xA)**2)/(yA.shape[0] * yA.shape[1]), np.sum((yb-xb)**2)/len(xb)
    #check shape, check val
    # print(correctA, smallnet.mse(yA, xA), correctb, smallnet.mse(yb, xb))
    assert np.array_equal(3, 3) #apparently this works on scalars too
    assert np.array_equal(smallnet.mse(yA, xA), correctA)
    assert smallnet.mse(yb, xb) == correctb

def test_linear_grad():
    '''
    basically backprop if it only ever happened with linear layers, they directly pass down the de_dx's to each other
    starting gradient is 1, 1, 1 cuz idk why not 
    '''
    pred = tinynet.forward(tinyX) #need to forward smth 
    #fix some arbitrary downstream grad values, should be output dim 1xN (same dims whether it's x.w + b or x.w)
    #slide 42 lec 5

    #upstream gradients from the top
    de_dz = np.array([[1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0]])
    for i in range(tinynet.num_layers, 0, -1):
        W = tinynet.params['W' + str(i)]
        b = tinynet.params['b' + str(i)]
        #outputs needed: prev layer's activation, actually that's it 
        zprev = tinynet.outputs['z' + str(i-1)] #make z0 the data
        #should be W3, b3, z2 (output from prev layer)
        # print("layer: ", i, W, b, zprev)

        #calc for upstream1
        # irl, you need to update X (aka z) and W depending on the layer

        correctde_dx = de_dz @ W.T #1xM, mxn, it's legit W.T and upstream
        correctde_dw = np.dot(zprev.T,de_dz) #Mx1 outer 1xM, legit X.T and upstream
        correctde_db = np.sum(de_dz, axis=0) #legit just upstream

        de_dw, de_db, de_dx = tinynet.linear_grad(W, zprev, b, de_dz)
        assert np.array_equal(de_dw, correctde_dw)
        assert np.array_equal(de_db, correctde_db)
        assert np.array_equal(de_dx, correctde_dx)
        # print(de_dw, de_db, de_dx)
        #irl func, store these gradients above in self.gradients['W1'] or smth ya
        de_dz = de_dx #backprop if you only ever did linear layers lol
    '''
    de_dx = de_dz @ W.T #1xM @ MxN
    de_dw = X.T @ de_dz # Mx1 outer 1xM
    de_db = de_dz #1xN #it's just the upstream * 1 bc dz/db of wx + b is just 1 
    return de_dw, de_db, de_dx
    '''
    assert True

def test_sigmoid_grad():
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    b = np.array([1, 2, 3])
    sA, sb = smallnet.sigmoid(A), smallnet.sigmoid(b)
    # print(sA, 1-sA, sA * (1-sA))
    # print(sb, 1-sb, sb * (1-sb))

    dsA, dsb = smallnet.sigmoid_grad(A), smallnet.sigmoid_grad(b)
    assert np.array_equal(dsA, sA * (1-sA))
    assert np.array_equal(dsb, sb * (1-sb))

def test_relu_grad():
    A = np.array([[-1, 2, 3],
                  [4, 5, -3]])
    b = np.array([1, -2, 3])
    correctA = np.array([[0, 1, 1],
                         [1, 1, 0]])
    correctb = np.array([1, 0, 1])
    # print(correctA, correctb)
    assert np.array_equal(smallnet.relu_grad(A), correctA)
    assert np.array_equal(smallnet.relu_grad(b), correctb)

def test_mse_grad():
    xA = np.array([[1, 2, 3],
                  [4, 5, 6]])
    yA = np.array([[4, 2, 5],
                   [2, 5, 4]])
    xb = np.array([1, 2, 3])
    yb = np.array([1, 5, 6])
    ''' 
    mse is a scalar because it combines the loss of each of the color channels.
    mse_grad is the gradient wrt the inputs to MSE function.

    The input is 3x1 scalar so the mse_grad will be of size 3x1 for single input example.
    or nx3 for a batch size of n.

    mse_grad same size as y I think, ig distribute mse evenly among output nodes yea apparently huh
    '''
    #distribute mse across all the boys

    #pretend xA (2 samples), xb (1 sample) are our predictions 
    assert np.array_equal(2*(xA - yA) / (yA.shape[0] * yA.shape[1]), tinynet.mse_grad(yA, xA))
    assert np.array_equal(2*(xb - yb) / (len(yb)), tinynet.mse_grad(yb, xb))

#check neural network structure (num layers, num nodes), key is to see the output so that you know it's right
def test_num_layers(): #num layers, not rly num nodes
    assert len(smallnet.params) == smallnet.num_layers * 2
    assert len(largenet.params) == largenet.num_layers * 2
    
#check forward pass (forward pass shape, forward pass output on rly simple networks)

def test_forward():
    tinyX = np.array([1, -2]) #num_inputs x input_size
    tinyX2 = np.array([[1, -2],
                       [2, 3]])
    smallforward, largeforward, tinyforward = smallnet.forward(X), largenet.forward(X), tinynet.forward(tinyX2)
    # print(smallforward, ", sum each row: ",  np.sum(smallforward, axis = 1), ", sum each col: ", np.sum(smallforward, axis=0))
    #test shapes 

    # print(smallnet.input_size, smallnet.hidden_sizes, smallnet.output_size) #2 FC, sigmoid the last 
    # print(largenet.input_size, largenet.hidden_sizes, largenet.output_size) #3 FC, sigmoid the last
    fc1 = tinyX2 @ tinynet.params['W1'] + tinynet.params['b1']
    relu1 = tinynet.relu(fc1)
    fc2 = relu1 @ tinynet.params['W2'] + tinynet.params['b2']
    relu2 = tinynet.relu(fc2)
    fc3 = relu2 @ tinynet.params['W3'] + tinynet.params['b3']
    sig1 = tinynet.sigmoid(fc3)
    # print("tiny forward output: ", tinyforward)
    # print("tinynet params: ", tinynet.params)
    # print("tiny layer outputs: ", tinynet.outputs) #3 FC, sigmoid the last
    # print("fc1: ", fc1, '\n', "r1: ", relu1, '\n', "fc2: ", fc2, '\n', "r2: ", relu2, '\n', "fc3: ", fc3, '\n', "sig1: ", sig1, '\n',)
    assert np.array_equal(sig1, tinyforward) #lmao seems to work when done manually

#check backward pass (backward pass shape, backward pass output on rly simple networks)

def test_backward():
    tinyX = np.array([1, -2]) #num_inputs x input_size
    tinyy = np.array([0, -1, 2]) #num_classes
    tinyX2 = np.array([[1, -2],
                       [2, 3]])
    tinyy2 = np.array([[0, -1, 2],
                       [1, -3, 4]])
    tinynet.forward(tinyX2)
    tinynet.backward(tinyy2)

    # largenet.forward(X)
    # largenet.backward(y)
    # print(tinynet.backward(tinyy2), tinynet.outputs)
    # print("gradients: ", tinynet.gradients)
    # print("layer outputs: ", tinynet.outputs)
    # print("layer gradients: ", tinynet.gradients)
    # print("layer params: ", tinynet.params)
    # print("largenet gradients: ", largenet.gradients)
    # print("largenet layer outputs: ", largenet.outputs)

    # for key in tinynet.params:
    #     print("param: ", key, tinynet.params[key].shape)
    # for key in tinynet.gradients:
    #     print("gradient: ", key, tinynet.gradients[key].shape)
    # for key in largenet.params:
    #     print("param: ", key, largenet.params[key].shape)
    # for key in largenet.gradients:
    #     print("gradient: ", key, largenet.gradients[key].shape)
    assert True
''' 
PART 2
'''

def test_adam():
    #m and v corresponding params must have same shape as W, b params 

    #check optimizer, reset net initialization bc of t lmao it keeps going up everytime update() is called
    input_size = 2 #input layer has 2 nodes
    num_inputs = 1
    hidden_size = 2
    optimizer = "Adam" #when initialize net
    # optimizer = "Adam"
    tinynet = init_toy_model(num_layers=3)
    tinyX2 = np.array([[1, -2],
                       [2, 3]])
    tinyy2 = np.array([[0, -1, 2],
                       [1, -3, 4]])

    #shape check
    # for key in tinynet.params:
    #     print("param: ", key, tinynet.params[key].shape)
    # for key in tinynet.gradients:
    #     print("gradient: ", key, tinynet.gradients[key].shape)
    for key in tinynet.adam_params:
        # print("adam param: ", key, tinynet.adam_params[key].shape)
        original_key = key[1:] #mW1 becomes W1 string for example
        assert tinynet.adam_params[key].shape == tinynet.params[original_key].shape

    tinynet.forward(tinyX2)
    tinynet.backward(tinyy2)
    print("BEFORE: ")
    print("adam params: ", tinynet.adam_params)
    print("W, b params: ", tinynet.params)
    print("gradient: ", tinynet.gradients)
    tinynet.update(max_t = 100, opt="Adam")
    print("AFTER: ")
    print("adam params: ", tinynet.adam_params)
    print("W, b params: ", tinynet.params)

    assert True

def test_mse_time():
    xA = np.array([[1, 2, 3],
                  [4, 5, 6]]) #pretend these are 2 samples
    yA = np.array([[4, 2, 5],
                   [2, 5, 4]])
    
    xA = np.reshape(np.arange(100000000), (10000000, 10))
    yA = np.ones((10000000, 10))
    xb = np.array([1, 2, 3]) #pretend this is one sample, ig you hope len(xb) == # outputs
    yb = np.array([4, 2, 5])

    t0 = time.time()
    msenumba = mse_numba(yA, xA)
    t1 = time.time()
    total = t1-t0

    t0 = time.time()
    msenet = tinynet.mse(yA, xA)
    t1 = time.time()
    total1 = t1-t0

    # print(msenumba, msenet)
    # print("numba: ", total, ", net: ", total1)
    pass

''' 
PART 3 if needed
'''

def test_softmax():
    pass

def test_L1():
    pass
