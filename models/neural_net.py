"""Neural network model."""

from typing import Sequence

import numpy as np

class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str, #SGD, Adam
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.outputs = {} 
        self.gradients = {}
        self.t = 0 #Adam t: epochs * batches_per_epoch

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        self.adam_params = {} 

        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            # if opt == "Adam":
            self.adam_params["mW" + str(i)] = np.zeros((sizes[i - 1], sizes[i]))
            self.adam_params["mb" + str(i)] = np.zeros(sizes[i])
            self.adam_params["vW" + str(i)] = np.zeros((sizes[i - 1], sizes[i]))
            self.adam_params["vb" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        return X @ W + b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg = 0.1, N = 3) -> np.ndarray:
        """Gradient of linear layer
            returns de_dw, de_db, de_dx
        """
        de_dx = de_dz @ W.T #1xM @ MxN
        de_dw = np.dot(X.T, de_dz) # Mx1 dot 1xM
        de_db = np.sum(de_dz, axis=0) #1xN, sum samples
        return de_dw, de_db, de_dx

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Returns max(0, X)
        """
        # https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy 
        return X * (X > 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Returns element-wise indicator I[X > 0]
        """
        return X > 0

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        ''' 
        Returns sigmoid by row
        '''
        return 1/(1+np.exp(-x))
       

    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        sig = self.sigmoid(X)
        return sig * (1-sig) 
    
    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray: #scalar version
        ''' 
        Returns scalar mse averaged over num_inputs * num_dims
        '''
        # if y is targets and p is predictions, AVERAGED OVER OUTPUT NODES not batch size 
        # https://www.bragitoff.com/2021/12/mean-squared-error-loss-function-and-its-gradient-derivative-for-a-batch-of-inputs-python-code/
        # can also use numba for JIT speedup
        num_outputs = len(p) #1d array
        if len(p.shape) > 1: #2d array
            num_outputs = y.shape[0] * y.shape[1] 
        return np.sum((y-p)**2)/num_outputs #this is a scalar, nxfeatures
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        ''' 
        Returns mse_grad as (num_inputs, num_dims aka shape of y) averaged over num_inputs * num_dims
        '''
        num_outputs = len(p) #1d array
        if len(p.shape) > 1: #2d array
            num_outputs = y.shape[0] * y.shape[1] 
        return 2*(p-y)/num_outputs #1xn_outputs 

    def softmax():
        return
    
    def softmax_grad():
        return
    
    def cross_entropy():
        return
    
    def cross_entropy_grad():
        return
    
    def forward(self, X: np.ndarray, final_activation = "sigmoid") -> np.ndarray:
        """Compute the outputs for X, a batch of samples
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        self.outputs:
            bi: output after each linear layer 
            zi: output after each non-linearity (relu, sigmoid at last layer)
            z0: data 
        """
        self.outputs = {} 
    
        z = X 
        self.outputs['z0'] = X
        for i in range(1, self.num_layers + 1): 
            W = self.params["W" + str(i)] 
            b = self.params["b" + str(i)]
            #linear
            z_b = self.linear(W, z, b) 
            self.outputs['b' + str(i)] = z_b
            if i == self.num_layers:
                #if at last layer do final activation
                if final_activation == "sigmoid":
                    z = self.sigmoid(z_b)
                #elif "softmax"
            else:
                z = self.relu(z_b) 
            self.outputs["z" + str(i)] = z 
        return z #sigmoid output

    def backward(self, y: np.ndarray, final_activation = "sigmoid") -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        self.gradients:
            Wi: gradient used to update Wi, same shape as Wi
            bi: gradient used to update bi, same shape as bi
        """
        self.gradients = {}

        de_dz = np.ones_like(self.output_size) 
        for i in range(self.num_layers, 0, -1):
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            zprev = self.outputs['z' + str(i-1)] 
            
            if i == self.num_layers: #last layer is [fc, sig, mse] 

                if final_activation == "sigmoid":
                    mse_grad = self.mse_grad(y, self.outputs['z' + str(i)]) #nx3 (3 outputs)
                    sigmoid_grad = self.sigmoid_grad(self.outputs['b' + str(i)]) #nx3
                    de_dz = sigmoid_grad * mse_grad #local * upstream element-wise
                
                #elif "softmax"

                #linear gradient
                de_dw, de_db, de_dx = self.linear_grad(W, zprev, b, de_dz) 
            else:
                #assume de_dz is initiallized from the final layer
                de_dz = de_dz * self.relu_grad(self.outputs['b' + str(i)]) #relu grad

                #linear
                de_dw, de_db, de_dx = self.linear_grad(W, zprev, b, de_dz) 
            #store dw, db
            self.gradients['W' + str(i)] = de_dw
            self.gradients['b' + str(i)] = de_db

            #send downstream to next layer
            de_dz = de_dx 
            
        return self.mse(y, self.outputs['z' + str(self.num_layers)]) 

    def update(
        self,
        max_t: int,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD"
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            max_t: epochs * batches_per_epoch
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        if opt == "SGD":
            for key in self.params:
                self.params[key] = self.params[key] - (lr * self.gradients[key])
        else:
            if self.t < max_t: 
                for i in range(1, self.num_layers + 1):
                    mWprev = self.adam_params['mW' + str(i)]
                    vWprev = self.adam_params['vW' + str(i)]
                    mbprev = self.adam_params['mb' + str(i)]
                    vbprev = self.adam_params['vb' + str(i)]

                    gradW = self.gradients["W" + str(i)]
                    gradb = self.gradients["b" + str(i)]

                    #moment calculations for W, b
                    mW = b1 * mWprev + (1.0 - b1) * gradW #moment calculation
                    self.adam_params['mW' + str(i)] = mW #update m, v

                    vW = b2 * vWprev + (1.0 - b2) * (gradW**2)
                    self.adam_params['vW' + str(i)] = vW

                    mb = b1 * mbprev + (1.0 - b1) * gradb 
                    self.adam_params['mb' + str(i)] = mb

                    vb = b2 * vbprev + (1.0 - b2) * (gradb**2)
                    self.adam_params['vb' + str(i)] = vb

                    #bias correction
                    mWhat = mW / (1.0 - b1**(self.t+1))
                    vWhat = vW / (1.0 - b2**(self.t+1))
                    mbhat = mb / (1.0 - b1**(self.t+1))
                    vbhat = vb / (1.0 - b2**(self.t+1))

                    #update params
                    self.params['W' + str(i)] = self.params['W' + str(i)] - lr * mWhat / (np.sqrt(vWhat) + eps)
                    self.params['b' + str(i)] = self.params['b' + str(i)] - lr * mbhat / (np.sqrt(vbhat) + eps)

            self.t += 1 #num times update is called
        return