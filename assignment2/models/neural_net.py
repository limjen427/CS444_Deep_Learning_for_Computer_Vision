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
        optimizer: str
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
        self.optimizer = optimizer

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])


    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return np.dot(X, W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return np.where(X > 0, 1, 0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
      # TODO ensure that this is numerically stable
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_grad(self,x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the sigmoid function with respect to its input x.
        Parameters:
            x: input
        Returns:
            Derivative of the sigmoid function with respect to x
        """
        return x * (1 - x)

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
      # TODO implement this
        return np.mean(np.square(y - p))

    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return -2 * (y - p) / y.size

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
    
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.

        self.outputs = {}
        input_x = X
        self.outputs['out0'] = input_x
        for i in range(1, self.num_layers+1):
            prev_out = self.outputs['out' + str(i-1)]
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]

            # print('forward', i, 'X:', prev_out.shape, 'W:', W.shape)
            linear = self.linear(W, prev_out, b)
            # print('b shape:', b.shape)
            self.outputs['linear' + str(i)] = linear

            if i < self.num_layers:
                output = self.relu(linear)
            else:
                output = self.sigmoid(linear)

            
            self.outputs['out' + str(i)] = output

        return self.outputs['out' + str(self.num_layers)]
        
    

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.sigmoid_grad if it helps organize your code.
        # Compute the output of the network

        self.gradients = {}
        final_out = self.outputs['out' + str(self.num_layers)]
        N = y.shape[0]

        loss = self.mse(y, final_out) # was not T before

        # print('y shape:', y.shape, 'final_out shape:', final_out.shape)

        prev_grad = self.sigmoid_grad(final_out) * self.mse_grad(y, final_out) # was not transposed before
        dW = np.dot(self.outputs['out' + str(self.num_layers-1)].T, prev_grad)
        db = np.sum(prev_grad, axis=0, keepdims=True) #0
        # print('grad b shape:', db.shape)
        self.gradients['W' + str(self.num_layers)] = dW
        self.gradients['b' + str(self.num_layers)] = db
        
        
        for i in range(self.num_layers - 1, 0, -1):
            relu_grad = self.relu_grad(self.outputs['linear' + str(i)])
            grad = np.dot(prev_grad, self.params['W' + str(i + 1)].T) * relu_grad
            prev_grad = grad
            dW = np.dot(self.outputs['out' + str(i-1)].T, grad) # was transposed before
            db = np.sum(grad, axis=0, keepdims=True)

            self.gradients['W' + str(i)] = dW
            self.gradients['b' + str(i)] = db

        return loss


    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = 'Adam'
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == "SGD":
          for i in range(1, self.num_layers + 1):
            # print(self.params['W' + str(i)].shape, self.gradients['W' + str(i)].shape)
            self.params['W' + str(i)] = self.params['W' + str(i)] - (lr * self.gradients['W' + str(i)])
            self.params['b' + str(i)] = self.params['b' + str(i)] - (lr * self.gradients['b' + str(i)])
        elif opt == "Adam":
          t = 1;
          self.m = {}
          self.v = {}
          for i in range(1, self.num_layers + 1):
            self.m['W' + str(i)] = np.zeros_like(self.params['W' + str(i)])
            self.m['b' + str(i)] = np.zeros_like(self.params['b' + str(i)])
            self.v['W' + str(i)] = np.zeros_like(self.params['W' + str(i)])
            self.v['b' + str(i)] = np.zeros_like(self.params['b' + str(i)])
              
          self.m["W" + str(i)] = b1 * self.m["W" + str(i)] + (1 - b1) * self.gradients["W" + str(i)]
          self.v["W" + str(i)] = b2 * self.v["W" + str(i)] + (1 - b2) * (self.gradients["W" + str(i)] ** 2)
          m_hat = self.m["W" + str(i)] / (1 - b1 ** t)
          v_hat = self.v["W" + str(i)] / (1 - b2 ** t)
          self.params["W" + str(i)] -= lr * m_hat / (np.sqrt(v_hat) + eps)

          self.m["b" + str(i)] = b1 * self.m["b" + str(i)] + (1 - b1) * self.gradients["b" + str(i)]
          self.v["b" + str(i)] = b2 * self.v["b" + str(i)] + (1 - b2) * (self.gradients["b" + str(i)] ** 2)
          m_hat = self.m["b" + str(i)] / (1 - b1 ** t)
          v_hat = self.v["b" + str(i)] / (1 - b2 ** t)
          temp = lr * m_hat / (np.sqrt(v_hat) + eps)
          temp = temp.reshape(-1)
          self.params["b" + str(i)] = self.params["b" + str(i)] - temp

        if opt == "Adam":
            t += 1


        