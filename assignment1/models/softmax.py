"""Softmax model."""

import numpy as np


class Softmax:
  def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
    """Initialize a new classifier.

    Parameters:
        n_class: the number of classes
        lr: the learning rate
        epochs: the number of epochs to train for
        reg_const: the regularization constant
    """
    self.w = None  # TODO: change this
    self.lr = lr
    self.epochs = epochs
    self.reg_const = reg_const
    self.n_class = n_class

  def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Calculate gradient of the softmax loss.

    Inputs have dimension D, there are C classes, and we operate on
    mini-batches of N examples.

    Parameters:
        X_train: a numpy array of shape (N, D) containing a mini-batch
            of data
        y_train: a numpy array of shape (N,) containing training labels;
            y[i] = c means that X[i] has label c, where 0 <= c < C

    Returns:
        gradient with respect to weights w; an array of same shape as w
    """
    # TODO: implement me
    gradient = self.reg_const * self.w
    const_k = np.max(self.score, axis = 1, keepdims=True)
    self.score = np.exp(self.score-const_k)
    self.score /= np.sum(self.score,axis=1, keepdims=True)
    
    
    ground_truth = self.score[np.arange(X_train.shape[0]), y_train]
    self.score[np.arange(X_train.shape[0]), y_train] = 0
    
    for i in range(X_train.shape[0]):
      gradient += X_train[i][:,np.newaxis] * self.score[i]
      gradient[:, y_train[i]] += (ground_truth[i] - 1) * X_train[i]
      
    return gradient

  def train(self, X_train: np.ndarray, y_train: np.ndarray):
    """Train the classifier.

    Hint: operate on mini-batches of data for SGD.

    Parameters:
        X_train: a numpy array of shape (N, D) containing training data;
            N examples with D dimensions
        y_train: a numpy array of shape (N,) containing training labels
    """
    # TODO: implement me
    self.w = np.random.rand(X_train.shape[1],10)
    betch_size = 128
    for e in range(self.epochs):
      self.lr /= (1+e)
      for i in range(X_train.shape[0]):
        x = X_train[i*betch_size: (i+1) * betch_size]
        y = y_train[i*betch_size: (i+1) * betch_size]
        self.score = x.dot(self.w)
        self.w -= self.lr * self.calc_gradient(x,y)
    return

  def predict(self, X_test: np.ndarray) -> np.ndarray:
    """Use the trained weights to predict labels for test data points.

    Parameters:
        X_test: a numpy array of shape (N, D) containing testing data;
            N examples with D dimensions

    Returns:
        predicted labels for the data in X_test; a 1-dimensional array of
            length N, where each element is an integer giving the predicted
            class.
    """
    # TODO: implement me
    pred = X_test.dot(self.w).argmax(axis=1)
    return pred
