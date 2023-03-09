"""Perceptron model."""

import numpy as np


class Perceptron:
  def __init__(self, n_class: int, lr: float, epochs: int):
    """Initialize a new classifier.

    Parameters:
        n_class: the number of classes
        lr: the learning rate
        epochs: the number of epochs to train for
    """
    self.w = None
    self.lr = lr
    self.epochs = epochs
    self.n_class = n_class

  def train(self, X_train: np.ndarray, y_train: np.ndarray):
    """Train the classifier.

    Use the perceptron update rule as introduced in the Lecture.

    Parameters:
        X_train: a number array of shape (N, D) containing training data;
            N examples with D dimensions
        y_train: a numpy array of shape (N,) containing training labels
    """
    N, D = X_train.shape
    C = self.n_class
    self.w = np.random.randn(C, D)

    for epoch in range(self.epochs):
      for i in range(N):
        w_yi = self.w[y_train[i]] @ X_train[i]
        for c in range(C):
          w_c = self.w[c] @ X_train[i]
          if w_c > w_yi:
            self.w[y_train[i]] += (self.lr * X_train[i])
            self.w[c] -= (self.lr * X_train[i])

        self.lr *= np.exp(-1 * (epoch * 0.001))

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
    # N, D = X_test.shape
    # self.w = self.w.reshape(D, 1)
    pred = np.dot(X_test, self.w.T)
    return np.argmax(pred, axis=1)
