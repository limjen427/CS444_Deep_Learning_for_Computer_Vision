"""Logistic regression model."""

import numpy as np


class Logistic:
  def __init__(self, lr: float, epochs: int, threshold: float):
    """Initialize a new classifier.

    Parameters:
        lr: the learning rate
        epochs: the number of epochs to train for
    """
    self.w = None  # TODO: change this
    self.lr = lr
    self.epochs = epochs
    self.threshold = threshold

  def sigmoid(self, z: np.ndarray) -> np.ndarray:
    """Sigmoid function.

    Parameters:
        z: the input

    Returns:
        the sigmoid of the input
    """
    # TODO: implement me
    return 1 / (1 + np.exp(-z))

  def train(self, X_train: np.ndarray, y_train: np.ndarray):
    """Train the classifier.

    Use the logistic regression update rule as introduced in lecture.

    Parameters:
        X_train: a numpy array of shape (N, D) containing training data;
            N examples with D dimensions
        y_train: a numpy array of shape (N,) containing training labels
    """
    # TODO: implement me
    N, D = X_train.shape
    self.w = np.random.randn(D)

    for epoch in range(self.epochs):
      z = np.dot(X_train, self.w)
      y_hat = self.sigmoid(z)
      grad = (1 / N) * np.dot(X_train.T, y_hat - y_train)
      self.w -= self.lr * grad
      self.lr *= np.exp(-1 * (epoch * 0.002))

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
    z = np.dot(X_test, self.w)
    y_hat = self.sigmoid(z)
    pred = [1 if i > self.threshold else 0 for i in y_hat]
    return pred