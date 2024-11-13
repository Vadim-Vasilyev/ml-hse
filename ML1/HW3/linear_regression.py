from __future__ import annotations

from typing import List

import numpy as np

import descents
from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Linear regression class
    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Fitting descent weights for x and y dataset
        :param x: features array
        :param y: targets array
        :return: self
        """
        for _ in range(self.max_iter):
            if type(self.descent) == descents.StochasticDescent:
                indices = np.random.choice(x.shape[0], self.descent.batch_size, replace=False)
                x = x[indices]
                y = y[indices]
                
            if _ == 0:
                loss = self.calc_loss(x, y)
                self.loss_history.append(loss)
            
            diff = self.descent.step(x, y)
            loss = self.calc_loss(x, y)
            self.loss_history.append(loss)
            
            if np.isnan(diff).any():
                break
            if _ != 0 and np.sum(diff * diff) < self.tolerance:
                break
        return self
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """
        return self.descent.calc_loss(x, y)
