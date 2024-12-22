from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional

import matplotlib.pyplot as plt

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds=None,
        bootstrap_type='Bernoulli',
        subsample=1.0,
        bagging_temperature=1.0,
        quantization_type=None, # Uniform/Quantile
        nbins: int = 255,
        rsm=None
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.early_stopping_rounds = early_stopping_rounds

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature

        self.rsm = rsm
        self.model_features: list = []

        self.quantization_type = quantization_type
        self.nbins = nbins

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self._feature_importances_ = None

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean() # Функция потерь рассчитана на y = +-1; в данных имеем {0, 1}
        self.loss_derivative = lambda y, z: -y * (1 - self.sigmoid(y * z))
            

    def partial_fit(self, X, y):
        if self.bootstrap_type == 'Bernoulli':
            indicies = np.random.choice(a=X.shape[0], size=int(np.ceil(self.subsample * X.shape[0])))
            X, y = X[indicies], y[indicies]
        elif self.bootstrap_type == 'Bayesian':
            weights = np.random.uniform(0, 1, X.shape[0])
            weights = (-np.log(weights))**self.bagging_temperature
            X = X.toarray() * weights.reshape(-1, 1)
        
        if self.rsm:
            used_features = np.random.choice(np.arange(X.shape[1]), self.rsm, replace=False)
            X = X[:, used_features]
            self.model_features.append(used_features)
        
        new_model = self.base_model_class(**self.base_model_params)
        new_model.fit(X, y)
        return new_model

    def fit(self,
            X_train, 
            y_train, 
            X_val=None, 
            y_val=None,
            plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        # Переводим таргет в {-1, 1}
        y_train = np.where(y_train == 1, 1, -1)

        # первая базовая модель
        old_predictions = np.zeros(y_train.shape[0])

        if X_val is not None and y_val is not None:
            val_predictions = np.zeros(y_val.shape[0])
            val_loss = self.loss_fn(y_val, val_predictions)
            self.history['val_loss'].append(val_loss)

        if self.early_stopping_rounds:
            decrease_steps = 0

        if self.quantization_type:
            X_train = self._quantize(X_train)

        if type(self.rsm) is float:
            self.rsm = int(np.ceil(X_train.shape[1] * self.rsm))
        
        for _ in range(self.n_estimators):
            S = -self.loss_derivative(y_train, old_predictions)
            new_model = self.partial_fit(X_train, S)
            new_predictions = new_model.predict(X_train) if not self.rsm else new_model.predict(X_train[:, self.model_features[-1]])
            best_gamma = self.find_optimal_gamma(y_train, old_predictions, new_predictions)
            
            self.models.append(new_model)
            self.gammas.append(best_gamma)

            old_predictions += self.learning_rate * best_gamma * new_predictions
            self.history['train_loss'].append(self.loss_fn(y_train, old_predictions))

            if X_val is not None and y_val is not None:
                val_predictions += self.learning_rate * best_gamma * (new_model.predict(X_val) 
                                                                      if not self.rsm else new_model.predict(X_val[:, self.model_features[-1]]))
                new_val_loss = self.loss_fn(y_val, val_predictions)
                self.history['val_loss'].append(new_val_loss)
            
            if self.early_stopping_rounds:
                if new_val_loss > val_loss:
                    decrease_steps += 1
                val_loss = new_val_loss
                if decrease_steps == self.early_stopping_rounds:
                    break

        if self.base_model_class == DecisionTreeRegressor:
            self._set_feature_importances()

        if plot:
            plt.plot(self.history['train_loss'], label='train')
            plt.ylabel('loss')
            plt.xlabel('models in ensemble')
            plt.legend()

    def predict_proba(self, X):
        if not self.rsm:
            steps = np.array([gamma * model.predict(X) * self.learning_rate
                              for gamma, model in zip(self.gammas, self.models)])
        else:
            steps = np.array([gamma * model.predict(X[:, features]) * self.learning_rate
                              for gamma, model, features in zip(self.gammas, self.models, self.model_features)])
            
        z = steps.sum(axis=0)
        p = self.sigmoid(z)
        return np.stack([1-p, p]).T
        
    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    @property
    def feature_importances_(self):
        if self._feature_importances_ is None:
            raise AttributeError('Can\'t access feature importances before fitting')
        return self._feature_importances_

    def _quantize(self, X):
        X = X.toarray()
        for i in range(X.shape[1]):
            if self.quantization_type == 'Uniform':
                bins = np.linspace(start=X[:, i].min(), stop=X[:, i].max(), num=self.nbins + 1)
            elif self.quantization_type == 'Quantile':
                bins = np.quantile(a=X[:, i], q=np.linspace(0, 1, self.nbins + 1))
            else:
                raise ValueError('Unknown quantization type')
                
            bin_numbers = np.digitize(X[:, i].reshape(1, -1), bins) - 1
            X[:, i] = bins[bin_numbers][:, np.newaxis]
        return X

    def _set_feature_importances(self):
        unscaled_importances = np.sum([model.feature_importances_ for model in self.models], axis=0)
        self._feature_importances_ = unscaled_importances / (np.sum(unscaled_importances))

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, X, y, title):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        loss_history = []
        predictions = np.zeros(y.shape[0])

        if not self.rsm:
            for gamma, model in zip(self.gammas, self.models):
                predictions += self.learning_rate * gamma * model.predict(X)
                loss_history.append(self.loss_fn(y, predictions))
        else:
            for gamma, model, features in zip(self.gammas, self.models, self.model_features):
                predictions += self.learning_rate * gamma * model.predict(X[:, features])
                loss_history.append(self.loss_fn(y, predictions))

        plt.plot(loss_history, label=title)
        plt.ylabel('loss')
        plt.xlabel('models in ensemble')
        plt.legend()
