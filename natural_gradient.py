"""
Natural Gradient portfolio allocation using Fisher information metric.
Optimises Sharpe ratio (mean / std) to encourage differentiation.
"""

import numpy as np
from scipy.special import softmax
from scipy.linalg import solve
import config

class NaturalGradientAllocator:
    def __init__(self, n_assets: int, learning_rate: float = 0.5, fisher_damp: float = 1e-6,
                 max_iter: int = 1000, tol: float = 1e-8, risk_free_rate: float = 0.0,
                 transaction_cost: float = 0.001):
        self.n_assets = n_assets
        self.lr = learning_rate
        self.fisher_damp = fisher_damp
        self.max_iter = max_iter
        self.tol = tol
        self.rf = risk_free_rate
        self.transaction_cost = transaction_cost
        self.theta_ = None
        self.weights_ = None

    def _softmax(self, theta):
        # Use temperature to sharpen distribution
        return softmax(theta * 10.0)   # higher temperature amplifies differences

    def _portfolio_return(self, returns, weights):
        return np.dot(returns, weights)

    def _sharpe(self, portfolio_returns):
        excess = portfolio_returns - self.rf / 252
        if np.std(excess) == 0:
            return 0.0
        return np.mean(excess) / np.std(excess) * np.sqrt(252)

    def _objective_and_grad(self, theta, returns):
        w = self._softmax(theta)
        port_returns = self._portfolio_return(returns, w)
        sharpe = self._sharpe(port_returns)
        loss = -sharpe   # minimise negative Sharpe

        # Numerical gradient (robust)
        eps = 1e-6
        grad = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            w_plus = self._softmax(theta_plus)
            port_plus = self._portfolio_return(returns, w_plus)
            sharpe_plus = self._sharpe(port_plus)
            loss_plus = -sharpe_plus
            grad[i] = (loss_plus - loss) / eps
        return loss, grad

    def _fisher_matrix(self, theta, returns):
        w = self._softmax(theta)
        # Fisher for softmax: diag(w) - w w^T
        F = np.diag(w) - np.outer(w, w)
        F += self.fisher_damp * np.eye(self.n_assets)
        return F

    def _natural_gradient_step(self, theta, returns):
        loss, grad = self._objective_and_grad(theta, returns)
        F = self._fisher_matrix(theta, returns)
        try:
            natural_grad = solve(F, -grad)
        except:
            natural_grad = -grad
        theta_new = theta + self.lr * natural_grad
        return theta_new, loss

    def fit(self, returns_matrix):
        T, n = returns_matrix.shape
        assert n == self.n_assets
        theta0 = np.zeros(n)
        theta = theta0.copy()
        prev_loss = np.inf
        for it in range(self.max_iter):
            theta, loss = self._natural_gradient_step(theta, returns_matrix)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        self.theta_ = theta
        self.weights_ = self._softmax(theta)
        return self

    def predict_weights(self):
        return self.weights_

    def top_etfs(self, tickers, top_n=3):
        idx = np.argsort(self.weights_)[::-1][:top_n]
        return [(tickers[i], self.weights_[i]) for i in idx]
