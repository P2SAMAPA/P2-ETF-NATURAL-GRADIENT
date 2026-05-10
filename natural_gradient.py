"""
Natural Gradient portfolio allocation using Fisher information metric.
Optimises mean‑downside utility (return‑seeking with tunable emphasis).
"""

import numpy as np
from scipy.special import softmax
from scipy.linalg import solve
import config   # to access RETURN_EMPHASIS

class NaturalGradientAllocator:
    def __init__(self, n_assets: int, learning_rate: float = 0.1, fisher_damp: float = 1e-4,
                 max_iter: int = 500, tol: float = 1e-6, risk_free_rate: float = 0.0,
                 downside_threshold: float = 0.0, transaction_cost: float = 0.001):
        self.n_assets = n_assets
        self.lr = learning_rate
        self.fisher_damp = fisher_damp
        self.max_iter = max_iter
        self.tol = tol
        self.rf = risk_free_rate
        self.downside_threshold = downside_threshold
        self.transaction_cost = transaction_cost
        self.theta_ = None
        self.weights_ = None

    def _softmax(self, theta):
        return softmax(theta)

    def _portfolio_return(self, returns, weights):
        return np.dot(returns, weights)

    def _objective_and_grad(self, theta, returns):
        """
        Objective: negative of (mean_excess - lam * downside_std)
        where lam = 1 / RETURN_EMPHASIS
        """
        w = self._softmax(theta)
        port_returns = self._portfolio_return(returns, w)
        excess = port_returns - self.rf / 252
        mean_excess = np.mean(excess)
        downside = excess[excess < self.downside_threshold]
        if len(downside) == 0:
            downside_std = 0.0
        else:
            downside_std = np.std(downside)
        # Weight on downside penalty (lower lam = more return‑seeking)
        lam = 1.0 / config.RETURN_EMPHASIS
        utility = mean_excess - lam * downside_std
        loss = -utility

        # Numerical gradient (robust, works for any objective)
        eps = 1e-5
        grad = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            w_plus = self._softmax(theta_plus)
            port_plus = self._portfolio_return(returns, w_plus)
            excess_plus = port_plus - self.rf / 252
            mean_plus = np.mean(excess_plus)
            downside_plus = excess_plus[excess_plus < self.downside_threshold]
            if len(downside_plus) == 0:
                std_plus = 0.0
            else:
                std_plus = np.std(downside_plus)
            utility_plus = mean_plus - lam * std_plus
            loss_plus = -utility_plus
            grad[i] = (loss_plus - loss) / eps
        return loss, grad

    def _fisher_matrix(self, theta, returns):
        w = self._softmax(theta)
        F = np.diag(w) - np.outer(w, w)
        F += self.fisher_damp * np.eye(self.n_assets)
        return F

    def _natural_gradient_step(self, theta, returns):
        loss, grad = self._objective_and_grad(theta, returns)
        F = self._fisher_matrix(theta, returns)
        try:
            natural_grad = solve(F, -grad)
        except np.linalg.LinAlgError:
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
