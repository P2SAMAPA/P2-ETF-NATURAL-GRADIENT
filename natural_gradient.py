"""
Natural Gradient portfolio allocation using Fisher information metric.
Optimises Sortino ratio over a historical window.
"""

import numpy as np
from scipy.special import softmax
from scipy.linalg import solve
import warnings

class NaturalGradientAllocator:
    """
    Implements natural gradient descent on the simplex of portfolio weights.
    The loss is negative Sortino ratio (to minimise).
    """

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

        # Parameterise weights via unconstrained vector theta, then softmax
        self.theta_ = None          # unconstrained parameters
        self.weights_ = None        # final portfolio weights (softmax(theta))

    def _softmax(self, theta):
        return softmax(theta)

    def _portfolio_return(self, returns, weights):
        """Daily portfolio log return (approx simple for Sortino)."""
        # weights: (n_assets,); returns: (T, n_assets)
        return np.dot(returns, weights)

    def _sortino(self, portfolio_returns):
        """Annualised Sortino ratio (negative for minimisation)."""
        excess = portfolio_returns - self.rf / 252   # risk‑free daily
        downside = excess[excess < self.downside_threshold]
        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0  # no downside → infinite Sortino, but treat as 0
        annualised_sortino = np.mean(excess) / np.std(downside) * np.sqrt(252)
        return annualised_sortino

    def _objective_and_grad(self, theta, returns):
        """Return negative Sortino and its gradient w.r.t theta."""
        w = self._softmax(theta)
        port_returns = self._portfolio_return(returns, w)

        # Sortino
        sortino = self._sortino(port_returns)
        loss = -sortino

        # Gradient of loss w.r.t w (from chain rule)
        # We need d(loss)/dw = - d(Sortino)/dw
        # Hedge: approximate gradient of Sortino w.r.t port returns:
        excess = port_returns - self.rf/252
        mu = np.mean(excess)
        sigma_down = np.std(excess[excess < self.downside_threshold])
        if sigma_down == 0:
            grad_loss_w = np.zeros(self.n_assets)
        else:
            # derivative of Sortino w.r.t μ: 1/σ_down * sqrt(252)
            d_mu = 1.0 / sigma_down * np.sqrt(252)
            # derivative of Sortino w.r.t σ_down: -μ / σ_down^2 * sqrt(252)
            d_sigma = -mu / (sigma_down**2) * np.sqrt(252)
            # Gradient of μ w.r.t w: average of returns across time
            grad_mu = np.mean(returns, axis=0)
            # Gradient of σ_down w.r.t w: we approximate using only downside returns
            # More precise: compute derivative of standard deviation of downside returns
            downside_returns = excess[excess < self.downside_threshold]
            if len(downside_returns) == 0:
                grad_sigma = np.zeros(self.n_assets)
            else:
                # Multiply downside returns by themselves? Actually:
                # σ = sqrt(mean(x^2) - mean(x)^2) for downside part
                # This is messy. For efficiency, we use finite difference approximation.
                # Instead, we approximate natural gradient directly using Fisher of the predictive distribution.
                # We'll use a simpler empirical Fisher based on output probabilities.
                pass

        # Simpler: use automatic differentiation via finite differences?
        # For clarity and robustness, we will estimate gradient numerically (slow but reliable for moderate n_assets <= 100).
        eps = 1e-5
        grad = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            w_plus = self._softmax(theta_plus)
            port_plus = self._portfolio_return(returns, w_plus)
            sortino_plus = self._sortino(port_plus)
            loss_plus = -sortino_plus
            grad[i] = (loss_plus - loss) / eps
        return loss, grad

    def _fisher_matrix(self, theta, returns):
        """
        Approximate Fisher Information Matrix of the portfolio distribution.
        We treat the portfolio as a categorical distribution over assets with probabilities w.
        Then Fisher = diag(w) - w w^T (for softmax parameterisation).
        This is the well‑known Fisher for the multinomial distribution.
        """
        w = self._softmax(theta)
        # Fisher = diag(w) - w w^T
        F = np.diag(w) - np.outer(w, w)
        # Damping
        F += self.fisher_damp * np.eye(self.n_assets)
        return F

    def _natural_gradient_step(self, theta, returns):
        loss, grad = self._objective_and_grad(theta, returns)
        F = self._fisher_matrix(theta, returns)
        # Solve F * natural_grad = -grad
        try:
            natural_grad = solve(F, -grad)
        except np.linalg.LinAlgError:
            natural_grad = -grad   # fallback
        theta_new = theta + self.lr * natural_grad
        return theta_new, loss

    def fit(self, returns_matrix):
        """
        returns_matrix : (T, n_assets) array of daily log returns for the training window.
        """
        T, n = returns_matrix.shape
        assert n == self.n_assets
        # Initialise theta as equal weights
        theta0 = np.zeros(n)   # softmax(0)=1/n
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
        """Return optimal portfolio weights (array of length n_assets)."""
        return self.weights_

    def top_etfs(self, tickers, top_n=3):
        """Return top N ETFs by weight."""
        idx = np.argsort(self.weights_)[::-1][:top_n]
        return [(tickers[i], self.weights_[i]) for i in idx]
