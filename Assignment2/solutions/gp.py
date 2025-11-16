# ---------- gaussian_process_regressor.py ----------
import numpy as np
from numpy.linalg import cholesky, solve

########### TO-DO ###########
# Implement the methods marked with "BEGINNING OF YOUR CODE" and "END OF YOUR CODE":
# fit(), predict()
# Do not change the function signatures
# Do not change any other code
#############################

class GaussianProcess:
    """
    Gaussian Process Regression (kernel/dual form) with optional variance prediction.

    A Gaussian Process defines a prior over functions:
        y(x) ~ GP(0, k(x, x') + sigma_n^2 δ(x, x'))

    where k is a positive semi-definite kernel (RBF, Matern, etc.) and
    sigma_n^2 is the noise variance.

    Training (fit):
        * Compute the Gram matrix with noise:
              K = k(X, X) + sigma_n^2 I
        * Solve for alpha = K^{-1} y
          (in practice use a numerically stable method such as Cholesky).

    Prediction (predict):
        For a test input x*:
            mean(x*) = k(X, x*)^T alpha
            var(x*)  = k(x*, x*) - v^T v,
                        where v = L^{-1} k(X, x*) and L is the Cholesky of K.

    Notes
    -----
    * This is the dual/kernel form: predictions are expressed as
      linear combinations of kernel evaluations between training and test data.
    * Variance output quantifies uncertainty of the GP posterior.
    """

    def __init__(self, kernel, noise_variance=1e-6):
        if kernel is None:
            raise ValueError("Provide a sklearn kernel object, e.g., RBF(length_scale=1.0).")
        self.kernel = kernel
        self.noise_variance = float(noise_variance)

        self.X_train_ = None
        self.y_train_ = None
        self.L_ = None       # Cholesky factor of K
        self.alpha_ = None   # K^{-1} y

    def fit(self, X, y):
        """
        Fit the Gaussian Process model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.
        y : array-like of shape (n_samples,)
            Training target values.

        Task
        ----
        * Build the kernel Gram matrix K(X, X).
        * Add noise variance alpha_n^2 I to stabilize inversion.
        * Compute a Cholesky factorization K = L L^T.
        * Solve for alpha = K^{-1} y efficiently using triangular solves.

        Returns
        -------
        self : GaussianProcess
            Fitted GP model with stored training data, Cholesky factor L, and alpha.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = X.shape[0]

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Compute and store training data
        self.X_train_ = X
        self.y_train_ = y

        # Build Gram matrix K(X, X)
        K = self.kernel(X, X)

        # Add noise variance to the diagonal for numerical stability
        K += self.noise_variance * np.eye(n)

        # Compute the Cholesky decomposition: K = L L^T
        self.L_ = cholesky(K)

        # Solve for alpha using triangular solves:
        # First: solve L * z = y
        z = solve(self.L_, y)

        # Second: solve L^T * alpha = z
        self.alpha_ = solve(self.L_.T, z)

        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return self

    def predict(self, X):
        """
        Predict posterior mean and standard deviation for test inputs.

        Parameters
        ----------
        X : array-like of shape (m, n_features)
            Test input data.

        Task
        ----
        * Compute kernel cross-covariance K(X_train, X).
        * Predictive mean:
              μ(x*) = K(X_train, x*)^T alpha
        * Standard deviation:
            - Compute k(x*, x*) for each test point.
            - Solve v = L^{-1} K(X_train, x*) using Cholesky factor.
            - Predictive variance: sigma^2(x*) = k(x*, x*) - ||v||^2.
            - Return sqrt of variance as standard deviation.

        Returns
        -------
        mean : np.ndarray of shape (m,)
            Posterior predictive mean.
        std : np.ndarray of shape (m,),
            Posterior predictive standard deviation.
        """

        if self.alpha_ is None:
            raise RuntimeError("Model is not fit yet.")
        X = np.asarray(X, dtype=np.float64)

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Compute cross-kernel matrix K(X_train, X_test)
        K_trans = self.kernel(self.X_train_, X)

        # Posterior predictive mean: K^T * alpha
        mean = K_trans.T @ self.alpha_

        # Compute v = L^{-1} * K_trans using triangular solve
        v = solve(self.L_, K_trans)

        # Compute kernel diag k(x*, x*)
        k_xx = self.kernel(X, X).diagonal()

        # Predictive variance = k(x*, x*) - sum(v^2, axis=0)
        var = k_xx - np.sum(v**2, axis=0)

        # Standard deviation is sqrt of variance
        # Numerical safety: clip tiny negatives due to floating point
        std = np.sqrt(np.maximum(var, 1e-12))

        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return mean, std
