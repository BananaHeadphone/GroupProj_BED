import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
import pymc as pm
import arviz as az

def kl_divergence_gaussian_params(mu_p, Sigma_p, mu_q, Sigma_q, epsilon=1e-8):
    """
    KL( N(mu_p, Sigma_p) || N(mu_q, Sigma_q) ).
    """
    mu_p = np.asarray(mu_p)
    mu_q = np.asarray(mu_q)
    Sigma_p = np.asarray(Sigma_p)
    Sigma_q = np.asarray(Sigma_q)

    # regularize
    d = mu_p.shape[0]
    Sigma_p += epsilon * np.eye(d)
    Sigma_q += epsilon * np.eye(d)

    inv_Sig_q = np.linalg.inv(Sigma_q)
    delta = mu_q - mu_p

    trace_term    = np.trace(inv_Sig_q @ Sigma_p)
    maha_term     = delta.T @ inv_Sig_q @ delta
    sign_p, logdet_p = np.linalg.slogdet(Sigma_p)
    sign_q, logdet_q = np.linalg.slogdet(Sigma_q)
    # assemble
    kl = 0.5 * ( trace_term
               + maha_term
               - d
               + (logdet_q - logdet_p) )
    return float(kl)


def kl_divergence_gaussian_known_q(data_p, mu_q, cov_q, epsilon=1e-8):
    """
    Estimate KL( P || Q ) under the assumption that P and Q are multivariate Gaussians,
    where P is estimated from data_p, and Q's parameters are known:
    
      data_p: array of shape (N, d) sampled from P
      mu_q:   array of shape (d,) — mean of Q
      cov_q:  array of shape (d, d) — covariance of Q
    
    Returns:
      scalar KL divergence.
    """
    # 1) Estimate P’s mean and (regularized) covariance
    mu_p = np.mean(data_p, axis=0)
    cov_p = np.cov(data_p, rowvar=False)
    cov_p += epsilon * np.eye(data_p.shape[1])
    
    # 2) Regularize Q’s covariance just in case
    cov_q = cov_q + epsilon * np.eye(data_p.shape[1])
    
    # 3) Precompute common terms
    inv_cov_q = np.linalg.inv(cov_q)
    d = data_p.shape[1]
    delta = mu_q - mu_p       # difference of means
    
    # 4) Compute individual pieces of the analytic Gaussian‐KL
    trace_term  = np.trace(inv_cov_q @ cov_p)
    mahalanobis = delta.T @ inv_cov_q @ delta
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    # (we assume sign_p and sign_q > 0 after reg.)
    
    # 5) Assemble final KL
    kl = 0.5 * (trace_term + mahalanobis - d + (logdet_q - logdet_p))
    return float(kl)

def kl_divergence_independent_normal(
    A,                    # shape (d1, d2)
    y,                    # shape (d1, d3)
    posterior_samples,    # shape (N, d2, d3)
    sigma=0.2,           # likelihood std
    sigma0=4.0,           # prior std (sqrt(4))
    mu0=5.0,              # prior mean
):
    # dims
    d1, d3 = y.shape
    N      = posterior_samples.shape[0]

    # 1) Posterior expected log-likelihood: E_{p(θ|y)}[ log p(y|θ) ]
    mu_post     = np.einsum('ij,njk->nik', A, posterior_samples)    # (N, d1, d3)
    diff_post   = y[None] - mu_post                                 # (N, d1, d3)
    sq_err_post = np.sum(diff_post**2, axis=(1,2))                  # (N,)
    const_term  = -0.5 * d1 * d3 * np.log(2*np.pi*sigma**2)
    loglik_post = const_term - 0.5*(sq_err_post / sigma**2)         # (N,)
    avg_loglik_post = np.mean(loglik_post)                          # scalar

    # 2) Exact log-evidence for Gaussian prior N(mu0, sigma0^2 I):
    #    y ~ N(A·mu0,  sigma^2 I + sigma0^2 A A^T)
    # 2a) Compute shifted mean of y
    mu0_mat = mu0 * np.ones((A.shape[1], d3))  # (d2, d3)
    mu_y    = A @ mu0_mat                      # (d1, d3)

    # 2b) Build total covariance
    Cov = sigma**2 * np.eye(d1) + sigma0**2 * (A @ A.T)

    # 2c) Cholesky factorization for stability
    L = np.linalg.cholesky(Cov)                # Cov = L @ L.T

    # 2d) Solve Σ^{-1}(y - mu_y) efficiently
    y_shifted = y - mu_y                       # (d1, d3)
    tmp       = np.linalg.solve(L,    y_shifted)   # (d1, d3)
    alpha     = np.linalg.solve(L.T,  tmp)         # (d1, d3)

    # 2e) Log-det of Cov
    logdetC = 2 * np.sum(np.log(np.diag(L)))   # scalar

    # 2f) Quadratic forms for each of the d3 columns
    quad_form   = np.sum(y_shifted * alpha, axis=0)  # (d3,)

    # 2g) Log-evidence per column, then **sum** over columns
    #     (so we get log p(Y) for the whole data matrix)
    log_py_cols = -0.5*d1*np.log(2*np.pi) - 0.5*logdetC - 0.5*quad_form  # (d3,)
    log_py      = np.sum(log_py_cols)                                  # scalar

    # 3) KL = E_p[log p(y|θ)] - log p(y)  — guaranteed ≥0
    kl = avg_loglik_post - log_py
    return kl







def kl_divergence_independent_features(P, Q, k=20, eps = 1e-10):
    """
    Joint KL(P||Q) via k-NN (Kozachenko–Leonenko).
    P, Q: (n_samples, n_dims)
    k: neighbors
    eps: small constant to avoid log(0)
    """
    n, d = P.shape
    m = Q.shape[0]
    # distances in P (exclude self)
    rho = NearestNeighbors(n_neighbors=k+1).fit(P)\
             .kneighbors(P, return_distance=True)[0][:, -1]
    # distances from P to Q
    nu  = NearestNeighbors(n_neighbors=k).fit(Q)\
             .kneighbors(P, return_distance=True)[0][:, -1]
    rho = np.maximum(rho, eps)
    nu  = np.maximum(nu, eps)
    return d * np.mean(np.log(nu/rho)) + np.log(m/(n-1))


def kl_divergence_univariate(data_p, data_q, grid_points=1000, epsilon=1e-10):
    """
    Compute the KL divergence for univariate samples using KDE and numerical integration.
    """
    kde_p = gaussian_kde(data_p)
    kde_q = gaussian_kde(data_q)
    
    grid_min = min(np.min(data_p), np.min(data_q))
    grid_max = max(np.max(data_p), np.max(data_q))
    x = np.linspace(grid_min, grid_max, grid_points)
    
    p = kde_p.evaluate(x)
    q = kde_q.evaluate(x)
    
    # Ensure numerical stability: normalize and clip small values
    p /= np.trapz(p, x)
    q /= np.trapz(q, x)
    
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    
    dx = x[1] - x[0]
    return np.sum(p * np.log(p / q)) * dx

def kl_divergence_independent_features_1(data_p, data_q, grid_points=1000, epsilon=1e-10):
    """
    Compute the KL divergence between two sets of samples assuming independent features.
    """
    data_p = np.array(data_p)
    data_q = np.array(data_q)
    n_features = data_p.shape[1]
    
    total_kl = 0.0
    for i in range(n_features):
        p_feature = data_p[:, i]
        q_feature = data_q[:, i]
        kl_feature = kl_divergence_univariate(p_feature, q_feature, grid_points, epsilon)
        total_kl += kl_feature
    return total_kl

import numpy as np
from scipy.linalg import sqrtm


def wasserstein_distance_normal(deterministic_vals, mu_q, Sigma_q):
    """
    2-Wasserstein distance between:
      - P = point-mass at deterministic_vals (so Cov_P = 0)
      - Q = N(mu_q, Sigma_q)

    W_2(P, Q) = sqrt( ||mu_P - mu_Q||^2 + Tr(Sigma_Q) ).
    """
    mu_p = np.asarray(deterministic_vals)
    mu_q = np.asarray(mu_q)
    Sigma_q = np.asarray(Sigma_q)

    diff = mu_p - mu_q
    sq_term    = diff.dot(diff)            # ||mu_P - mu_Q||^2
    trace_term = np.trace(Sigma_q)         # Tr(Σ_Q)

    return np.sqrt(sq_term + trace_term)


def wasserstein_distance_deterministic(deterministic_vals, samples_q):
    """
    2-Wasserstein distance between:
      - P = point-mass at deterministic_vals (so Cov_P = 0)
      - Q ≈ MVN(mu_Q, Sigma_Q) estimated from samples_q

    W_2(P, Q) = sqrt( ||mu_P - mu_Q||^2 + Tr(Sigma_Q) ).
    """
    # ensure arrays
    mu_p = np.asarray(deterministic_vals)
    Xq = np.asarray(samples_q)
    
    # estimate Q's mean and covariance
    mu_q = Xq.mean(axis=0)
    # rowvar=False: each column is a variable, each row an observation
    Sigma_q = np.cov(Xq, rowvar=False)
    
    diff = mu_p - mu_q
    sq_term = diff.dot(diff)            # ||mu_P - mu_Q||^2
    trace_term = np.trace(Sigma_q)      # Tr(Sigma_Q)
    
    w2_squared = sq_term + trace_term
    return np.sqrt(w2_squared)