import pymc as pm
import numpy as np
import random
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from pytensor import function
import logging
from functools import partial
import pytensor.tensor as tt   
from pytensor import shared
from scipy.linalg import cho_factor, cho_solve

import numpy as np

def normal_eig(
    candidate_design,  # a list or 1D array of engine‐indices
    A,                  # (N_engines × N_airports) design matrix
    sigma_noise,        # noise *std*
    num_dusts,          # number of dust types
    prior_sigma_or_cov  # scalar, 1D array of stds, or full covariance matrix
):
    """
    Compute the Expected Information Gain (mutual information) for the
    single candidate_design, allowing a general prior covariance Σ₀.
    """
    N_air = A.shape[1]
    N_d   = num_dusts
    D     = N_air * N_d

    # 1) Build full prior covariance Σ₀
    pc = prior_sigma_or_cov
    if np.isscalar(pc):
        # isotropic σ → Σ₀ = σ² I
        Sigma0 = np.eye(D) * (pc**2)
    else:
        pc = np.asarray(pc)
        if pc.ndim == 1:
            # vector of per-component stds → Σ₀ = diag(pc**2)
            if pc.shape[0] != D:
                raise ValueError(f"Expected prior std vector of length {D}, got {pc.shape[0]}")
            Sigma0 = np.diag(pc**2)
        elif pc.ndim == 2:
            # full covariance
            if pc.shape != (D, D):
                raise ValueError(f"Expected prior cov shape {(D,D)}, got {pc.shape}")
            Sigma0 = pc
        else:
            raise ValueError("prior_sigma_or_cov must be scalar, 1D array, or 2D matrix")

    # 2) prior precision & log‐det Σ₀
    Sigma0_inv = np.linalg.inv(Sigma0)
    sign, logdet0 = np.linalg.slogdet(Sigma0)
    if sign <= 0:
        raise np.linalg.LinAlgError("Prior covariance must be positive-definite")

    # 3) noise precision 1/σ_noise²
    inv_noise = 1.0 / (sigma_noise**2)

    # 4) build A_big = kron(I_Nd, A_sub)
    A_sub = A[np.array(candidate_design), :]    # (n_sel, N_air)
    A_big = np.kron(A_sub, np.eye(num_dusts))   # (n_sel*N_d, D)

    # 5) posterior precision K_post = Σ₀⁻¹ + (1/σ²) AᵀA
    K_post = Sigma0_inv + inv_noise * (A_big.T @ A_big)

    # 6) Cholesky for log‐det K_post
    L = np.linalg.cholesky(K_post)
    logdet_K = 2.0 * np.sum(np.log(np.diag(L)))

    # 7) EIG = 0.5 * (log det Σ₀ + log det K_post)
    eig = 0.5 * (logdet0 + logdet_K)
    return eig


def initial_eig(
    selected_engines,
    A,
    sigma,
    num_dusts,
    N=3000,
    M=3000,
):
    """
    Compute Expected Information Gain (EIG) for an initial design using a nested Monte Carlo estimator.
    
    Parameters:
        selected_engines (list or tuple): Indices of the chosen engines for the design.
        A (numpy.ndarray): The design matrix.
        sigma (float): Noise standard deviation.
        num_dusts (int): Number of dust types.
        N (int): Number of outer Monte Carlo samples.
        M (int): Number of inner Monte Carlo samples.
    
    Returns:
        numpy.ndarray: The computed EIG for each measurement corresponding to the design.
    """
    # Determine the number of selected engines
    num_selected_engines = len(selected_engines)

    # Build the covariance matrix for the likelihood.
    # This matrix is diagonal with each entry equal to sigma.
    sigma_designed = np.diag([sigma] * (num_selected_engines * num_dusts))

    # Extract the rows from A corresponding to the chosen engines.
    A_designed = A[selected_engines, :]

    # ---------------------------------------------------------
    # Outer Monte Carlo sampling: sample from the prior distribution
    # over the dust proportions at the airports.
    # ---------------------------------------------------------
    with pm.Model() as outer_model:
        # Define a uniform prior for dust proportions for each airport and each dust type.
        xn = pm.Normal("xn", mu = 5, sigma = 4, shape=(A.shape[1], num_dusts))
        #xn = pm.Uniform("xn", lower = 0, upper = 10, shape=(A.shape[1], num_dusts)) 
        # (An alternative using a Dirichlet prior is commented out.)

        # Define the likelihood as a multivariate normal.
        # The expected mean is given by the matrix product A_designed @ xn.
        y = pm.MvNormal(
            "y",
            mu=(A_designed @ xn).flatten(),
            cov=sigma_designed,
            shape=num_selected_engines * num_dusts
        )
        # Generate N samples from the prior predictive distribution.
        outer_samples = pm.sample_prior_predictive(samples=N, return_inferencedata=False)

    # Extract the outer samples for the dust proportions and the corresponding observations.
    xn_samples = outer_samples["xn"]  # shape: (N, num_airports, num_dusts)
    y_samples  = outer_samples["y"]   # shape: (N, num_selected_engines * num_dusts)

    # ---------------------------------------------------------
    # Inner Monte Carlo sampling: sample alternative dust proportions.
    # ---------------------------------------------------------
    with pm.Model() as inner_model:
        # Define an independent uniform prior for inner sampling.
        xm = pm.Normal("xm", mu = 5, sigma = 4, shape=(A.shape[1], num_dusts))
        #xm = pm.Uniform("xm", lower = 0, upper = 10, shape=(A.shape[1], num_dusts)) 
        # Generate M samples from the inner prior.
        inner_samples = pm.sample_prior_predictive(samples=M, return_inferencedata=False)

    # Extract the inner samples.
    xm_samples = inner_samples["xm"]  # shape: (M, num_airports, num_dusts)

    # ---------------------------------------------------------
    # Compute the Expected Information Gain (EIG)
    # ---------------------------------------------------------

    # Invert the covariance matrix for later likelihood computation.
    sigma_designed = np.linalg.inv(sigma_designed)

    # Compute the model mean for the inner samples using Einstein summation.
    # This computes A_designed @ xm for each inner sample.
    mu_m = np.einsum('ij,ljk->lik', A_designed, xm_samples)
    # Reshape mu_m to have dimensions (M, num_selected_engines*num_dusts)
    mu_m = mu_m.reshape(-1, num_selected_engines * num_dusts)

    # For each outer sample, compute the difference between observed y_samples and the predicted means from inner samples.
    diff_m = (y_samples[:, None, :] - mu_m[None, :, :]) @ sigma_designed

    # Compute the likelihood for inner samples.
    # The likelihood is evaluated as: exp(-0.5 * (y-mu)' Sigma (y-mu))
    likelihood_m = np.exp(-0.5 * diff_m[:, :, None, :] @ (y_samples[:, None, :] - mu_m[None, :, :])[:, :, :, None])
    # Average the likelihood over the inner samples.
    likelihood_m = np.mean(likelihood_m, axis=1)
      
    # Compute the model mean for the outer samples.
    mu_n = np.einsum('ij,ljk->lik', A_designed, xn_samples)
    mu_n = mu_n.reshape(-1, num_selected_engines * num_dusts)

    # Compute the difference and likelihood for the outer samples.
    diff_n = (y_samples - mu_n) @ sigma_designed
    likelihood_n = np.exp(-0.5 * diff_n[:, None, :] @ (y_samples - mu_n)[:, :, None])
    # Compute the log-ratio between the outer and inner likelihoods, using a small constant (1e-16) to avoid division by zero.
    likelihood_n = np.mean(np.log((1e-16 + likelihood_n) / (1e-16 + likelihood_m)), axis=0)

    # The EIG is given by the averaged log likelihood ratio.
    eig = likelihood_n
    return eig


def sequential_eig_reusesamples(selected_engines, A, sigma, num_dusts, xn_samples, xm_samples):
    """
    Compute Expected Information Gain (EIG) for the sequential design using reused samples.
    
    Parameters:
        selected_engines (list or tuple): Indices of the engines selected for measurement.
        A (numpy.ndarray): The design matrix.
        sigma (float): Noise standard deviation.
        num_dusts (int): Number of dust types.
        xn_samples (numpy.ndarray): Outer samples for the dust proportions.
        xm_samples (numpy.ndarray): Inner samples for the dust proportions.
    
    Returns:
        eig (numpy.ndarray): The computed expected information gain for each measurement.
    """
    num_selected_engines = len(selected_engines)
    
    # Build the covariance matrix for the measurements (diagonal with sigma entries)
    sigma_designed = sigma * np.ones(num_selected_engines * num_dusts)
    sigma_designed = np.diag(sigma_designed)
    
    # Subset the design matrix A for the selected engines
    A_designed = A[selected_engines, :]
    
    # Generate simulated measurement samples (y_samples) for each outer sample xn
    num_xn = np.size(xn_samples, axis=0)
    y_samples = np.zeros([num_xn, num_selected_engines * num_dusts])
    for i in range(num_xn):
        xn = xn_samples[i]
        # Draw one measurement sample from a multivariate normal distribution with mean (A_designed @ xn)
        y_samples[i] = np.random.multivariate_normal((A_designed @ xn).flatten(), sigma_designed, size=1)
    
    # Invert the covariance matrix for later likelihood computations
    sigma_designed = np.linalg.inv(sigma_designed)
    
    # Compute model predictions for inner samples using Einstein summation
    mu_m = np.einsum('ij,ljk->lik', A_designed, xm_samples)
    mu_m = mu_m.reshape(-1, num_selected_engines * num_dusts)
    
    # Compute the difference between simulated measurements and inner predictions
    diff_m = (y_samples[:, None, :] - mu_m[None, :, :]) @ sigma_designed
    # Calculate likelihoods for inner samples using the Gaussian likelihood formula
    likelihood_m = np.exp(-0.5 * diff_m[:, :, None, :] @ (y_samples[:, None, :] - mu_m[None, :, :])[:, :, :, None])
    likelihood_m = np.mean(likelihood_m, axis=1)
    
    # Compute model predictions for outer samples
    mu_n = np.einsum('ij,ljk->lik', A_designed, xn_samples)
    mu_n = mu_n.reshape(-1, num_selected_engines * num_dusts)
    
    # Compute the difference for outer samples and calculate the corresponding likelihoods
    diff_n = (y_samples - mu_n) @ sigma_designed
    likelihood_n = np.exp(-0.5 * diff_n[:, None, :] @ (y_samples - mu_n)[:, :, None])
    # Compute the expected log-ratio (EIG) by averaging over the samples, adding a small constant for numerical stability
    likelihood_n = np.mean(np.log((1e-16 + likelihood_n) / (1e-16 + likelihood_m)), axis=0)
    
    eig = likelihood_n
    return eig



def logp_x(value, kde_estimator):
    """
    Compute the log probability density of 'value' using a given KDE estimator.
    
    Parameters:
        value (array-like): The value(s) at which to evaluate the density.
        kde_estimator: A kernel density estimator object with a callable interface.
    
    Returns:
        Tensor: The log density evaluated at the given value, with a small constant added to avoid log(0).
    """
    # Ensure the input is a NumPy array
    val = np.array(value)
    # Evaluate the density using the KDE estimator
    density = kde_estimator(val)
    # Return the log density, adding a small constant for numerical stability
    return tt.as_tensor_variable(np.log(density + 1e-10))



def random_kde(kde_estimator, rng=None, size=None):
    """
    Generate a random sample from the KDE estimator.
    
    Parameters:
        kde_estimator: A kernel density estimator object with a resample method.
        rng: (Optional) Random number generator instance.
        size (int or tuple): Desired output sample shape.
    
    Returns:
        numpy.ndarray: Randomly sampled data from the KDE with the specified size.
    
    Raises:
        ValueError: If the KDE sample size is smaller than the desired number of elements.
    """
    # If size is None, return a single draw (PyMC requires a single sample)
    if size is None:
        size = ()
    # Ensure size is a tuple
    if not isinstance(size, tuple):
        size = (size,)
    # Calculate the total number of desired elements
    desired = np.prod(size, dtype=int)
    # Draw a sample from the KDE (the resample returns an array of shape (dim, num_samples))
    raw = kde_estimator.resample(1).T[0]
    # If the drawn sample has more elements than desired, slice it down to the desired size
    if raw.size > desired:
        raw = raw[:desired]
    elif raw.size < desired:
        raise ValueError(f"KDE sample size {raw.size} is smaller than desired {desired}.")
    return raw


def sequential_eig_kde(selected_engines, A, sigma, num_dusts, kde_estimator, N=5000, M=5000):
    """
    Compute Expected Information Gain (EIG) for the sequential design using a KDE-based 
    prior representation and a Nested Monte Carlo estimator.
    
    Parameters:
        selected_engines (list or tuple): Indices of the engines selected for measurement.
        A (numpy.ndarray): The design matrix.
        sigma (float): Noise standard deviation.
        num_dusts (int): Number of dust types.
        kde_estimator: Kernel density estimator for constructing the prior.
        N (int): Number of outer Monte Carlo samples.
        M (int): Number of inner Monte Carlo samples.
    
    Returns:
        eig (numpy.ndarray): The computed EIG for each measurement.
    """
    num_selected_engines = len(selected_engines)
    
    # Build covariance matrix for measurements: diagonal with entries equal to sigma
    sigma_designed = np.diag([sigma] * (num_selected_engines * num_dusts))
    
    # Subset the design matrix A for the selected engines
    A_designed = A[selected_engines, :]
    
    # Create partial functions that incorporate the KDE estimator for log probability and random sampling
    logp_with_kde = partial(logp_x, kde_estimator=kde_estimator)
    random_with_kde = partial(random_kde, kde_estimator=kde_estimator)
    
    # ---------------------------------------------------------
    # Outer Monte Carlo sampling using a custom DensityDist based on KDE
    # ---------------------------------------------------------
    with pm.Model() as outer_model:
        # Define a custom density distribution for the dust proportions using the KDE
        xn = pm.DensityDist("xn", logp=logp_with_kde, random=random_with_kde, shape=(A.shape[1] * num_dusts))
        # Reshape the flat vector into a (num_airports, num_dusts) matrix
        xn = xn.reshape((A.shape[1], num_dusts))
        # Define the likelihood using a multivariate normal distribution
        y = pm.MvNormal(
            "y",
            mu=(A_designed @ xn).flatten(),
            cov=sigma_designed,
            shape=num_selected_engines * num_dusts
        )
        # Draw N outer samples from the prior predictive distribution
        outer_samples = pm.sample_prior_predictive(samples=N, return_inferencedata=False)
    
    xn_samples = outer_samples["xn"]  # Shape: (N, num_airports, num_dusts)
    y_samples  = outer_samples["y"]   # Shape: (N, num_selected_engines * num_dusts)
    
    # ---------------------------------------------------------
    # Inner Monte Carlo sampling using the same KDE-based DensityDist
    # ---------------------------------------------------------
    with pm.Model() as inner_model: 
        xm = pm.DensityDist("xm", logp=logp_with_kde, random=random_with_kde, shape=(A.shape[1] * num_dusts))
        xm = xm.reshape((A.shape[1], num_dusts))
        inner_samples = pm.sample_prior_predictive(samples=M, return_inferencedata=False)
    
    xm_samples = inner_samples["xm"]  # Shape: (M, num_airports, num_dusts)
    
    # ---------------------------------------------------------
    # Compute EIG using the nested Monte Carlo estimator
    # ---------------------------------------------------------
    # Reshape inner samples if needed
    xm_samples = xm_samples.reshape(-1, A.shape[1], num_dusts)
    # Invert the covariance matrix for the Gaussian likelihood computation
    sigma_designed = np.linalg.inv(sigma_designed)
    
    # Compute predictions for inner samples (mu_m)
    mu_m = np.einsum('ij,ljk->lik', A_designed, xm_samples)
    mu_m = mu_m.reshape(-1, num_selected_engines * num_dusts)
    
    # Compute differences between outer measurements and inner predictions
    diff_m = (y_samples[:, None, :] - mu_m[None, :, :]) @ sigma_designed
    # Compute likelihood for inner samples and average over inner samples
    likelihood_m = np.exp(-0.5 * diff_m[:, :, None, :] @ (y_samples[:, None, :] - mu_m[None, :, :])[:, :, :, None])
    likelihood_m = np.mean(likelihood_m, axis=1)
    
    # Reshape outer samples appropriately
    xn_samples = xn_samples.reshape(-1, A.shape[1], num_dusts)
    # Compute predictions for outer samples (mu_n)
    mu_n = np.einsum('ij,ljk->lik', A_designed, xn_samples)
    mu_n = mu_n.reshape(-1, num_selected_engines * num_dusts)
    
    # Compute differences and likelihood for outer samples
    diff_n = (y_samples - mu_n) @ sigma_designed
    likelihood_n = np.exp(-0.5 * diff_n[:, None, :] @ (y_samples - mu_n)[:, :, None])
    # Compute the expected log-ratio (EIG) by averaging over samples
    likelihood_n = np.mean(np.log((1e-16 + likelihood_n) / (1e-16 + likelihood_m)), axis=0)
    
    eig = likelihood_n
    return eig

def sequential_eig_gmm(selected_engines, A, sigma, num_dusts, posteriordata, N=5000, M=5000):
    """
    Compute Expected Information Gain (EIG) for the sequential design using a KDE-based 
    prior representation and a Nested Monte Carlo estimator.
    
    Parameters:
        selected_engines (list or tuple): Indices of the engines selected for measurement.
        A (numpy.ndarray): The design matrix.
        sigma (float): Noise standard deviation.
        num_dusts (int): Number of dust types.
        kde_estimator: Kernel density estimator for constructing the prior.
        N (int): Number of outer Monte Carlo samples.
        M (int): Number of inner Monte Carlo samples.
    
    Returns:
        eig (numpy.ndarray): The computed EIG for each measurement.
    """
    num_selected_engines = len(selected_engines)
    num_airports = A.shape[1]
    # Build covariance matrix for measurements: diagonal with entries equal to sigma
    sigma_designed = np.diag([sigma] * (num_selected_engines * num_dusts))
    
    # Subset the design matrix A for the selected engines
    A_designed = A[selected_engines, :]
    
    
    # Choose number of mixture components (adjust as needed)
    n_components = 3
    # Note: posteriordata should be a 2D array where each column is a sample.
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    # Fit the GMM; transpose if necessary so each row is an observation.
    gmm.fit(posteriordata.T)

    # Get dimensionality (expected to match flattened x shape)
    d = posteriordata.shape[0]

    # Convert GMM parameters to Theano shared variables
    gmm_weights = shared(gmm.weights_, name="gmm_weights")
    gmm_means = shared(gmm.means_, name="gmm_means")
    gmm_covs = shared(gmm.covariances_, name="gmm_covs")

    # Define a differentiable log-density function for the GMM
    def gmm_logp(x_flat):
        log_probs = []
        for i in range(n_components):
            diff = x_flat - gmm_means[i]             # (d,)
            diff_col = diff[:, None]                 # (d, 1)
            inv_cov = tt.nlinalg.matrix_inverse(gmm_covs[i])
            quad = tt.dot(diff_col.T, tt.dot(inv_cov, diff_col))[0, 0]
            # Compute log determinant using numpy (as a constant)
            sign, logdet = np.linalg.slogdet(gmm_covs.get_value()[i])
            log_prob_i = tt.log(gmm_weights[i]) - 0.5 * (quad + logdet + d * np.log(2 * np.pi))
            log_probs.append(log_prob_i)
        log_probs_stacked = tt.stack(log_probs)
        return tt.log(tt.sum(tt.exp(log_probs_stacked)))
    
    # ---------------------------------------------------------
    # Outer Monte Carlo sampling using a custom DensityDist based on KDE
    # ---------------------------------------------------------
    with pm.Model() as outer_model:
        # Define the prior for x.
        xn = pm.Uniform("xn", lower=0, upper=5, shape=(num_airports, num_dusts))
        # Add the GMM-based potential. Flatten x to a vector.
        pm.Potential("gmm_logp", gmm_logp(xn.flatten()))
        
        # Define the likelihood using a multivariate normal distribution
        y = pm.MvNormal(
            "y",
            mu=(A_designed @ xn).flatten(),
            cov=sigma_designed,
            shape=num_selected_engines * num_dusts
        )
        # Draw N outer samples from the prior predictive distribution
        outer_samples = pm.sample(N, target_accept=0.9)
    
    xn_samples = outer_samples.posterior["xn"].values  # Shape: (N, num_airports, num_dusts)
    y_samples  = outer_samples.posterior["y"].values   # Shape: (N, num_selected_engines * num_dusts)
    
    # ---------------------------------------------------------
    # Inner Monte Carlo sampling using the same KDE-based DensityDist
    # ---------------------------------------------------------
    with pm.Model() as inner_model: 
        # Define the prior for x.
        xm = pm.Uniform("xm", lower=0, upper=5, shape=(num_airports, num_dusts))
        # Add the GMM-based potential. Flatten x to a vector.
        pm.Potential("gmm_logp", gmm_logp(xm.flatten()))
        xm = xm.reshape((A.shape[1], num_dusts))
        inner_samples = pm.sample(M, target_accept=0.9)
    
    xm_samples = inner_samples.posterior["xm"].values  # Shape: (M, num_airports, num_dusts)
    
    # ---------------------------------------------------------
    # Compute EIG using the nested Monte Carlo estimator
    # ---------------------------------------------------------
    # Reshape inner samples if needed
    xm_samples = xm_samples.reshape(-1, A.shape[1], num_dusts)
    # Invert the covariance matrix for the Gaussian likelihood computation
    sigma_designed = np.linalg.inv(sigma_designed)
    
    # Compute predictions for inner samples (mu_m)
    mu_m = np.einsum('ij,ljk->lik', A_designed, xm_samples)
    mu_m = mu_m.reshape(-1, num_selected_engines * num_dusts)
    
    # Compute differences between outer measurements and inner predictions
    diff_m = (y_samples[:, None, :] - mu_m[None, :, :]) @ sigma_designed
    # Compute likelihood for inner samples and average over inner samples
    likelihood_m = np.exp(-0.5 * diff_m[:, :, None, :] @ (y_samples[:, None, :] - mu_m[None, :, :])[:, :, :, None])
    likelihood_m = np.mean(likelihood_m, axis=1)
    
    # Reshape outer samples appropriately
    xn_samples = xn_samples.reshape(-1, A.shape[1], num_dusts)
    # Compute predictions for outer samples (mu_n)
    mu_n = np.einsum('ij,ljk->lik', A_designed, xn_samples)
    mu_n = mu_n.reshape(-1, num_selected_engines * num_dusts)
    
    # Compute differences and likelihood for outer samples
    diff_n = (y_samples - mu_n) @ sigma_designed
    likelihood_n = np.exp(-0.5 * diff_n[:, None, :] @ (y_samples - mu_n)[:, :, None])
    # Compute the expected log-ratio (EIG) by averaging over samples
    likelihood_n = np.mean(np.log((1e-16 + likelihood_n) / (1e-16 + likelihood_m)), axis=0)
    
    eig = likelihood_n
    return eig