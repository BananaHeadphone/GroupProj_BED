import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import itertools
import arviz as az
import random
from tqdm import tqdm
#from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from pytensor import function
import logging
from functools import partial
import pytensor.tensor as tt   
from pytensor import shared
import EIG_computation as ec
from scipy.linalg import cho_factor, cho_solve

class BED():
    '''Model for Bayesian experimental design'''

    def __init__(self, A, num_dusts, sigma):
        """
        Initialize the Bayesian Experimental Design model.
        
        Parameters:
            A (numpy.ndarray): Matrix representing the relationship between engines and airports.
            num_dusts (int): Number of dust types (or components) being modeled.
            sigma (float): Standard deviation (noise level) used in the model.
        """
        self.A = A
        self.num_dusts = num_dusts
        self.sigma = sigma
        
        # Determine the number of engines and airports from the shape of A
        self.num_engines = A.shape[0]
        self.num_airports = A.shape[1]
        
        # Reduce verbosity of PyMC logging (suppress info/warnings from PyMC)
        logging.getLogger("pymc").setLevel(logging.WARNING)

    def normal_design(self, A, num_chosen_engines, prior_sigma):
        # Generate all possible combinations of engine indices of the chosen size.
        candidate_designs = list(itertools.combinations(range(A.shape[0]), num_chosen_engines))
        
        # Evaluate the EIG for each candidate design.
        # tqdm is used here to show a progress bar during computation.
        eig_values = [ec.normal_eig(design, A, self.sigma, self.num_dusts, prior_sigma)
                      for design in tqdm(candidate_designs, desc="Computing EIG for each design")]
        
        # Select the design with the maximum EIG.
        optimal_design = candidate_designs[np.argmax(eig_values)]
        print("Optimal engine subset for measurement:", optimal_design)
        return optimal_design, eig_values

    def initial_design(self, num_chosen_engines, N, M):
        """
        Determine the optimal subset of engines (design) by computing the EIG for all candidate designs.
        
        Parameters:
            num_chosen_engines (int): Number of engines to select.
            N (int): Number of outer Monte Carlo samples for EIG estimation.
            M (int): Number of inner Monte Carlo samples for EIG estimation.
        
        Returns:
            tuple: The optimal design (subset of engine indices) and a list of EIG values for each candidate.
        """
        # Generate all possible combinations of engine indices of the chosen size.
        candidate_designs = list(itertools.combinations(range(self.num_engines), num_chosen_engines))
        
        # Evaluate the EIG for each candidate design.
        # tqdm is used here to show a progress bar during computation.
        eig_values = [ec.initial_eig(design, self.A, self.sigma, self.num_dusts, N, M)
                      for design in tqdm(candidate_designs, desc="Computing EIG for each design")]
        
        # Select the design with the maximum EIG.
        optimal_design = candidate_designs[np.argmax(eig_values)]
        print("Optimal engine subset for measurement:", optimal_design)
        return optimal_design, eig_values


    def do_experiment_normal(self, optimal_design, measure, design_matrix=None):
        """
        Analytic posterior for the Gaussian‐linear model:
          x ~ N(mu0, Sigma0),   y | x ~ N(A_sub x, sigma^2 I).
    
        Always flattens in Fortran‐order (“column‐major”) so that
        kron(I_nd, A_sub) @ x_flat and the measurement vector y align.
        """
        # 1) dims and noise
        nd    = self.num_dusts
        na    = self.num_airports
        sigma = self.sigma   # here sigma is the *std* of the noise
    
        # 2) pick or use provided design_matrix
        if design_matrix is not None:
            A_sub = np.asarray(design_matrix)       # (n_sel, na)
        else:
            A_sub = self.A[optimal_design, :]       # (n_sel, na)
        n_sel = A_sub.shape[0]
    
        # 3) build the big design: shape (n_sel*nd, na*nd)
        A_big = np.kron(np.eye(nd), A_sub)
    
        # 4) prior precision Σ0⁻¹ for x ~ N(5·1, 4²·I)
        D = na * nd
        mu0        = 5.0 * np.ones(D)
        Sigma0_inv = np.eye(D) / (4.0 ** 2)
    
        # 5) noise precision = 1/σ²
        inv_var = 1.0 / (sigma ** 2)
    
        # 6) flatten y in Fortran order
        y = np.asarray(measure)
        if y.ndim == 2:
            # from shape (n_sel, nd) → vector length n_sel*nd
            y = y.ravel(order="F")
        elif y.ndim != 1:
            raise ValueError(f"measure must be 1D or 2D, got {y.shape!r}")
    
        # 7) posterior precision K and RHS b
        #    K = Σ0⁻¹ + (1/σ²) A_bigᵀ A_big
        #    b = Σ0⁻¹ μ0 + (1/σ²) A_bigᵀ y
        K = Sigma0_inv + inv_var * (A_big.T @ A_big)
        b = Sigma0_inv @ mu0 + inv_var * (A_big.T @ y)
    
        # 8) Cholesky‐solve for μ_post and Σ_post = K⁻¹
        cho, lower = cho_factor(K, check_finite=False)
        mu_post    = cho_solve((cho, lower), b,            check_finite=False)
        Sigma_post = cho_solve((cho, lower), np.eye(D),   check_finite=False)
    
        return mu_post, Sigma_post



    
    def do_experiment(self, optimal_design, measure, N, posteriordata=None):
        """
        Perform an experiment based on the optimal design and observed measurements.
        Uses a differentiable parametric model (Gaussian mixture) for the prior if posteriordata is provided.
        """
        # Retrieve relevant dimensions and parameters.
        num_selected_engines = len(optimal_design)
        num_dusts = self.num_dusts
        num_airports = self.num_airports
        sigma = self.sigma
        A = self.A
    
        # Initialize the PyMC model.
        model = pm.Model()
    
        # Create a covariance matrix for the measurements.
        sigma_designed = sigma * np.ones(num_selected_engines * num_dusts)
        sigma_designed = np.diag(sigma_designed)
    
        # Subset the design matrix A to include only the selected engines.
        A_designed = A[optimal_design, :]
    
        with model:
            if posteriordata is None:
                # Use a Uniform prior if no posterior data is available.
                x = pm.Normal("x", mu = 5, sigma = 4, shape=(num_airports, num_dusts))
                #x = pm.Uniform("x", lower = 0, upper = 10, shape=(num_airports, num_dusts))
            elif isinstance(posteriordata, np.ndarray):
                x = pm.Normal("x", mu = 5, sigma = 4, shape=(num_airports, num_dusts))
                num_selected_engines = posteriordata.shape[0]
                sigma_designed = sigma * np.ones(num_selected_engines * num_dusts)
                sigma_designed = np.diag(sigma_designed)
            else:
                posteriordata = posteriordata.posterior["x"].values
                posteriordata = posteriordata.reshape(posteriordata.shape[0] * posteriordata.shape[1],
                                                      posteriordata.shape[2] * posteriordata.shape[3])
                posteriordata = posteriordata.T
                # Choose number of mixture components (adjust as needed)
                n_components = 12
                # Note: posteriordata should be a 2D array where each column is a sample.
                gmm = GaussianMixture(n_components=n_components, covariance_type='diag')
                # Fit the GMM; transpose if necessary so each row is an observation.
                gmm.fit(posteriordata.T)
    
                # Get dimensionality (expected to match flattened x shape)
                d = posteriordata.shape[0]
    
                # Convert GMM parameters to Theano shared variables
                gmm_weights = shared(gmm.weights_, name="gmm_weights")
                gmm_means = shared(gmm.means_, name="gmm_means")
                gmm_covs = shared(gmm.covariances_, name="gmm_covs")
    
                def gmm_logp(x_flat):
                    log_probs = []
                    for i in range(n_components):
                        diff = x_flat - gmm_means[i]             # (d,)
                        diff_col = diff[:, None]                 # (d, 1)
                        # Convert 1D variance vector to a diagonal matrix
                        cov_matrix = tt.diag(gmm_covs[i])
                        inv_cov = tt.nlinalg.matrix_inverse(cov_matrix)
                        quad = tt.dot(diff_col.T, tt.dot(inv_cov, diff_col))[0, 0]
                        # For diagonal matrices, the log determinant is sum(log(variances))
                        logdet = tt.sum(tt.log(gmm_covs[i]))
                        log_prob_i = tt.log(gmm_weights[i]) - 0.5 * (quad + logdet + d * np.log(2 * np.pi))
                        log_probs.append(log_prob_i)
                    log_probs_stacked = tt.stack(log_probs)
                    # For numerical stability, use the logsumexp trick:
                    max_log = tt.max(log_probs_stacked)
                    return max_log + tt.log(tt.sum(tt.exp(log_probs_stacked - max_log)))
                
                # Define the prior for x.
                x = pm.Normal("x", mu = 5, sigma = 4, shape=(num_airports, num_dusts))
                # Add the GMM-based potential. Flatten x to a vector.
                pm.Potential("gmm_logp", gmm_logp(x.flatten()))
                        
            
            # Define the likelihood using a multivariate normal:
            if isinstance(posteriordata, np.ndarray):
                y = pm.MvNormal("y",
                            mu=(posteriordata @ x).flatten(),
                            cov=sigma_designed,
                            shape=num_selected_engines * num_dusts,
                            observed=measure)
            else:
                y = pm.MvNormal("y",
                            mu=(A_designed @ x).flatten(),
                            cov=sigma_designed,
                            shape=num_selected_engines * num_dusts,
                            observed=measure)
        with model:
            # Use a gradient-based sampler since our model is now differentiable.
            posterior_samples = pm.sample(N, target_accept=0.85)
    
        return posterior_samples


        
    def reuse_posterior_samples(self, data, N, M):
        """
        Reuse posterior samples from previous inference by selecting N samples for outer estimation 
        and M samples for inner estimation.
        
        Parameters:
            data: The trace or inference data containing the posterior samples for 'x'.
            N (int): Number of outer samples to extract.
            M (int): Number of inner samples to extract.
        
        Returns:
            xn_samples: Selected outer samples (for 'x').
            xm_samples: Selected inner samples (for 'x').
        """
        num_dusts = self.num_dusts
        num_airports = self.num_airports
    
        # Extract posterior samples for 'x' and reshape them to (total_samples, num_airports, num_dusts)
        posterior_samples = data.posterior["x"].values 
        # Determine total number of samples available in the posterior
        NN = np.shape(posterior_samples[0, :, 0, 0])[0]
        assert(NN >= M + N), "Not enough posterior samples to extract both N and M samples"
        # Create an array of indices for all available samples
        indx = np.arange(NN)
        # Select the last (M+N) indices to ensure samples are from the most recent draws
        indx = indx[-(M + N):]   

        # Shuffle the indices to randomize the selection
        np.random.shuffle(indx)
        # Select N samples for xn_samples (outer samples) and M samples for xm_samples (inner samples)
        xn_samples = posterior_samples[:,indx[:N],:,:].reshape(-1, num_airports, num_dusts)      
        xm_samples = posterior_samples[:,indx[-M:],:,:].reshape(-1, num_airports, num_dusts)
        
        return xn_samples, xm_samples
    
    
    def compute_mi(self, xn_samples, xm_samples):
        """
        Compute the mutual information between corresponding components of two sets of samples.
        
        This function iterates over each dust type and airport, extracts the samples, and computes
        the mutual information using a regression-based estimator.
        
        Parameters:
            xn_samples (numpy.ndarray): Array of outer samples with shape (N, num_airports, num_dusts).
            xm_samples (numpy.ndarray): Array of inner samples with shape (M, num_airports, num_dusts).
        
        Returns:
            mi (float): The accumulated mutual information across all airports and dust types.
        """
        num_dusts = self.num_dusts
        num_airports = self.num_airports
        mi = 0
        # Loop over each dust type
        for j in range(num_dusts):
            # Loop over each airport
            for i in range(num_airports):  
                # Extract samples for the current airport and dust type from both sample sets
                xn_single = xn_samples[:, i, j]
                xm_single = xm_samples[:, i, j]
                # Compute mutual information for these samples (reshaped to 2D for regression)
                mi += mutual_info_regression(xn_single.reshape(-1, 1), xm_single, random_state=42)
        return mi  # Return the accumulated mutual information value
    
    


    
    
    def sequential_design(self, num_chosen_engines, arg1, arg2=None, M=3000, N=3000):
        """
        Determine the optimal sequential design by computing the EIG for each candidate subset 
        of engines. The method allows for either reusing provided posterior samples or generating 
        new samples via KDE.
        
        Parameters:
            num_chosen_engines (int): Number of engines to select in the design.
            arg1: Either reused posterior samples for 'x' (if arg2 is provided) or data for KDE estimation.
            arg2: (Optional) If provided, represents the second set of reused posterior samples.
            M (int): Number of inner Monte Carlo samples.
            N (int): Number of outer Monte Carlo samples.
        
        Returns:
            tuple: The optimal engine design (subset of indices) and a list of EIG values for each candidate.
        """
        # If arg2 is provided, assume arg1 and arg2 are the reused posterior samples (xn and xm respectively)
        if arg2 is not None:
            xn_samples = arg1
            xm_samples = arg2
        
        # Generate all possible candidate designs (combinations of engine indices)
        candidate_designs = list(itertools.combinations(range(self.num_engines), num_chosen_engines))
        
        # Compute EIG for each candidate design
        if arg2 is not None:
            # Use the sequential EIG function that reuses samples
            eig_values = [ec.sequential_eig_reusesamples(design, self.A, self.sigma, self.num_dusts, xn_samples, xm_samples)
                          for design in tqdm(candidate_designs, desc="Computing EIG for each design")]
        else:
            posteriordata = arg1
            posteriordata = posteriordata.posterior["x"].values
            posteriordata = posteriordata.reshape(posteriordata.shape[0] * posteriordata.shape[1],
                                                  posteriordata.shape[2] * posteriordata.shape[3])
            posteriordata = posteriordata.T
            # If reused samples are not provided, create a KDE estimator from arg1
            kde_estimator = gaussian_kde(posteriordata)
            # Use the sequential EIG function that employs KDE-based sampling
            eig_values = [ec.sequential_eig_kde(design, self.A, self.sigma, self.num_dusts, kde_estimator, M, N)
                          for design in tqdm(candidate_designs, desc="Computing EIG for each design")]
        
        # Identify the candidate design with the maximum EIG
        optimal_design = candidate_designs[np.argmax(eig_values)]
        print("Optimal engine subset for measurement:", optimal_design)
        return optimal_design, eig_values

