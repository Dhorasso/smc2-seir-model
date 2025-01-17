########################################################################
# This file content the codes for the Particle Marginal Metropolis-Hasting (PMMH) kernel
#  and the log prior distribution
############################################################################


import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import uniform, norm, truncnorm, lognorm, gamma, invgamma


def PMMH_kernel(model, Z_current, current_theta_particles, state_history, theta_names,
               observed_data, state_names, initial_theta_info, num_state_particles,
               theta_mean_current, theta_covariance_current, observation_distribution, m, t, tw, pmmh_moves, c):
    """
    Perform Particle Marginal Metropolis-Hastings (PMMH) for a given model.

    Parameters:
    model: The model to be used.
    Z_current (float): Current value of the marginal likelihood.
    current_theta_particles (ndarray): Current particles for the theta parameters.
    state_history (list): The history of the states over time.
    theta_names (list): Names of the theta parameters.
    observed_data (ndarray): The observed data.
    state_names (list): Names of the state variables.
    initial_theta_info (dict): Information about the initial theta parameters.
    num_state_particles (int): Number of state particles for the particle filter.
    theta_mean_current (ndarray): Current mean of the theta parameters.
    theta_covariance_current (ndarray): Current covariance of the theta parameters.
    observation_distribution (str): The type of observation distribution.
    m (int): Index of the particle.
    t (int): Current time step.
    tw (int): Window size for the (O-SMC^2).
    pmmh_moves (int): Number of PMMH move in the rejuvenation step.
    c (int): scaling factor for the covariance matrix in the PMMH kernel.

    Returns:
    dict: A dictionary containing the updated marginal likelihood ('Z'), 
          state, theta, and the acceptance rate ('acc').
    """

    acc = 0
    I = 1e-5 * np.eye(theta_covariance_current.shape[0])  # Identity matrix for regularization

    # Regularization to ensure positive-definite covariance
    theta_covariance_current = c * theta_covariance_current + I

    state_current = state_history[t][m]
    state_current_t_k = state_history[max(0, t - tw)][m]
    theta_current = current_theta_particles[m]

    # Precompute log prior for the current theta
    log_prior_current = log_prior(initial_theta_info, theta_current)

    for i in range(pmmh_moves):
        # Sample theta proposal from multivariate normal
        theta_proposal = np.random.multivariate_normal(theta_mean_current, theta_covariance_current)
        log_prior_proposal = log_prior(initial_theta_info, theta_proposal)

        if log_prior_proposal != -np.inf:
            # Evaluate the current and proposal log-posterior
            current = Z_current + log_prior_current 
            
            # Run particle filter to evaluate proposal likelihood
            untrans_theta_proposal = untransform_theta(theta_proposal, initial_theta_info)
            PF_results = Particle_Filter(model, state_names, state_current_t_k, untrans_theta_proposal, 
                                         theta_names, observed_data, num_state_particles, resampling_method='stratified', 
                                         observation_distribution=observation_distribution)
            
            Z_proposal = PF_results['margLogLike']
            state_proposal = PF_results['particle_state']
            proposal = Z_proposal + log_prior_proposal
            
            # Log-posterior comparison with multivariate normal log-density
            proposal += log_multivariate_normal_pdf(theta_current, theta_mean_current, theta_covariance_current)
            current += log_multivariate_normal_pdf(theta_proposal, theta_mean_current, theta_covariance_current)
            
            # Compute the acceptance ratio
            ratio = proposal - current
            r = np.exp(ratio)

            # Ensure the acceptance probability is real
            if np.isreal(r):
                if np.random.uniform() < min(1, r):
                    Z_current = Z_proposal
                    state_current = state_proposal
                    theta_current = theta_proposal
                    log_prior_current = log_prior_proposal
                    acc += 1

    # Return results, including the acceptance rate
    return {
        'Z': Z_current,
        'state': state_current,
        'theta': theta_current, 
        'acc': acc / pmmh_moves
    }



##################################################################
#  log of a multivariate normal distribution
# without the normalizing constant
def log_multivariate_normal_pdf(x, mean, cov):
    """
    Compute the log of the multivariate normal probability density function.

    Parameters:
    x (ndarray): The point at which the log-pdf is evaluated.
    mean (ndarray): The mean of the distribution.
    cov (ndarray): The covariance matrix of the distribution.

    Returns:
    float: The log of the multivariate normal pdf evaluated at x without the
    of constant of proportionality.
    """
    diff = x - mean
    cov_inv = np.linalg.inv(cov)  # Inverse of covariance matrix    
    return -0.5 *np.dot(diff.T, np.dot(cov_inv, diff))


##################################################################
############ Log Prior plus Jacobian adjustment ###################

def log_prior(initial_theta_info, theta):
    """
    Compute the log of the prior distribution for the given parameters, with Jacobian adjustments.

    This function calculates the log prior for each parameter based on its specified distribution and transformation.
    If a parameter is transformed (e.g., log, logit), the Jacobian adjustment is applied.

    Parameters:
    - initial_theta_info (dict): Dictionary containing prior information for each parameter. Each key corresponds 
      to a parameter name, and its value should contain a dictionary with the following keys:
        - 'prior': A list containing prior information [lower, upper, mean, std, distribution, transformation_type].
    - theta (np.array): Array of transformed parameter values (i.e., parameters that may have been transformed before use).

    Returns:
    - total_log_prior (float): The total log prior for all parameters, adjusted for any transformations.
    """
    
    theta_names = list(initial_theta_info.keys())
    total_log_prior = 0  # Initialize the total log prior

    for i, value in enumerate(theta):
        # Unpack prior information for each parameter
        lower, upper, mean, std, distribution, trans = initial_theta_info[theta_names[i]]['prior']

        # Handle transformations and Jacobian adjustment
        if trans == 'log':
            # Apply log transformation: theta = exp(value), and adjust the log prior
            jacobian_adjustment = value  # Jacobian for log transformation: exp(value)
            theta_original = np.exp(value)
        elif trans == 'logit':
            # Apply logit transformation: theta = 1 / (1 + exp(-value)), and adjust the log prior
            theta_original = 1 / (1 + np.exp(-value))  # Inverse logit (sigmoid) transformation
            jacobian_adjustment = np.log(theta_original) + np.log(1 - theta_original)  # Jacobian for logit
        else:
            # No transformation applied, no Jacobian adjustment needed
            theta_original = value
            jacobian_adjustment = 0

        # Compute the log prior based on the distribution
        if distribution == 'uniform':
            log_prior = uniform.logpdf(theta_original, loc=lower, scale=upper - lower)
        elif distribution == 'normal':
            log_prior = norm.logpdf(theta_original, loc=mean, scale=std)
        elif distribution == 'truncnorm':
            a, b = (lower - mean) / std, (upper - mean) / std
            log_prior = truncnorm.logpdf(theta_original, a, b, loc=mean, scale=std)
        elif distribution == 'lognormal':
            log_prior = lognorm.logpdf(theta_original, loc=mean, scale=std)
        elif distribution == 'gamma':
            log_prior = gamma.logpdf(theta_original, lower, scale=upper)
        elif distribution == 'invgamma':
            log_prior = invgamma.logpdf(theta_original, lower, scale=upper)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution}")

        # Accumulate the log prior for all parameters
        total_log_prior += log_prior

    # Return the total log prior with the Jacobian adjustment
    return total_log_prior + jacobian_adjustment

