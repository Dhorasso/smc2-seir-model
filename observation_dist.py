import numpy as np
from scipy.stats import poisson, norm, nbinom

def compute_log_weight(observed_data, model_data, theta, theta_names, distribution_type):
    """
    Compute log-likelihood for observed data given model predictions.

    Parameters:
    - observed_data: Array or DataFrame with observed data points.
    - model_data: Array or DataFrame with model predictions.
    - theta: Parameters of the model.
    - theta_names: Names of the parameters in theta.
    - distribution_type: Distribution type ('poisson', 'normal', 'normal_approx_NB', or 'negative_binomial').

    Returns:
    - log_likelihoods: Array of log-likelihoods for each observed data point.
    """
    # Convert model predictions to a minimum positive value to avoid log(0) issues
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())

    # Map parameter names to values
    param = dict(zip(theta_names, theta))

    # Calculate log-likelihood based on the distribution type
    if distribution_type == 'poisson':
        log_likelihoods = poisson.logpmf(observed_data['obs'], mu=model_est_case)

    elif distribution_type == 'normal':
        sigma_normal = param.get('phi', 0.1)  # Default value for 'phi' if not provided
        log_likelihoods = norm.logpdf(np.log(epsi + observed_data['obs']),
                                      loc=np.log(epsi + model_est_case),
                                      scale=sigma_normal)

    elif distribution_type == 'normal_approx_NB':
        overdispersion = param.get('phi', 0.1)  # Default value for 'phi' if not provided
        variance = model_est_case * (1 + overdispersion * model_est_case)
        variance = np.maximum(variance, 1)  # Ensure variance is at least 1
        log_likelihoods = norm.logpdf(observed_data['obs'], loc=model_est_case, scale=np.sqrt(variance))

    elif distribution_type == 'negative_binomial':
        overdispersion = param.get('phi', 0.1)  # Default value for 'phi' if not provided
        log_likelihoods = nbinom.logpmf(observed_data['obs'], 1 / overdispersion, 1 / (1 + overdispersion * model_est_case))

    else:
        raise ValueError("Invalid distribution type specified.")

    # Handle non-finite log-likelihood values (e.g., NaN or Inf)
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf

    return log_likelihoods
