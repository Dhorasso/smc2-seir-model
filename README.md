# Sequential Monte Carlo Squared for online inference in stochastic epidemic models

This repository content the Python code to run a Sequential Monte Carlo Squared (SMC^2) for online inference in a SEIR model.
This framework supports both offline and online SMC^2 applications for parameter and state estimation in dynamic systems.

---


## Installation
To install and set up the environment for running this model, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Dhorasso/smc2_seir.git
    ```


## Model Inputs

The `SMC_squared` function requires the following inputs:

### Required Parameters
- **`model`**: The model function (e.g., SIR, SEIR, or other stochastic models).
- **`initial_state_info`**: Dictionary specifying the prior distributions for state variables. Each state variable should include:
  - `[lower_bound, upper_bound, mean, std_deviation, distribution_type, transformation]`.
- **`initial_theta_info`**: Dictionary specifying the prior distributions for parameters. Each parameter should include:
  - `[lower_bound/shape, upper_bound/scale, mean, std_deviation, distribution_type, transformation]`.
    
    - Supported distributions include: `'uniform'`, `'normal'`, `'truncnorm'`, `'lognormal'`, `'gamma'`, `'invgamma'`.
    - Supported transformation (apply to the drawn constraint values) include: `'log'`, `'logit'`. if not define then it remain the same.
- **`observed_data`**: Observed data in a `pandas.DataFrame` format with the column name of observation data `'obs'`.
- **`num_state_particles`**: Number of state particles to use in the Particle_Filter.
- **`num_theta_particles`**: Number of parameter particles.

### Optional Parameters
- **`resampling_threshold`**: Threshold for resampling based on the effective sample size (ESS). *(Default: 0.5)*.
- **`resampling_method`**: Resampling method (`'stratified', 'systematic', 'residual', or 'multinomial'`). *(Default: `'stratified'`)*.
- **`observation_distribution`**: Distribution for observations (`'poisson', 'normal', 'normal_approx_NB', or 'negative_binomial`). *(Default: `poisson`)*.
- **`tw`**: Window size for online SMC^2 (If the it run the full-SMC^2).
- **`SMC2_results`**: Results from previous SMC^2 runs (used as priors for online updates).
- **`Real_time`**: Whether to run in real-time mode using prior SMC^2 results. *(Default: `False`)*.
- **`forecast_days`**: Number of forecast days. *(Default: `0`)*.
- **`show_progress`**: Whether to display a progress bar during computation. *(Default: `True`)*.

---

## Model Outputs

The function returns a dictionary containing:
- **`margLogLike`**: Marginal log-likelihood of the observed data given the model.
- **`trajState`**: State variable trajectories over time.
- **`trajtheta`**: Parameter trajectories over time.
- **`ESS`**: Effective sample size over time.
- **`acc`**: Acceptance rate over time.
- **`current_theta_particles`**: Current parameter particles.
- **`current_state_particles`**: Current state particles.
- **`current_state_particles_all`**: All state particles at the current time step.
- **`state_history`**: History of state trajectories.

---

## Example Usage

Here’s an example of using the `SMC_squared` function with an SEIR model:

```python
N_pop = 4965439
Ips_0_min = 10
Ips_0_max = 50
E_0 = 1
S_0_min = N_pop - Ips_0_max - E_0
S_0_max = N_pop - Ips_0_min - E_0

# Initial state information
state_info = {
    'S': {'prior': [S_0_min, S_0_max, 0, 0, 'uniform']},
    'E': {'prior': [E_0, E_0, 0, 0, 'uniform']},
    'A': {'prior': [Ips_0_min, Ips_0_max, 0, 0, 'uniform']},
    'I': {'prior': [0, 0, 0, 0, 'uniform']},
    'R': {'prior': [0, 0, 0, 0, 'uniform']},
    'NI': {'prior': [0, 0, 0, 0, 'uniform']},
    'B': {'prior': [0.6, 0.8, 0, 0, 'uniform']}
}

# Initial parameter information
theta_info = {
    'ra': {'prior': [0.1, 0.5, 0.15, 0.05, 'uniform', 'logit']},
    'pa': {'prior': [0.3, 1, 0.15, 0.05, 'uniform', 'logit']},
    'sigma': {'prior': [1/5, 1/3, 1/4, 0.1, 'truncnorm', 'log']},
    'gamma': {'prior': [1/7.5, 1/4.5, 1/6, 0.2, 'truncnorm', 'log']},
    'nu_beta': {'prior': [0.05, 0.15, 0.1, 0.05, 'uniform', 'log']},
    'phi': {'prior': [0.01, 0.2, 0, 0, 'uniform', 'log']}
}

# Running the SMC^2 function
results = SMC_squared(
    model=stochastic_model_covid,
    initial_state_info=state_info,
    initial_theta_info=theta_info,
    observed_data=data,
    num_state_particles=500,
    num_theta_particles=1000,
    observation_distribution='normal_approx_NB',
    tw=80,
    forecast_days=projection_day,
    show_progress=True
)

print("Marginal log-likelihood:", results['margLogLike'])
