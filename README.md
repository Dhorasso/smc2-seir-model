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
- **`model`**: The model function (e.g., SIR, SEIR, or other stochastic models).  The user can refer to  the file [epi_model.py](https://github.com/Dhorasso/smc2_seir/blob/main/epi_model.py) for some examples. 
- **`initial_state_info`**: Dictionary specifying the prior distributions for state variables. Each state variable should include:
  - `[lower_bound, upper_bound, mean, std_deviation, distribution_type, transformation]`.
- **`initial_theta_info`**: Dictionary specifying the prior distributions for parameters. Each parameter should include:
  - `[lower_bound/shape, upper_bound/scale, mean, std_deviation, distribution_type, transformation]`.
    
    - Supported distributions include: `'uniform'`, `'normal'`, `'truncnorm'`, `'lognormal'`, `'gamma'`, `'invgamma'`.
    - Supported transformation (apply to the drawn constraint values) include: `'log'`, `'logit'`. if not define then it remain the same.
- **`observed_data`**: Observed data in a `pandas.DataFrame` format with the column name of observation data `'obs'`.
- **`num_state_particles`**: Number of state particles to use in the Particle_Filter.
- **`num_theta_particles`**: Number of parameter particles.
- - **`observation_distribution`**: Probability distribution for observations. The user can refer to  the file [observation_dist.py](https://github.com/Dhorasso/smc2_seir/blob/main/observation_dist.py) for some examples. 

### Optional Parameters
- **`resampling_threshold`**: Threshold for resampling based on the effective sample size (ESS). *(Default: 0.5)*.
- **`pmmh_moves`**: Number of PMMH move in the rejuvenation step. *(Default: 5)*
- **`c`**: Scaling factor for the covariance matrix in the PMMH kernel. *(Default: 0.5)*
- **`n_jobs`**: Number of processor in the PMMH parallel computing   *(Default: 10)* user can increase depending computer performance
- **`resampling_method`**: Resampling method (`'stratified', 'systematic', 'residual', or 'multinomial'`). *(Default: `'stratified'`)*.
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


An example of using the `SMC_squared` function with an SEIR model for the Irish COVID-19 data is provided in the file [test_covid.py](https://github.com/Dhorasso/smc2_seir/blob/main/test_covid.py).


