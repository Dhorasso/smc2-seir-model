# Sequential Monte Carlo Squared for online inference in stochastic epidemic models

This repository content the code to run a Sequential Monte Carlo Squared (SMC^2)   for online inference in a SEIR model 


## Installation
To install and set up the environment for running this model, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Dhorasso/smc2_seir.git
    ```


##  Model Inputs

The  `Kernel_Smoothing_Filter` function thake as inputs:

- `model`: Model function (e.g., SIR, SEIR, extended-SEIR stochastic model)
- `initial_state_info`: Information about the initial state of the system  (dictionary)
- `initial_theta_info`: Initial parameters information  (dictionary)
- `observed_data`: Observed data (a DataFrame)
- `num_particles`: Number of particles 
- `resampling_threshold`: Threshold for effective sample size in resampling  
- `delta`: Parameter for updating theta during resampling  (default is 0.99) 
- `population_size`: Total population size  
- `resampling_method`: Method for particle resampling ('stratified' by default)  
- `observation_distribution`: Distribution of observations ('poisson by default) 
- `forecast_days`: Number of days to forecast  (default is 0)
- `dt`: Time step (default is 1)
- `num_cores`: Number of processor to be used in parallel ( defaut all available -1) 
- `show_progress`: Whether to display a progress bar during computation  (default is TRue)

#### Initial State Information (`initial_state_info`)

The `initial_state_info` dictionary should contain the initial state variables of the model. Each state variable should be defined with the following information:
- `state name `and  `prior distribution`: A list specifying `[lower_bound, upper_bound, mean, std_deviation, distribution_type]`.

#### Initial Parameters Information (`initial_theta_info`)

 The initial_theta_info dictionary should contain the initial parameters of the model. Each parameter should be defined with the following information:

- `parameter name` and `prior distribution`: A list specifying `[lower_bound/shape, upper_bound/scale, mean, std_deviation, distribution_type]`.The distribution can be 'uniform', 'normal', 'trunorm', 'lognormal', 'gamma', or 'invgamma'. The lower and upper values only work for 'uniform' and 'trunorm'.

##  Model Outputs 
- `margLogLike`: Marginal log-likelihood of the observed data given the model.
- `trajState`: Trajectories of the state variables over time.
- `trajtheta`: Trajectories of the model parameters over time.

## Example Usage
