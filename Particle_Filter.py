import numpy as np
import pandas as pd
import gc


def Particle_Filter(model, state_names, current_state_particles, theta, theta_names, observed_data, 
                    num_state_particles, resampling_method='stratified', 
                    observation_distribution='normal_approx_NB', add=0, end=False, forecast_days=0):
    """
    Perform Particle Filter to estimate the state and compute the marginal log-likelihood.

    Parameters:
    - model: The model function.
    - state_names (list): Names of the state variables.
    - current_state_particles (ndarray): The initial state particles.
    - theta (ndarray): Model parameters.
    - theta_names (list): Names of the theta parameters.
    - observed_data (pd.DataFrame): Observed data for likelihood calculation.
    - num_state_particles (int): Number of particles.
    - resampling_method (str): Resampling method ('stratified', 'systematic', etc.).
    - observation_distribution (str): Type of observation distribution 
    - add (int): Flag to indicate whether to store state history.
    - end (bool): Flag to control trajectory addition.
    - forecast_days (int): Number of forecast days after observed data ends.

    Returns:
    - dict: Marginal log-likelihood, particle states, state history, and trajectory states.
    """
    # Initialize particle weights and trajectory placeholders
    particle_weights = np.ones(num_state_particles) / num_state_particles
    num_timesteps = len(observed_data)
    
    traj_state = [{key: [] for key in ['time'] + state_names} for _ in range(num_state_particles)]
    state_hist = [None] * num_timesteps
    marginal_log_likelihood = 0

    # Main loop for particle filtering
    for t in range(num_timesteps + forecast_days):
        t_start, t_end = (0, 0) if t == 0 else (t - 1, t)

        # Process observations for the current timestep if within the observed data range
        if t < num_timesteps:
            current_data_point = observed_data.iloc[t]

        # Solve model and calculate trajectories
        trajectories = state_transition(model, theta, current_state_particles, state_names, theta_names, t_start, t_end)
        model_points = trajectories.to_numpy()

        # Compute log weights for the model if within observed data range
        if t < num_timesteps:
            weights = compute_log_weight(current_data_point, trajectories, theta, theta_names, observation_distribution)

            # Normalize and resample Particles
            A = np.max(weights)
            weights_mod = np.ones_like(weights) if A < -1e2 else np.exp(weights - A)
            normalized_weights = weights_mod / np.sum(weights_mod)
            resampled_indices = resampling_style(normalized_weights, resampling_method)
            current_state_particles = model_points[resampled_indices]

            # Likelihood update
            zt = max(np.mean(np.exp(weights)),1e-12) # Prevent division by zero
            marginal_log_likelihood += np.log(zt)

        # Store state history and trajectories
        if add == 1:
            if end:
                # Use Parallel for efficient trajectory addition
                traj_state = Parallel(n_jobs=10)(
                    delayed(lambda traj, j: pd.DataFrame(
                        {'time': list(traj['time']) + [t], 
                         **{name: list(traj[name]) + [current_state_particles[j][i]] for i, name in enumerate(state_names)}}
                    ))(traj, j) 
                    for j, traj in enumerate(traj_state)
                )
            else:
                state_hist[t] = current_state_particles
        
        # Perform garbage collection to free up memory
        gc.collect()

    # Return the results
    return {
        'margLogLike': marginal_log_likelihood, 
        'particle_state': current_state_particles, 
        'state_hist': state_hist, 
        'traj_state': traj_state
    }
