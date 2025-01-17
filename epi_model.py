
###############################################################################################################
#  This file con


import pandas as pd
import numpy as np


def stochastic_model_covid(y, theta, theta_names, dt=1):
    """
    Vectorized discrete-time stochastic compartmental model for COVID-19 dynamics.
    
    Parameters:
    ----------
    y : np.ndarray
        A 2D array of compartments with shape (num_particles, num_compartments). 
        Columns represent the following compartments:
        [S (Susceptible), E (Exposed), A (Asymptomatic), I (Infected), R (Recovered), 
         NI (New Infections), B (Transmission Rate)].
    theta : np.ndarray
        A 1D array of parameter values, one per particle. The parameters must match the order in `theta_names`.
    theta_names : list of str
        List of parameter names, matching the order in `theta`.
    dt : float, optional
        Time step for discrete updates (default is 1).

    Returns:
    -------
    np.ndarray
        Updated 2D array of compartments with the same shape as input `y`.
    """

    # Unpack compartments (columns of y)
    S, E, A, I, R, NI, B = y.T

    # Calculate total population for each particle
    N = S + E + A + I + R

    # Unpack parameters into a dictionary for easy access
    param = dict(zip(theta_names, theta))
    pa = param.get('pa', 0.5)  # Default value for `pa` if not specified
    ra = param.get('ra', 0.5)  # Default value for `ra` if not specified
    sigma = param['sigma']
    gamma = param['gamma']
    nu_beta = param.get('nu_beta', 0.1)  # Default value for `nu_beta` if not specified

    # Transition probabilities (vectorized)
    P_SE = 1 - np.exp(-B * (I + ra * A) / N * dt)  # Susceptible → Exposed
    P_EAI = 1 - np.exp(-sigma * dt)                 # Exposed → Asymptomatic/Infected
    P_AR = 1 - np.exp(-gamma * dt)                  # Asymptomatic → Recovered
    P_IR = P_AR                                      # Infected → Recovered

    # Simulate transitions using binomial draws
    Y_SE = np.random.binomial(S.astype(int), P_SE)   # S → E
    Y_EAI = np.random.binomial(E.astype(int), P_EAI)  # E → A/I
    Y_EA = np.random.binomial(Y_EAI, pa)              # E → A
    Y_EI = Y_EAI - Y_EA                               # E → I
    Y_AR = np.random.binomial(A.astype(int), P_AR)    # A → R
    Y_IR = np.random.binomial(I.astype(int), P_IR)    # I → R

    # Update compartments
    S_next = S - Y_SE
    E_next = E + Y_SE - Y_EAI
    A_next = A + Y_EA - Y_AR
    I_next = I + Y_EI - Y_IR
    R_next = R + Y_AR + Y_IR
    
    # Update transmission rate with stochastic volatility 
    B_next = B * np.exp(nu_beta * np.random.normal(0, 1, size=B.shape) * dt)

    # Update new infections
    NI_next = Y_EI

    # Combine updated compartments into a 2D array
    y_next = np.column_stack((S_next, E_next, A_next, I_next, R_next, NI_next, B_next))
    
    # Ensure all compartments remain non-negative
    return np.maximum(y_next, 0)


def state_transition(model, theta, initial_state, state_names, theta_names, t_start, t_end, dt=1):
    """
    Solve a stochastic disease model using vectorized computation.

    Parameters:
    - model: Model function
    - theta: Parameter array
    - initial_state: Initial state array (num_particles x num_compartments)
    - state_names: Names of compartments
    - theta_names: Names of parameters
    - t_start: Start time
    - t_end: End time
    - dt: Time step (default is 1)

    Returns:
    - results_df: DataFrame containing results at the last time step
    """
    # Time points
    t_points = np.arange(t_start, t_end + dt, dt)
    num_steps = len(t_points)

    # Initialize results array
    num_particles, num_compartments = initial_state.shape
    results = np.zeros((num_steps, num_particles, num_compartments))
    results[0] = initial_state  # Set initial state

    # Run the model for each time step
    for i in range(1, num_steps):
        results[i] = model(results[i - 1], theta, theta_names, dt)

    # Return results as DataFrame for the last time step
    return pd.DataFrame(results[-1], columns=state_names)

