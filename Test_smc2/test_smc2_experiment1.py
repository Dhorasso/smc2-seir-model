#####################################################################################
# Application of SMC^2 for the Experiment 1 in the paper
# Note: All functions must be in the same folder.
#####################################################################################



# import the necessary libraies

# Standard Libraries
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson, norm, nbinom
from numpy.random import binomial, normal
from joblib import Parallel, delayed  # For parallel computing
from plotnine import *
from tqdm import tqdm

# SMC2 Libraries
from smc2 import SMC_squared
from smc_visualization import trace_smc, plot_smc
# Style Configuration
plt.style.use('ggplot')

############  SEPTP 1: Import/create your dataset ###########################
#### Generate the simulated data with costant parameters ###############
#######################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def seir_const_beta(y, theta, dt=1):
    S, E, I, R, NI = y
    N = S + E + I + R
    beta, sigma, gamma = theta

    # Transition probabilities
    P_SE = 1 - np.exp(-beta * I / N * dt)
    P_EI = 1 - np.exp(-sigma * dt)
    P_IR = 1 - np.exp(-gamma * dt)

    # Binomial distributions for transitions
    B_SE = binomial(S, P_SE)
    B_EI = binomial(E, P_EI)
    B_IR = binomial(I, P_IR)

    # Update compartments
    S -= B_SE
    E += B_SE - B_EI
    I += B_EI - B_IR
    R += B_IR
    NI = B_EI

    # Ensure non-negative values
    return [max(0, compartment) for compartment in [S, E, I, R, NI]]


def solve_seir_const_beta(model, theta, InitialState, t_start, t_end, dt=1):
    t_values = np.arange(t_start, t_end + dt, dt)
    results = np.zeros((len(t_values), len(InitialState)))

    # Set initial conditions
    results[0] = InitialState

    # Solve using Euler method
    for i in range(1, len(t_values)):
        results[i] = model(results[i - 1], theta, dt)

    # Convert to DataFrame for easy handling
    results_df = pd.DataFrame(results, columns=['S', 'E', 'I', 'R', 'NI'])
    results_df['time']=t_values
    results_df['obs'] = np.random.poisson(results_df['NI'])
    return results_df


# Parameters and initial state
true_theta = [0.6, 1/3, 1/5] # true parameters
InitialState = [6000-1, 0, 1, 0, 0]
t_start = 0
t_end = 100
dt= 1

np.random.seed(123)
simulated_data = solve_seir_const_beta(seir_const_beta, true_theta, InitialState, t_start, t_end, dt)

# Plot results
plt.figure(figsize=(10, 4))

plt.plot(simulated_data['time'].index, simulated_data['obs'], label='New Infectied')

plt.xlabel('Time')
plt.ylabel('Population')

plt.legend()
plt.grid(True)
plt.show()

simulated_data
########################################################################################
########### SEPTP 2: Define your compartmental model# ####################################
# The epi_model.py file contains others example of SEIR models
##########################################################################################

def stochastic_seir_model_const_beta(y, theta, theta_names, dt=1):
    """
    Vectorized discrete-time stochastic SEIR compartmental model.
    Parameters:
    ----------
    y : np.ndarray
        A 2D array of compartments with shape (num_particles, num_compartments). 
        Columns represent the following compartments:
        [S (Susceptible), E (Exposed), I (Infected), R (Removed), 
         NI (new infected)].
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
    S, E, I, R, NI = y.T

    # Calculate total population for each particle
    N = S + E + I + R

    # Unpack parameters into a dictionary for easy access
    param = dict(zip(theta_names, theta))
    beta = param['beta']       # transmission rate   
    sigma = param['sigma']     # latency rate
    gamma = param['gamma']     # recovery rate   

    # Transition probabilities (vectorized)
    P_SE = 1 - np.exp(-beta * I / N * dt)             # Susceptible → Exposed
    P_EI = 1 - np.exp(-sigma * dt)                 # Exposed → Infected
    P_IR = 1 - np.exp(-gamma * dt)                  # Infected → Removed

    # Simulate transitions using binomial draws
    Y_SE = binomial(S.astype(int), P_SE)   # S → E
    Y_EI = binomial(E.astype(int), P_EI)  # E → I
    Y_IR = binomial(I.astype(int), P_IR)    # I → R

    # Update compartments
    S_next = S - Y_SE
    E_next = E + Y_SE - Y_EI
    I_next = I + Y_EI - Y_IR
    R_next = R + Y_IR

    # Update new infected
    NI_next = Y_EI

    # Combine updated compartments into a 2D array
    y_next = np.column_stack((S_next, E_next, I_next, R_next, NI_next))
    
    # Ensure all compartments remain non-negative
    return np.maximum(y_next, 0)



#################################################################################################
############ SEPTP 3: Define your observation distribution ######################################
# The observation_dist.py contains some examples, you extend to incorporate multiple dataste
#################################################################################################
# Poisson log-likelihood
def obs_dist_poisson(observed_data, model_data, theta, theta_names):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    log_likelihoods = poisson.logpmf(observed_data['obs'], mu=model_est_case)
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods

#################################################################################################
############ SEPTP 4: Run the SMC^2 #####################################################################
# You need to defined initial conditions for the state and prior for the parameter you want to estimate
##########################################################################################################

np.random.seed(123) # Set a seed for reproducibility

# # ##### # setting state and parameter
# Initial state information
state_info = {
    'S': {'prior': [6000-5, 6000, 0,0, 'uniform']},  # Susceptibles
    'E': {'prior': [0, 0, 0,0, 'uniform']},          # Exposed
    'I': {'prior': [0,5, 0,0, 'uniform']},          # Infected
    'R': {'prior': [0, 0, 0,0, 'uniform']},         # Removed
    'NI': {'prior': [0, 0, 0,0, 'uniform']},        # New Infected
}

# Initial parameter information
theta_info = {
    'beta': {'prior': [1e-5, 1,0,0, 'uniform','log']},   # Transmission rate
    'sigma': {'prior': [1e-5, 1,0,0, 'uniform','log']},  # Latency rate
    'gamma': {'prior': [1e-5, 1,0,0, 'uniform','log']},  # Removal rate
}




np.random.seed(123)
smc2_results = SMC_squared(
    model=stochastic_seir_model_const_beta,
    initial_state_info=state_info,
    initial_theta_info=theta_info,
    observed_data=simulated_data,
    num_state_particles=200,
    num_theta_particles=400,
    observation_distribution=obs_dist_poisson,
)


# Print the Marginal log-likelihood
print("Marginal log-likelihood:", smc2_results['margLogLike'])

##########################################################################################################
########## SETP5: Visualize the Results ####################################################################
# You can plot the filtered estimate of the state and parametersx
############################################################################################################

# state trajectory particles and extract corresponding matrix
trajParticles_state = smc2_results['trajState']
matrix_dict_state = trace_smc(trajParticles_state)

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 1, figsize=(10, 4))  # Modify number of rows/columns if you have more states

for (state, matrix) in matrix_dict_state.items():
    if state == 'NI':
        # Plot the SMC results using the plot_smc function (make sure to pass the correct ax)
        plot_smc(matrix, axs)

        # Plot observed data points (similar to geom_point)
        axs.scatter(simulated_data['time'], simulated_data['obs'], color='darkorange', edgecolor='salmon', s=30, label='Observed Data')

        # Customize labels and axis
        axs.set_xlabel('Time (days)', fontsize=14, fontweight='bold')
        axs.set_ylabel('Daily new cases', fontsize=14, fontweight='bold')

        # Show legend and grid
        axs.legend(loc='upper right')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot 
plt.show()

#############################################################################
# Plot for the parameter trajectories
##############################################################################

# Example of labels 
L = [r'$\beta$', r'$\sigma$', r'$\gamma$']

trajParticles_theta = smc2_results['trajtheta']
matrix_dict_theta = trace_smc(trajParticles_theta)

data_for_corner = []
labels = []

# Collect data and labels for each parameter
for i, (state, matrix) in enumerate(matrix_dict_theta.items()):
    data = matrix[:, -1]  # Using the last column of each matrix
    data_for_corner.append(data)
    labels.append(L[i])

data_for_corner = np.column_stack(data_for_corner)

# Create a 3x3 subplot layout
N=len(matrix_dict_theta) 
fig, axs = plt.subplots(N-1, N, figsize=(18, 8))

# Row 1: Plot time series using `plot_smc(matrix)` for each parameter
for i, (state, matrix) in enumerate(matrix_dict_theta.items()):

    plot_smc(matrix, ax=axs[0, i])  # Assuming plot_smc returns data suitable for plotting
    # axs[0, i].set_title(f'Time Series of {L[i]}', fontsize=14)
    #plot_smc2(matrix_dict_full[state], ax=axs[0, i])
    axs[0, i].set_xlabel('Time (days)', fontsize=18, fontweight='bold')
    axs[0, i].set_ylabel(L[i],  fontsize=18, fontweight='bold')
    
    # Add a horizontal line for the true parameter values
    axs[0, i].axhline(y=true_theta[i], color='orange', linestyle='--', linewidth=3, label='True Value')
axs[0, i].legend(fontsize=14)

# Row 2: Histograms for each parameter
for idx, label in enumerate(labels):
    axs[1, idx].hist(data_for_corner[:, idx], bins=20, density=True, color='navy', alpha=0.4, lw=5)
    sns.kdeplot(data_for_corner[:, idx], ax=axs[1, idx], color='dodgerblue', lw=3)
    
    # Calculate 0.25 and 0.975 CIs
    ci_025 = np.percentile(data_for_corner[:, idx], 2.5)
    ci_975 = np.percentile(data_for_corner[:, idx], 97.5)
    median_estimate = np.median(data_for_corner[:, idx])
    
    # Set title including the median value
    axs[1, idx].set_title(f'{label}= {median_estimate:.3f} (95%CrI: [{ci_025:.3f}, {ci_975:.3f}])', fontsize=14, fontweight='bold')
    
    axs[1, idx].set_xlabel(L[idx], fontsize=18, fontweight='bold')
    axs[1, idx].set_ylabel('Density', fontsize=18, fontweight='bold')
    
    # Add vertical line for true value and median
    axs[1, idx].axvline(true_theta[idx], color='orange', linestyle='--', linewidth=3, label='True Value')
    axs[1, idx].axvline(median_estimate, color='k', linewidth=3, label='Median')
    
    # Add dashed lines for 0.25 and 0.975 CIs
    axs[1, idx].axvline(ci_025, color='dodgerblue', linestyle='--', linewidth=2)
    axs[1, idx].axvline(ci_975, color='dodgerblue', linestyle='--', linewidth=2)
    
    # Add legend
axs[1, idx].legend(fontsize=14)

plt.tight_layout()
plt.show()



