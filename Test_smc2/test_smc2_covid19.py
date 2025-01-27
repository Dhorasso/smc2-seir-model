#####################################################################################
# Application of SMC^2 for the COVID-19  data in Ireland
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

############  SEPTP 1:import your dataset #######################################
# Assuming the uploaded file is named "COVID-19_HPSC_Detailed_Statistics_Profile.csv"
######################################################################
file_path = r"COVID-19_HPSC_Detailed_Statistics_Profile.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)
# Restrict the observations to 280 days
days = 280
data = df.iloc[:days].copy()

# Ensure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Plotting with matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the Confirmed Covid Cases
ax.plot(data['Date'], data['ConfirmedCovidCases'], color='blue')

# Formatting the plot
# ax.set_title("COVID-19 Confirmed Cases Over Time", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Number of Cases", fontsize=12)
ax.grid(True)

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


##############################################################################################
########### SEPTP 2: Define your compartmental model###################################
# The epi_model.py file contains others example of SEIR models
#############################################################################################

def stochastic_seair_model(y, theta, theta_names, dt=1):
    """
    Vectorized discrete-time stochastic SEAIR compartmental model.
    
    Parameters:
    ----------
    y : np.ndarray
        A 2D array of compartments with shape (num_particles, num_compartments). 
        Columns represent the following compartments:
        [S (Susceptible), E (Exposed), A (Asymptomatic), I (Infected), R (Recovered), 
         NI (New Infected), B (Transmission Rate)].
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

    # Update new infected
    NI_next = Y_EI

    # Combine updated compartments into a 2D array
    y_next = np.column_stack((S_next, E_next, A_next, I_next, R_next, NI_next, B_next))
    
    # Ensure all compartments remain non-negative
    return np.maximum(y_next, 0)

#####################################################################################################
############ SEPTP 3: Define your observation distribution ############################################
# The observation_dist.py contains some examples, you extend to incorporate multiple dataste
######################################################################################################

def obs_dist_normal_approx_NB(observed_data, model_data, theta, theta_names):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    param = dict(zip(theta_names, theta))
    overdispersion = param.get('phi', 0.1)  # Default value for 'phi' if not provided
    variance = model_est_case * (1 + overdispersion * model_est_case)
    variance = np.maximum(variance, 1)  # Ensure variance is at least 1
    log_likelihoods = norm.logpdf(observed_data['ConfirmedCovidCases'], loc=model_est_case, scale=np.sqrt(variance)) 
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods

#######################################################################################################
############ SEPTP 4: Run the SMC^2 #############################################################
# You need to defined initial conditions for the state and prior for the parameter you want to estimate
#########################################################################################################
np.random.seed(123) # This is the random seed we use to generate also synthetic data

N_pop = 4965439
A_0_min = 10
A_0_max = 50
E_0 = 1
S_0_min = N_pop - A_0_max - E_0
S_0_max = N_pop - A_0_min - E_0

# Initial state information
state_info = {
    'S': {'prior': [S_0_min, S_0_max, 0, 0, 'uniform']},
    'E': {'prior': [E_0, E_0, 0, 0, 'uniform']},
    'A': {'prior': [A_0_min, A_0_max, 0, 0, 'uniform']},
    'I': {'prior': [0, 0, 0, 0, 'uniform']},
    'R': {'prior': [0, 0, 0, 0, 'uniform']},
    'NI': {'prior': [0, 0, 0, 0, 'uniform']}, # New Infected
    'B': {'prior': [0.6, 0.8, 0, 0, 'uniform']}
}

# Initial parameter information
theta_info = {
    'ra': {'prior': [0.1, 0.5, 0, 0, 'uniform', 'logit']},
    'pa': {'prior': [0.3, 1, 0, 0, 'uniform', 'logit']},
    'sigma': {'prior': [1/5, 1/3, 1/4, 0.1, 'truncnorm', 'log']},
    'gamma': {'prior': [1/7.5, 1/4.5, 1/6, 0.2, 'truncnorm', 'log']},
    'nu_beta': {'prior': [0.05, 0.15, 0.1, 0.05, 'uniform', 'log']},
    'phi': {'prior': [0.01, 0.2, 0, 0, 'uniform', 'log']} # Overdisperssion parameter (use for 'normal'(here is the std.),
                                                          # 'normal_approx_NB', or 'negative_binomial observation_distribution)
}

# Running the SMC^2 function
smc2_results = SMC_squared(
    model=stochastic_seair_model,
    initial_state_info=state_info,
    initial_theta_info=theta_info,
    observed_data=data,
    num_state_particles=500,
    num_theta_particles=1000,
    observation_distribution=obs_dist_normal_approx_NB,
    tw=80, # increase the window size for better results
    c=0.5  , # chose between (0.5 , 1)
    forecast_days=0,
    show_progress=True
)

# Print the Marginal log-likelihood
print("Marginal log-likelihood:", smc2_results['margLogLike'])



#############################################################################################
########## SETP5: Visualize the Results #####################################################
# You can plot the filtered estimate of the state and parametersx
##############################################################################################

# state trajectory particles and extract corresponding matrix
state_trajectory_particles = smc2_results ['trajState']
state_matrix_dict = trace_smc(state_trajectory_particles)

# Initialize parameter trajectory particles and extract corresponding matrix
parameter_trajectory_particles = smc2_results ['trajtheta']
theta_matrix_dict = trace_smc(parameter_trajectory_particles)

# Extract median values for parameters
gamma = np.nanmedian(theta_matrix_dict['gamma'][:, -1])  # Median recovery rate
pa = np.nanmedian(theta_matrix_dict['pa'][:, -1])        # Fraction of infections that are asymptomatic
ra = np.nanmedian(theta_matrix_dict['ra'][:, -1])        # Relative recovery rate for asymptomatic cases

# Calculate the effective reproduction number Rt and add it to the satate dict
state_matrix_dict['Rt'] = (
    state_matrix_dict['B'] * 
    ((1 - pa) * 1 / gamma + pa * ra * 1 / gamma) * 
    state_matrix_dict['S'] / N_pop
)

############################################################################################################

# Set up a 1-row, 2-column figure layout
fig, axs = plt.subplots(1, 2, figsize=(10*2, 2.5*2))  # 1 row, 2 columns

# Plot 'NI' (New Infected) on the first subplot
for state, matrix in state_matrix_dict.items():
    if state == 'NI':
        # Plot model predictions using plot_smc function
        plot_smc(matrix, ax=axs[0], Date= data['Date'])  # Use d1 for Date
        # Scatter observed data for fitting
        axs[0].scatter(
            data['Date'], data['ConfirmedCovidCases'],
            color='orange', edgecolor='salmon', s=30, label='Observed Data'
        )


        # Set labels for the 'NI' plot
        axs[0].set_xlabel('Date', fontsize=14, fontweight='bold')
        axs[0].set_ylabel('Daily New Cases', fontsize=14, fontweight='bold')
        axs[0].legend(loc='upper right', fontsize=14)
        # axs[0].axvline(x=d4['Date'][days4 - 1], color='white', linestyle=':', linewidth=0.1)

# Plot 'Rt' (Effective Reproduction Number) on the second subplot
for state, matrix in state_matrix_dict.items():
    if state == 'Rt':
        # Plot the effective reproduction number using plot_smc function
        plot_smc(matrix, ax=axs[1], Date= data['Date'], window=7) ## we have use here a 7 days moving average (default is 1)
        # Add a horizontal line at Rt = 1
        axs[1].axhline(y=1, color='k', linestyle='--', linewidth=3, label='$R_{eff}(t) = 1$')

        # Set limits and labels for the 'Rt' plot
        axs[1].set_ylim(0, 8)
        axs[1].set_xlabel('Date', fontsize=14, fontweight='bold')
        axs[1].set_ylabel(r'Effective Reproduction Number $R_{eff}(t)$', fontsize=10, fontweight='bold')
        axs[1].legend(loc='upper right', fontsize=14)
        # axs[1].axvline(x=d4['Date'][days4 - 1], color='white', linestyle=':', linewidth=0.1)

plt.tight_layout( w_pad=4)
plt.show()

################################################################################
#################################################################################
# Plot for the parameter trajectories

# Determine grid size
total_plots = len(theta_matrix_dict)
nrows = math.ceil(math.sqrt(total_plots))
ncols = math.ceil(total_plots / nrows)

# Plot for parameter trajectories
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))

# Flatten axes array for easy iteration
axes = axes.flatten()

for i, (state, matrix) in enumerate(theta_matrix_dict.items()):
    if i >= nrows * ncols:
        break
    plot_smc(matrix, ax=axes[i])  # Replace with your plotting function
    ci_025 = np.percentile(matrix[:, -1], 2.5)
    ci_975 = np.percentile(matrix[:, -1], 97.5)
    median_estimate = np.median(matrix[:, -1])
    axes[i].set_title(f'{state} = {median_estimate:.3f} (95% CrI: [{ci_025:.3f}, {ci_975:.3f}])',
                      fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Date', fontsize=14, fontweight='bold')
    axes[i].set_ylabel(state, fontsize=16, fontweight='bold')

    axes[i].legend(fontsize=14, loc='best')

# Adjust layout and display the figure
plt.tight_layout()
plt.show()