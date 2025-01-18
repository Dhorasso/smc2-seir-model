#####################################################################################
# Application of SMC^2 for the COVID-19  data in Ireland
#####################################################################################


# import the necessary libraies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # For parallel computing
from plotnine import*
from tqdm import tqdm 
from smc_visualization import trace_smc, plot_smc

############  SEPTP 1:import your dataset #######################################
# Assuming the uploaded file is named "COVID-19_HPSC_Detailed_Statistics_Profile.csv"
######################################################################
file_path = r"COVID-19_HPSC_Detailed_Statistics_Profile.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Assuming df is your original DataFrame
d = df[['Date', 'ConfirmedCovidCases', 'ConfirmedCovidDeaths', 'HospitalisedCovidCases']].fillna(0)

# Create a column of cumulative deaths
d['Death'] = d['ConfirmedCovidDeaths'].cumsum()

# Restrict the observations to 280 days
days = 280
data = d.iloc[:days].copy()  # use iloc to avoid potential slicing issues


# Rename the 'ConfirmedCovidCases' column to 'obs' (avoid inplace=True)
data = data.rename(columns={'ConfirmedCovidCases': 'obs'})

# Plot using ggplot
(ggplot(data) +
 aes(x='Date', y='obs') +
 geom_line()

)

##############################################################################################

########### SEPTP 2: Define your compartmental model########################################################
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
    P_EI = 1 - np.exp(-sigma * dt)                 # Exposed → Infected
    P_IR = 1 - np.exp(-gamma * dt)                  # Infected → Recovered


    # Simulate transitions using binomial draws
    Y_SE = np.random.binomial(S.astype(int), P_SE)   # S → E
    Y_EI = np.random.binomial(E.astype(int), P_EAI)  # E → I
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

######################################################################################################################################

############ SEPTP 3: Define your observation distribution #####################################################################################
# The observation_dist.py contains some examples, you extend to incorporate multiple dataste
#######################################################################################################################################

def obs_dist_normal_approx_NB(observed_data, model_data, theta, theta_names):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    param = dict(zip(theta_names, theta))
    overdispersion = param.get('phi', 0.1)  # Default value for 'phi' if not provided
    variance = model_est_case * (1 + overdispersion * model_est_case)
    variance = np.maximum(variance, 1)  # Ensure variance is at least 1
    log_likelihoods = norm.logpdf(observed_data['obs'], loc=model_est_case, scale=np.sqrt(variance))
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods

######################################################################################################################################

############ SEPTP 4: Run the SMC^2 #####################################################################################
# You need to defined initial conditions for the state and prior for the parameter you want to estimate
#######################################################################################################################################

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
    'phi': {'prior': [0.01, 0.2, 0, 0, 'uniform', 'log']} # Overdisperssion parameter (use for 'normal'(here is the std.),
                                                          # 'normal_approx_NB', or 'negative_binomial observation_distribution)
}

# Running the SMC^2 function
results = SMC_squared(
    model=stochastic_model_covid,
    initial_state_info=state_info,
    initial_theta_info=theta_info,
    observed_data=data,
    num_state_particles=500,
    num_theta_particles=1000,
    observation_distribution=obs_dist_normal_approx_NB,
    tw=80,
    forecast_days=projection_day,
    show_progress=True
)

# Print the Marginal log-likelihood
print("Marginal log-likelihood:", results['margLogLike'])

####################################################################################################################################################

########## SETP5: Visualize the Results ###############################################################################
# You can plot the filtered estimate of the state and parametersx
###########################################################################################################################
