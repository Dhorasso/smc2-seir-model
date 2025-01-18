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

############ import your dataset #######################################
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

########### Define your compartmental model########################################################
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

############ Define your observation distribution #####################################################################################
# The observation_dist.py contains some examples, you extend to incorporate multiple dataste
#######################################################################################################################################
