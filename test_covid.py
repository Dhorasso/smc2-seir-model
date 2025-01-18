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
