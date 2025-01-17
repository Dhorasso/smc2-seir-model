def trace_smc(Traject):

    matrix_dict = {}
    stateName=list(Traject[0].columns[1:])
    # Iterate through each state name
    for state in stateName:
      # Extract matrices for each state from all dataframes
      state_matrices = [df[state].values.reshape(1, -1) for df in Traject]

      # Concatenate matrices horizontally
      combined_matrix = np.concatenate(state_matrices, axis=1)

      # Reshape the combined matrix based on the shape of the original dataframe
      reshaped_matrix = combined_matrix.reshape(-1,Traject[0].shape[0])

      # Store the reshaped matrix in the dictionary with state name as the key
      matrix_dict[state] = reshaped_matrix 


    return matrix_dict

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

# Importing the style package
plt.style.use('ggplot')

def corrected_matrix(matrix):
    # Calculate the IQR for each column
    q1 = np.percentile(matrix, 25, axis=0)
    q3 = np.percentile(matrix, 75, axis=0)
    iqr = q3 - q1
    
    # Calculate the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Replace outliers with appropriate values
    for col in range(matrix.shape[1]):
        col_values = matrix[:, col]
        col_non_outliers = col_values[(col_values >= lower_bound[col]) & (col_values <= upper_bound[col])]
        max_non_outlier = np.mean(col_non_outliers)
        min_non_outlier = np.mean(col_non_outliers)
        
        # Replace outliers above the upper bound
        outliers_above = col_values[col_values > upper_bound[col]]
        if len(outliers_above) > 0:
            matrix[col_values > upper_bound[col], col] = max_non_outlier
        
        # Replace outliers below the lower bound
        outliers_below = col_values[col_values < lower_bound[col]]
        if len(outliers_below) > 0:
            matrix[col_values < lower_bound[col], col] = min_non_outlier
    
    return matrix

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

# Importing the style package
plt.style.use('ggplot')

def corrected_matrix(matrix):
    # Calculate the IQR for each column
    q1 = np.percentile(matrix, 25, axis=0)
    q3 = np.percentile(matrix, 75, axis=0)
    iqr = q3 - q1
    
    # Calculate the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Replace outliers with appropriate values
    for col in range(matrix.shape[1]):
        col_values = matrix[:, col]
        col_non_outliers = col_values[(col_values >= lower_bound[col]) & (col_values <= upper_bound[col])]
        max_non_outlier = np.mean(col_non_outliers)
        min_non_outlier = np.mean(col_non_outliers)
        
        # Replace outliers above the upper bound
        outliers_above = col_values[col_values > upper_bound[col]]
        if len(outliers_above) > 0:
            matrix[col_values > upper_bound[col], col] = max_non_outlier
        
        # Replace outliers below the lower bound
        outliers_below = col_values[col_values < lower_bound[col]]
        if len(outliers_below) > 0:
            matrix[col_values < lower_bound[col], col] = min_non_outlier
    
    return matrix



def plot_smc(matrix, ax, color='skyblue', col1='steelblue', col2='midnightblue', Date=None, smooth=False, window_size=1):
    """
    Plot the SMC results using Matplotlib.
    
    Parameters:
    matrix: The input matrix where columns represent time steps and rows represent samples.
    ax: The matplotlib axis to plot on.
    color: The main color for the plot (used for the outermost ribbon).
    col1: The color for the middle ribbon (50% credible interval).
    col2: The color for the median line.
    Date: Optional date array for x-axis. If None, numeric time steps are used.
    smooth: Boolean flag to apply smoothing to the curves.
    window_size: The window size for smoothing (default is 7).
    """
   
    # Replace outliers in the matrix
    matrix = corrected_matrix(matrix)
    
    # Calculate the median along the columns (axis=0)
    median_values = np.nanmedian(matrix, axis=0)

    # Calculate the 95%, 90%, 75%, and 50% credible intervals
    credible_interval_95 = np.nanpercentile(matrix, [2.5, 97.5], axis=0)
    credible_interval_90 = np.nanpercentile(matrix, [5, 95], axis=0)
    credible_interval_75 = np.nanpercentile(matrix, [12.5, 85.5], axis=0)
    credible_interval_50 = np.nanpercentile(matrix, [25, 75], axis=0)

    # Apply smoothing if needed
    if smooth:
        median_values = pd.Series(median_values).rolling(window=window_size, min_periods=1).mean().values
        credible_interval_95 = pd.DataFrame(credible_interval_95).T.rolling(window=window_size, min_periods=1).mean()
        credible_interval_90 = pd.DataFrame(credible_interval_90).T.rolling(window=window_size, min_periods=1).mean()
        credible_interval_75 = pd.DataFrame(credible_interval_75).T.rolling(window=window_size, min_periods=1).mean()
        credible_interval_50 = pd.DataFrame(credible_interval_50).T.rolling(window=window_size, min_periods=1).mean()

    # Define time steps as either numeric or date-based
    T = matrix.shape[1]
    if Date is not None:
        time_steps = Date
    else:
        time_steps = np.arange(T)

    # Use a color map
    blues = matplotlib.colormaps['Blues']

    # Plot credible intervals using fill_between with the Blues colormap
    ax.fill_between(time_steps, credible_interval_95[0], credible_interval_95[1], color=blues(0.08), label='95% CrI')
    ax.fill_between(time_steps, credible_interval_90[0], credible_interval_90[1], color=blues(0.25), label='90% CrI')
    ax.fill_between(time_steps, credible_interval_75[0], credible_interval_75[1], color=blues(0.43), label='75% CrI')
    ax.fill_between(time_steps, credible_interval_50[0], credible_interval_50[1], color=blues(0.75), label='50% CrI')

    # Plot the median line
    ax.plot(time_steps, median_values, color=col2, lw=2.5, label='Median')

    # Add grid, legend, and ticks
    
    ax.grid(True, which='both', linestyle='-', linewidth=0.8)
    
    # Apply date formatting if Date is provided
    if Date is not None:
        ax.minorticks_on()
        # Set major ticks at the start of each month and format them
        ax.xaxis.set_major_locator(MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(DateFormatter('%b %y'))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # Set minor ticks to the 15th of each month
        ax.xaxis.set_minor_locator(DayLocator(bymonthday=16))  # Minor tick on 15th of each month
        
        # Set minor tick grid lines
        ax.grid(True, which='minor', linestyle='--', linewidth=0.4)

    # Add legend
    # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)

