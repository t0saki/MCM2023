import numpy as np
import pandas as pd

# Load the data from forecast.csv
data = pd.read_csv('forecast.csv')

# Add 5% noise to the predicted_mean column
noisy_predicted_mean = data['predicted_mean'] * np.random.normal(1, 0.02, size=len(data))

# Replace the predicted_mean column with the noisy version
data['predicted_mean'] = noisy_predicted_mean

# Save the new data to a new CSV file
data.to_csv('noisy_forecast.csv', index=False)