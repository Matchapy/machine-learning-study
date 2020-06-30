#%% imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% read file
file_name = 'HumidTempComfort.csv'
data = pd.read_csv(file_name)
data.head()

#%% Plot data
plt.scatter(data['Humidity'],data['Temp'],c=data['Comfortability'])
plt.colorbar()


# %%
