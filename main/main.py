import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data import reader

data = reader.get_data()

# Uncomment this to take a look at the data
print(f"[INFO]: Columns: {', '.join(data.columns)}")
# there is 24 IDs (ranging from 0 to 23)
print(f"[INFO]: Unique IDs: {', '.join(map(str, data['id'].unique()))}")
# there is 15 trials (ranging from 1 to 16 number 10 skipped for some reason)
print(f"[INFO]: Unique trials: {', '.join(map(str, data['trial'].unique()))}")
# there is 6 IDs (ranging from 0 to 5)
print(f"[INFO]: Unique actions: {', '.join(map(str, data['act'].unique()))}")
print(f"[INFO]: Number of data points: {len(data)}")


# choosing a trials and a participant ID gives a unique timeseries
action = data[(data["id"] == 0) & (data["trial"] == 1)]
sum_of_squares = np.square(action["userAcceleration.x"]) + np.square(action["userAcceleration.y"]) + np.square(action["userAcceleration.z"])
plt.plot(np.sqrt(sum_of_squares))
plt.show()