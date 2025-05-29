import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data import reader

data = reader.get_data()

# Uncomment this to take a look at the data
print(f"[INFO]: Columns: {', '.join(data.columns)}")
# there are 24 IDs (ranging from 0 to 23)
print(f"[INFO]: Unique IDs: {', '.join(map(str, data['id'].unique()))}")
# there are 15 trials (ranging from 1 to 16 number 10 skipped for some reason)
print(f"[INFO]: Unique trials: {', '.join(map(str, data['trial'].unique()))}")
# there are 6 actions (ranging from 0 to 5)
print(f"[INFO]: Unique actions: {', '.join(map(str, data['act'].unique()))}")
print(f"[INFO]: Number of data points: {len(data)}")

# Uncomment to show an example plot
# choosing a trials and a participant ID gives a unique timeseries
# action = data[(data["id"] == 0) & (data["trial"] == 1)]
# sum_of_squares = np.square(action["userAcceleration.x"]) + np.square(action["userAcceleration.y"]) + np.square(action["userAcceleration.z"])
# plt.plot(np.sqrt(sum_of_squares))
# plt.show()


# Uncomment to show bounds of the dataset
# print(np.min(data["attitude.roll"]), np.min(data["attitude.pitch"]), np.min(data["attitude.yaw"]))
# print(np.max(data["attitude.roll"]), np.max(data["attitude.pitch"]), np.max(data["attitude.yaw"]))
# print(np.min(data["gravity.x"]), np.min(data["gravity.y"]), np.min(data["gravity.z"]))
# print(np.max(data["gravity.x"]), np.max(data["gravity.y"]), np.max(data["gravity.z"]))
# print(np.min(data["rotationRate.x"]), np.min(data["rotationRate.y"]), np.min(data["rotationRate.z"]))
# print(np.max(data["rotationRate.x"]), np.max(data["rotationRate.y"]), np.max(data["rotationRate.z"]))
# print(np.min(data["userAcceleration.x"]), np.min(data["userAcceleration.y"]), np.min(data["userAcceleration.z"]))
# print(np.max(data["userAcceleration.x"]), np.max(data["userAcceleration.y"]), np.max(data["userAcceleration.z"]))

