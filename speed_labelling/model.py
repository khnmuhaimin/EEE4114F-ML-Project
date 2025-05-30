import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from data.reader import ACT_LABELS, get_data
from function_cache.function_cache import DEFAULT_CACHE
from windower.windower import SAMPLES_PER_WINDOW, train_window_start_indices


HIGH_MOTION_ACTIONS = ["dws","ups", "wlk", "jog"]
LOW_MOTION_ACTIONS = ["std", "sit"]
HIGH_MOTION_ACTION_LABELS = [i for i in range(len(ACT_LABELS)) if ACT_LABELS[i] in HIGH_MOTION_ACTIONS]
LOW_SPEED_ACTION_LABELS = [i for i in range(len(ACT_LABELS)) if ACT_LABELS[i] in LOW_MOTION_ACTIONS]


@DEFAULT_CACHE.memoize()
def compute_acceleration_magnitude() -> np.array:
    data = get_data()
    sum_of_squares = np.sum(
            [
                np.square(data["userAcceleration.x"]),
                np.square(data["userAcceleration.y"]),
                np.square(data["userAcceleration.z"])
            ],
            axis=0,
            dtype=np.float32
        )
    return np.sqrt(sum_of_squares)


@DEFAULT_CACHE.memoize()
def compute_acceleration_std() -> pd.DataFrame:
    acceleration = compute_acceleration_magnitude()
    window_start_indices =  train_window_start_indices()
    acceleration_std = np.empty_like(window_start_indices)
    for i, window_index in enumerate(window_start_indices):
        acceleration_std[i] = np.std(acceleration[window_index:window_index+SAMPLES_PER_WINDOW])
    return acceleration_std


@DEFAULT_CACHE.memoize()
def compute_high_motion() -> pd.DataFrame:
    data = get_data()
    window_start_indices =  train_window_start_indices()
    high_motion = np.empty_like(window_start_indices)
    for i, window_index in enumerate(window_start_indices):
        high_motion[i] = data.at[window_index, "act"] in HIGH_MOTION_ACTION_LABELS
    return high_motion


def plot_ln_acceleration() -> None:
    high_motion = compute_high_motion()
    acceleration_std = compute_acceleration_std()
    plt.scatter(acceleration_std, np.zeros(len(acceleration_std)), c=high_motion, cmap='viridis')
    plt.yticks([])  # Hide y-axis since it's meaningless here
    plt.show()


@DEFAULT_CACHE.memoize()
def get_speed_level_model():
    high_motion = compute_high_motion()
    acceleration_std = compute_acceleration_std()
    model = LogisticRegression()
    model.fit(acceleration_std.reshape(-1, 1), high_motion)
    return model