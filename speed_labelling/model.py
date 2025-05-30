import sys
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

WALK_ACTION_LABEL = ACT_LABELS.index("wlk")
JOG_ACTION_LABEL = ACT_LABELS.index("jog")
HORIZONTAL_ACTION_LABELS = [i for i in range(len(ACT_LABELS)) if ACT_LABELS[i] in ["wlk", "jog"]]


@DEFAULT_CACHE.memoize(tag="SPEED_MODEL")
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


@DEFAULT_CACHE.memoize(tag="SPEED_MODEL")
def compute_acceleration_std() -> pd.DataFrame:
    acceleration = compute_acceleration_magnitude()
    window_start_indices =  train_window_start_indices()
    acceleration_std = np.empty_like(window_start_indices, dtype=np.float32)
    for i, window_index in enumerate(window_start_indices):
        acceleration_std[i] = np.std(acceleration[window_index:window_index+SAMPLES_PER_WINDOW])
    return acceleration_std


@DEFAULT_CACHE.memoize(tag="SPEED_MODEL")
def compute_high_motion() -> pd.DataFrame:
    data = get_data()
    window_start_indices =  train_window_start_indices()
    high_motion = np.empty_like(window_start_indices)
    for i, window_index in enumerate(window_start_indices):
        high_motion[i] = data.at[window_index, "act"] in HIGH_MOTION_ACTION_LABELS
    return high_motion


def plot_acceleration_std_histograms() -> None:
    high_motion = compute_high_motion()
    acceleration_std = compute_acceleration_std()
    all_data = acceleration_std
    bins = np.histogram_bin_edges(all_data, bins=50)
    plt.hist(acceleration_std[high_motion == 1], bins=bins, alpha=0.5, label='High Motion')
    plt.hist(acceleration_std[high_motion == 0], bins=bins, alpha=0.5, label='Low Motion')
    plt.legend()
    plt.show()


@DEFAULT_CACHE.memoize(tag="SPEED_MODEL")
def get_speed_level_model():
    high_motion = compute_high_motion()
    acceleration_std = compute_acceleration_std()
    model = LogisticRegression()
    model.fit(acceleration_std.reshape(-1, 1), high_motion)
    return model


@DEFAULT_CACHE.memoize(tag="HORIZONTAL_MODEL")
def compute_vertical_acceleration() -> np.array:
    data = get_data()
    return np.sum(
        [
            np.prod([data["userAcceleration.x"], data["gravity.x"]], axis=0),
            np.prod([data["userAcceleration.y"], data["gravity.y"]], axis=0),
            np.prod([data["userAcceleration.z"], data["gravity.z"]], axis=0)
        ],
        axis=0
    )


@DEFAULT_CACHE.memoize(tag="HORIZONTAL_MODEL")
def train_window_start_indices_for_horizontal_motion_training():
    data = get_data()
    window_start_indices = train_window_start_indices()
    vertical_or_non_vertical_motion_action = lambda row: data.at[row, "act"] in HORIZONTAL_ACTION_LABELS
    return [i for i in window_start_indices if vertical_or_non_vertical_motion_action(i)]


@DEFAULT_CACHE.memoize(tag="HORIZONTAL_MODEL")
def compute_vertical_acceleration_std():
    vertical_acceleration = compute_vertical_acceleration()
    window_start_indices =  train_window_start_indices_for_horizontal_motion_training()
    acceleration_std = np.empty_like(window_start_indices, dtype=np.float32)
    for i, window_index in enumerate(window_start_indices):
        acceleration_std[i] = np.std(vertical_acceleration[window_index:window_index+SAMPLES_PER_WINDOW])
    return acceleration_std


@DEFAULT_CACHE.memoize(tag="HORIZONTAL_MODEL")
def compute_is_walking() -> pd.DataFrame:
    data = get_data()
    window_start_indices = train_window_start_indices_for_horizontal_motion_training()
    is_walking = np.empty_like(window_start_indices)
    for i, window_index in enumerate(window_start_indices):
        is_walking[i] = data.at[window_index, "act"] == WALK_ACTION_LABEL
    return is_walking


def plot_horizontal_motion_histograms() -> None:
    is_walking = np.array(compute_is_walking())
    acceleration_std = np.array(compute_vertical_acceleration_std())
    bins = np.histogram_bin_edges(acceleration_std, bins=50)
    plt.hist(acceleration_std[is_walking == 1], bins=bins, alpha=0.5, label='Walking')
    plt.hist(acceleration_std[is_walking == 0], bins=bins, alpha=0.5, label='Jogging')
    plt.legend()
    plt.show()



# plot_acceleration_std_histograms()
plot_horizontal_motion_histograms()


