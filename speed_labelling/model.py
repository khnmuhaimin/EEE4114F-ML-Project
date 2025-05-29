import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.reader import ACT_LABELS


HIGH_SPEED_ACTIONS = ["dws","ups", "wlk", "jog"]
LOW_SPEED_ACTIONS = ["std", "sit"]
HIGH_MOTION_ACTION_LABELS = [i for i in range(ACT_LABELS) if ACT_LABELS[i] in HIGH_SPEED_ACTIONS]
LOW_SPEED_ACTION_LABELS = [i for i in range(ACT_LABELS) if ACT_LABELS[i] in LOW_SPEED_ACTIONS]


def compute_acceleration_magnitude_to_speed_label(data: pd.DataFrame) -> pd.DataFrame:
    windows = list(data["window"].unique())
    sum_of_squares = np.sum(
            [
                np.square(data["userAcceleration.x"]),
                np.square(data["userAcceleration.y"]),
                np.square(data["userAcceleration.z"])
            ],
            axis=0,
            dtype=np.float32
        )
    acceleration_magnitude = np.sqrt(sum_of_squares, dtype=np.float32)
    acceleration_std = np.empty_like(windows, dtype=np.float32)
    high_motion = np.empty_like(windows, dtype=np.uint8)
    for i, window in enumerate(windows):
        acceleration_std[i] = np.std(acceleration_magnitude[data["window"] == window])
        high_motion[i] = data[data["window"] == window]["action"][0] in HIGH_MOTION_ACTION_LABELS
    result = pd.DataFrame.from_dict({
        "acceleration_std": acceleration_std,
        "high_motion": high_motion
    })
    return result


def compute_ln_acceleration_std(data: pd.DataFrame) -> pd.DataFrame:
    data["ln_acceleration"] = np.log(data["acceleration_std"])
    return data


def plot_ln_acceleration(data: pd.DataFrame) -> None:
    plt.scatter(data["ln_acceleration"], np.zeros(len(data)), c=data["high_motion"], cmap='viridis')
    plt.yticks([])  # Hide y-axis since it's meaningless here
    plt.show()


