import random
import numpy as np
import pandas as pd
from collections import defaultdict

from data.reader import get_data
from function_cache.function_cache import DEBUG_CACHE, DEFAULT_CACHE

"""
Let's think

The data should already be sorted.
So that means finding suitable window indices should be as simple as finding the start index of each wnidow
i can label the samples for each trial
then i need to look for samples where sample % 250 == 0
i also need to delete the first and last 5 seconds of each trial

after deleting, i need to see which will end up having the most data
then i need to oversample some actions so that

for validation and test sets, i need to sample data in a way that's representative of the labels. this should be done first.
then out of the remaining windows, i should delete the first and last 5 seconds of each trial
then i should oversample
then i should store

i realized that i want all samples to be represented equally across all sets

that means that i should first determine the number of max number of windows out of all labels
then i should oversample the rest so that i have an equal amount of each label.

i also want to convert some values
"""

SAMPLES_PER_WINDOW = 250

@DEFAULT_CACHE.memoize()
def number_of_windows_with_no_overlap_per_action() -> dict[np.uint8, int]:
    data = get_data()
    unique_trials = data["trial"].unique()
    unique_ids = data["id"].unique()
    actions_to_windows = defaultdict(int)

    for trial in unique_trials:
        for id in unique_ids:
            valid_samples = data[(data["trial"] == trial) & (data["id"] == id)]
            windows = len(valid_samples) // SAMPLES_PER_WINDOW
            action = valid_samples["act"].iloc[0]
            actions_to_windows[action] += windows

    return dict(actions_to_windows)


@DEFAULT_CACHE.memoize()
def max_number_of_windows_with_no_overlap_for_all_actions() -> int:
    actions_to_windows = number_of_windows_with_no_overlap_per_action()
    max_action = max(list(actions_to_windows.items()), key=lambda item: item[1])
    return max_action[1]


@DEFAULT_CACHE.memoize()
def number_of_timeseries_per_action() -> dict[np.uint8, int]:
    data = get_data()
    unique_trials = data["trial"].unique()
    unique_ids = data["id"].unique()
    trials_to_actions = defaultdict(int)
    for trial in unique_trials:
        for id in unique_ids:
            first_sample_index = data.index[((data["trial"] == trial) & (data["id"] == id))][0]
            action = data["act"].iloc[first_sample_index]
            trials_to_actions[action] += 1
    return dict(trials_to_actions)



"""
I need to do some more thinking

Each trial (aka file) is uniquely identified using an participant ID and a trial.
The number of windows from each trial should be proportional to their length.
That means I need to know the length of all trails of a certain action.
I also need to know the length of a specific trial

"""

@DEFAULT_CACHE.memoize()
def number_of_samples_per_timeseries() -> dict[np.uint8, dict[np.uint8, int]]:
    """
    Format of result:
    { trial: { id: num_samples}}
    """
    data = get_data()
    unique_trials = data["trial"].unique()
    unique_ids = data["id"].unique()
    samples_per_time_series = {}
    for trial in unique_trials:
        samples_per_time_series[trial] = {}
        for _id in unique_ids:
            samples_per_time_series[trial][_id] = ((data["trial"] == trial) & (data["id"] == _id)).sum()
    return samples_per_time_series


@DEFAULT_CACHE.memoize()
def number_of_samples_per_action() -> dict[np.uint8, int]:
    data = get_data()
    unique_actions = data["act"].unique()
    samples_per_action = {}
    for action in unique_actions:
        number_of_samples = (data["act"] == action).sum()
        samples_per_action[action] = number_of_samples
    return samples_per_action




"""
I need to do some thinking

I want to know how many windows i want from each timeseries.
The number of windows per timeseries must be proportional the that timeseries length / length of all timeseries with that action

I need to know the number of samples per action which i already have a function for a
"""

@DEFAULT_CACHE.memoize()
def action_per_timeseries() -> dict[np.uint8, dict[np.uint8, int]]:
    data = get_data()
    unique_trials = data["trial"].unique()
    unique_ids = data["id"].unique()
    _action_per_time_series = {}
    for trial in unique_trials:
        _action_per_time_series[trial] = {}
        for id in unique_ids:
            first_index = data.index[(data["trial"] == trial) & (data["id"] == id)][0]
            _action_per_time_series[trial][id] = data["act"].iloc[first_index]
    return _action_per_time_series


@DEFAULT_CACHE.memoize()
def number_of_windows_per_timeseries():
    data = get_data()
    samples_per_time_series = number_of_samples_per_timeseries()
    samples_per_action = number_of_samples_per_action()
    _action_per_timeseries = action_per_timeseries()
    required_windows_per_action = max_number_of_windows_with_no_overlap_for_all_actions()
    windows_per_timeseries = {}
    unique_trials = data["trial"].unique()
    unique_ids = data["id"].unique()
    for trial in unique_trials:
        windows_per_timeseries[trial] = {}
        for id in unique_ids:
            action = _action_per_timeseries[trial][id]
            ratio_of_windows = samples_per_time_series[trial][id] / samples_per_action[action]
            windows_per_timeseries[trial][id] = int(np.floor(ratio_of_windows * required_windows_per_action))
    return windows_per_timeseries
    


@DEFAULT_CACHE.memoize()
def start_index_per_timeseries() -> dict[np.uint8, dict[np.uint8, int]]:
    data = get_data()
    unique_trials = data["trial"].unique()
    unique_ids = data["id"].unique()
    _start_indices_per_timeseries = {}
    for trial in unique_trials:
        _start_indices_per_timeseries[trial] = {}
        for _id in unique_ids:
            first_index = data.index[(data["trial"] == trial) & (data["id"] == _id)][0]
            _start_indices_per_timeseries[trial][_id] = first_index
    return _start_indices_per_timeseries


@DEFAULT_CACHE.memoize()
def window_start_indices_per_timeseries() -> dict[np.uint8, dict[np.uint8, list[int]]]:
    _start_indices_per_timeseries = start_index_per_timeseries()
    samples_per_timeseries = number_of_samples_per_timeseries()
    windows_per_timeseries = number_of_windows_per_timeseries()
    _window_start_indices_per_timeseries = {}
    for trial, inner_dict in _start_indices_per_timeseries.items():
        _window_start_indices_per_timeseries[trial] = {}
        for _id, start_index in inner_dict.items():
            window_start_indices = np.linspace(
                0,
                samples_per_timeseries[trial][_id] - SAMPLES_PER_WINDOW,
                num=windows_per_timeseries[trial][_id],
                dtype=int
            )
            window_start_indices = [i + start_index for i in window_start_indices]
            _window_start_indices_per_timeseries[trial][_id] = window_start_indices
    return _window_start_indices_per_timeseries


@DEFAULT_CACHE.memoize()
def window_start_indices_per_action() -> dict[np.uint8, list[int]]:
    _window_start_indices_per_timeseries = window_start_indices_per_timeseries()
    _action_per_timeseries = action_per_timeseries()
    _window_starts_indices_per_action = defaultdict(list)
    for trial, inner in _window_start_indices_per_timeseries.items():
        for _id, window_start_indices in inner.items():
            action = _action_per_timeseries[trial][_id]
            _window_starts_indices_per_action[action] += window_start_indices
    return dict(_window_starts_indices_per_action)


@DEFAULT_CACHE.memoize()
def split_indices():
    random.seed(1234)
    train, val, test = [], [], []
    for action, window_start_indices in window_start_indices_per_action().items():
        window_start_indices = window_start_indices.copy()
        random.shuffle(window_start_indices)
        n = len(window_start_indices)
        n_train = int(n * 0.6)
        n_val = int((n - n_train) * 0.5)

        train += window_start_indices[:n_train]
        val += window_start_indices[n_train:n_train + n_val]
        test += window_start_indices[n_train + n_val:]
    return train, val, test


def train_window_start_indices():
    return split_indices()[0]


def validate_window_start_indices():
    return split_indices()[1]

def test_window_start_indices():
    return split_indices()[2]