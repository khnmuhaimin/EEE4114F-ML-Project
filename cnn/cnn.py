import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

from data.reader import get_data
from windower.windower import SAMPLES_PER_WINDOW, split_indices, split_indices_by_users

# Fixed compute_acceleration_magnitude to use existing data
def compute_acceleration_magnitude(data: pd.DataFrame) -> np.array:
    sum_of_squares = np.sum(
        [
            np.square(data["userAcceleration.x"]),
            np.square(data["userAcceleration.y"]),
            np.square(data["userAcceleration.z"])
        ],
        axis=0
    )
    return np.sqrt(sum_of_squares)

def compute_angular_velocity_magnitude(data: pd.DataFrame) -> np.array:
    sum_of_squares = np.sum(
        [
            np.square(data["rotationRate.x"]),
            np.square(data["rotationRate.y"]),
            np.square(data["rotationRate.z"])
        ],
        axis=0
    )
    return np.sqrt(sum_of_squares)

# Load data
data = get_data()

# Preprocessing
simplified = data.drop(columns=["weight", "height", "age", "gender", "trial", "gravity.x", "gravity.y", "gravity.z", "act", "id"], errors="ignore")

# Compute acceleration magnitude
simplified["acceleration"] = compute_acceleration_magnitude(data)
simplified["rotationRate"] = compute_angular_velocity_magnitude(data)

# Create circular features for angles
simplified["sin_roll"] = np.sin(simplified["attitude.roll"])
simplified["cos_roll"] = np.cos(simplified["attitude.roll"])
simplified["sin_yaw"] = np.sin(simplified["attitude.yaw"])
simplified["cos_yaw"] = np.cos(simplified["attitude.yaw"])
simplified["sin_pitch"] = np.sin(simplified["attitude.pitch"])
simplified["cos_pitch"] = np.cos(simplified["attitude.pitch"])
simplified = simplified.drop(columns=["attitude.roll", "attitude.pitch", "attitude.yaw"])

# Get window indices
split_idx = split_indices_by_users()  # Assuming this returns [train_indices, test_indices]
window_start_indices = np.concatenate(split_idx)  # Combine all window starts

print(simplified.head())

# Create windows
windows = []
for start in window_start_indices:
    window = simplified.iloc[start : start + SAMPLES_PER_WINDOW]
    windows.append(window.to_numpy())
X = np.stack(windows)  # shape: (num_windows, window_size, num_features)

# Get labels for each window (using start index)
y = data.iloc[window_start_indices]["act"]

# Convert string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalization - should be done per feature using training data only
# First separate train/test windows
train_indices = split_idx[0]  # First split is train
test_indices = split_idx[1]   # Second split is test

# Calculate normalization parameters using training data only
train_windows = X[[i for i, start in enumerate(window_start_indices) if start in set(train_indices)]]
train_data = train_windows.reshape(-1, X.shape[2])  # Flatten windows

scaler = StandardScaler()
scaler.fit(train_data)

# Apply normalization to all windows
for i in range(X.shape[0]):
    X[i] = scaler.transform(X[i])

# Model architecture - improved
num_classes = y_categorical.shape[1]
model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', 
           input_shape=(SAMPLES_PER_WINDOW, X.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Prepare train/test indices
train_idx = [i for i, start in enumerate(window_start_indices) if start in set(train_indices)]
test_idx = [i for i, start in enumerate(window_start_indices) if start in set(test_indices)]
print(len(train_idx), len(test_idx))

# Train using proper train/test split
# history = model.fit(
#     X[train_idx], y_categorical[train_idx],
#     epochs=20,
#     batch_size=32,
#     validation_data=(X[test_idx], y_categorical[test_idx])
# )
# 
# model.save('cnn_model.keras')



# Load model
model = load_model('cnn_model.keras')

# Load your test data (X_test, y_test)
# Ensure y_test is in class index format (e.g., [2, 0, 1, ...])
# If it's one-hot, convert using np.argmax
X_test = X[test_idx]
y_true = np.argmax(y_categorical[test_idx], axis=1)

# Get model predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

print(cm)