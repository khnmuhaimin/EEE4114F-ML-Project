import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

confusion_matrix = np.array([
    [289, 14, 10, 5, 0, 0],
    [3, 289, 3, 6, 0, 0],
    [7, 31, 267, 13, 1, 0],
    [2, 0, 0, 313, 0, 0],
    [0, 0, 0, 0, 340, 0],
    [0, 0, 0, 0, 0, 321]
])

labels = ["dws", "ups", "wlk", "jog", "std", "sit"]

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, cbar=False)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()