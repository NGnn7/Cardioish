import os
import numpy as np
from scipy.io import loadmat
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# TPAT feature extractor function
def tpat(signal):
    ct = []
    for row in signal:
        ix = np.argsort(row)[::-1]
        ct.extend(ix[:12])

    tt = np.zeros((12, 12))
    for i in range(len(ct) - 1):
        tt[ct[i], ct[i + 1]] += 1

    for i in range(12):
        tt[i, :] /= (np.sum(tt[i, :]) + np.finfo(float).eps)

    return tt.flatten()


# Load all .mat files and extract features
X = []
y = []
for file in os.listdir():
    if file.endswith(".mat"):
        data = loadmat(file)
        y_value = int(file[:2])
        y.append(y_value)
        signal = data['sinyal']
        X.append(tpat(signal))

X = np.array(X)
y = np.array(y)

# Normalize X
X = (X - X.min()) / (X.max() - X.min() + np.finfo(float).eps)

# Feature Selection using NCA with weights
mdl = SelectFromModel(KNeighborsClassifier(n_neighbors=1), threshold='mean').fit(X, y)
feature_weights = mdl.estimator_.feature_importances_
sorted_indices = np.argsort(feature_weights)[::-1]

# Cumulative weight sum and threshold
cumulative_weights = np.cumsum(feature_weights[sorted_indices]) / np.sum(feature_weights[sorted_indices])
threshold = 0.85
startIndex = np.argmax(cumulative_weights >= threshold) if np.any(cumulative_weights >= threshold) else 10
t1 = 0.9999999
stopIndex = np.argmax(cumulative_weights >= t1) if np.any(cumulative_weights >= t1) else X.shape[1]

# Best features and kNN training
best_loss = float('inf')
loss_list = []
for ts in range(stopIndex - startIndex):
    selected_features = X[:, sorted_indices[:startIndex + ts]]
    knn = KNeighborsClassifier(n_neighbors=1, metric='cityblock', weights='uniform')
    skf = StratifiedKFold(n_splits=10)
    loss = 1 - np.mean(cross_val_score(knn, selected_features, y, cv=skf))
    loss_list.append(loss)
    if loss < best_loss:
        best_loss = loss
        best_features = selected_features

# TPAT KNN Classifier configuration
max_accuracy = 0
for _ in range(100):
    classifier_accuracy = []
    distance_metrics = ["cityblock", "euclidean", "cosine"]
    weight_options = ["uniform", "distance"]

    for metric in distance_metrics:
        for weight in weight_options:
            for neighbors in range(1, 11):
                knn = KNeighborsClassifier(n_neighbors=neighbors, metric=metric, weights=weight)
                skf = StratifiedKFold(n_splits=10)
                cv_accuracy = np.mean(cross_val_score(knn, best_features, y, cv=skf))
                classifier_accuracy.append(cv_accuracy)

    best_config_accuracy = max(classifier_accuracy)
    if best_config_accuracy > max_accuracy:
        max_accuracy = best_config_accuracy

# Cardioish symbol explanations and display generation
cardioish = ["Ld1", "Ld2", "Ld3", "AVR", "AVL", "AVF", "V1S", "V2S", "V3S", "V4S", "V5S", "V6S"]
explanations = [
    "Ld1: Monitors electrical activity between the left atrium and left ventricle, often used for lateral wall assessment.",
    "Ld2: Monitors heart rhythm and P-wave, important for arrhythmia detection.",
    "Ld3: Used in inferior MI diagnosis, monitors electrical activity in the lower heart region.",
    "AVR: Shows electrical activity of the right ventricle and heart base.",
    "AVL: Evaluates upper lateral wall of the left ventricle, used in lateral MI diagnosis.",
    "AVF: Monitors electrical activity in the inferior wall of the left ventricle, essential for inferior MI diagnosis.",
    "V1S: Monitors right ventricular and septal activity, important in diagnosing right ventricular overload and septal MI.",
    "V2S: Important in anterior and septal MI diagnosis, shows activities over the anterior and septal walls.",
    "V3A: Monitors anterior MI and septum activities, shows damage in the front wall of the heart.",
    "V4A: Monitors electrical activity in the lower part of the left ventricle, used in anterior MI diagnosis.",
    "V5L: Monitors lateral wall of the left ventricle, plays a critical role in lateral MI diagnosis.",
    "V6L: Monitors electrical activity in the lower-lateral wall of the left ventricle, used in lateral MI diagnosis."
]

# Generate transition matrix and entropy
hlead = np.zeros(12)
for index in range(len(best_features[0])):
    lead_index = sorted_indices[index] % 12
    hlead[lead_index] += 1

tt_matrix = np.zeros((12, 12))
for i in range(len(hlead) - 1):
    tt_matrix[int(hlead[i]), int(hlead[i + 1])] += 1

entropy = -np.sum((hlead / np.sum(hlead)) * np.log2(hlead / np.sum(hlead) + np.finfo(float).eps))

# Visualization of transition matrix
plt.figure(figsize=(10, 8))
plt.imshow(tt_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Causal Connectivity Transition Matrix")
plt.show()

print("Entropy:", entropy)