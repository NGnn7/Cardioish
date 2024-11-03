import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


# Function for calculating causal connectivity
def calculate_causal_connectivity(features, y):
    num_regions = features.shape[1] // 12  # Assuming features split into 12 regions
    causal_matrix = np.zeros((12, 12))

    for i in range(12):
        for j in range(12):
            if i != j:
                # Select features for regions i and j only for causal influence calculation
                region_i_features = features[:, i * num_regions:(i + 1) * num_regions]
                region_j_features = features[:, j * num_regions:(j + 1) * num_regions]

                # Fit a KNN model to approximate causal influence (can replace with Granger causality or similar in practice)
                knn = KNeighborsClassifier(n_neighbors=1)
                skf = StratifiedKFold(n_splits=10)
                score_i_to_j = np.mean(cross_val_score(knn, region_i_features, y, cv=skf))

                # Causal influence is recorded in the causal_matrix
                causal_matrix[i, j] = score_i_to_j

    return causal_matrix


# Load data and extract features with tpat
def load_and_preprocess_data():
    X = []
    y = []
    files = ['file1.mat', 'file2.mat']  # Replace with actual .mat filenames

    for file in files:
        data = loadmat(file)
        y_value = int(file[:2])  # Assuming label is in filename
        y.append(y_value)
        signal = data['sinyal']
        X.append(tpat(signal))

    X = np.array(X)
    y = np.array(y)

    # Normalize X
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y


# Visualize the causal matrix
def plot_causal_matrix(causal_matrix, labels):
    plt.figure(figsize=(10, 8))
    plt.imshow(causal_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Causal Connectivity Transition Matrix")
    plt.xlabel("Source Region")
    plt.ylabel("Target Region")
    plt.xticks(ticks=np.arange(12), labels=labels, rotation=45)
    plt.yticks(ticks=np.arange(12), labels=labels)
    plt.show()


# Define Cardioish symbols and explanations
cardioish = ["Ld1", "Ld2", "Ld3", "AVR", "AVL", "AVF", "V1S", "V2S", "V3A", "V4A", "V5L", "V6L"]

# Main Execution
X, y = load_and_preprocess_data()

# Select features using NCA
mdl = SelectFromModel(KNeighborsClassifier(n_neighbors=1), threshold='mean').fit(X, y)
X_selected = mdl.transform(X)

# Calculate causal connectivity matrix
causal_matrix = calculate_causal_connectivity(X_selected, y)

# Plot the causal connectivity matrix
plot_causal_matrix(causal_matrix, cardioish)

# Entropy Calculation for each symbol based on usage in the matrix
entropy_values = -np.nansum(causal_matrix * np.log2(causal_matrix + np.finfo(float).eps), axis=1)
print("Entropy values for each region:", dict(zip(cardioish, entropy_values)))