"""

CONCATENATED MODEL

"""
# Import required libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Flatten
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
import random

# Set random seeds for reproducibility
random.seed(1)
np.random.seed(0)
tf.random.set_seed(0)

# Define a function to load and preprocess the data
def load_and_preprocess_data(filepath, labels_file):
    # Read the data into a Pandas DataFrame
    df = pd.read_csv(filepath)
    
    # Convert string representations of lists to actual lists
    df['GNN_Embedding'] = df['GNN_Embedding'].apply(eval)
    df['CNN_Embedding'] = df['CNN_Embedding'].apply(eval)

    # Read labels into a DataFrame
    labels_df = pd.read_csv(labels_file)
    labels_list = labels_df['Scent'].tolist()

    # Binarize the labels
    mlb = MultiLabelBinarizer(classes=labels_list)
    labels = mlb.fit_transform(df['odor_labels_filtered'].str.strip("[]").str.replace("'", "").str.split(", "))
    Y = pd.DataFrame(labels, columns=mlb.classes_)

    # Create dataframes for GNN and CNN embeddings
    gnn = pd.DataFrame(df['GNN_Embedding'].tolist())
    cnn = pd.DataFrame(df['CNN_Embedding'].tolist())

    # Concatenate GNN and CNN dataframes and fill NA values with 0
    X = pd.concat([gnn, cnn], axis=1)
    X = X.fillna(0)

    return X, Y

# Load the data
X, Y = load_and_preprocess_data("embeddings&labels.csv", "scentClassescnn.csv")

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Build the CNN model
embedding_dim = 128
cnn_model = tf.keras.Sequential()
cnn_model.add(tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)))  # Reshape the input
cnn_model.add(LeakyReLU(alpha=0.01))  # Add Leaky ReLU activation
# Add a series of 1D convolutional layers with increasing dilation rates
cnn_model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=2, padding='causal'))
cnn_model.add(LeakyReLU(alpha=0.01))
cnn_model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=4, padding='causal'))
cnn_model.add(LeakyReLU(alpha=0.01))
cnn_model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=8, padding='causal'))
cnn_model.add(LeakyReLU(alpha=0.01))
cnn_model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=16, padding='causal'))
cnn_model.add(LeakyReLU(alpha=0.01))
cnn_model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=32, padding='causal'))
cnn_model.add(LeakyReLU(alpha=0.01))
cnn_model.add(tf.keras.layers.Conv1D(128, 1))  # Additional 1D convolutional layer
cnn_model.add(LeakyReLU(alpha=0.01))
cnn_model.add(tf.keras.layers.Conv1D(embedding_dim, 1))  # Final 1D convolutional layer
cnn_model.add(LeakyReLU(alpha=0.01))
cnn_model.add(Flatten())  # Flatten the tensor
cnn_model.add(Dense(112, activation='relu'))  # Fully connected layer
cnn_model.add(Dense(Y.shape[1], activation='sigmoid'))  # Output layer

# Compile the model
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = cnn_model.fit(X_train, Y_train, epochs=30, batch_size=32)

# Make predictions on the test set
Y_pred = cnn_model.predict(X_test)

# Calculate AUC-ROC scores for each label
auc_scores = {}
valid_indices = []
for i, label in enumerate(Y_test.columns):
    unique_classes = np.unique(Y_test.iloc[:, i])
    if len(unique_classes) == 1:
        print(f"Skipping AUC ROC calculation for label {label} as it has only one unique class in the test set.")
        continue
    auc_scores[label] = roc_auc_score(Y_test.iloc[:, i], Y_pred[:, i])
    valid_indices.append(i)

# Calculate and print AUC-ROC scores
if len(auc_scores) > 0:
    mean_auc = np.mean(list(auc_scores.values()))
    median_auc = np.median(list(auc_scores.values()))
    weighted_auc = np.average(list(auc_scores.values()), weights=Y_test.iloc[:, valid_indices].sum(axis=0))
    micro_auc = roc_auc_score(Y_test.iloc[:, valid_indices].values.ravel(), Y_pred[:, valid_indices].ravel())
    macro_auc = np.mean(list(auc_scores.values()))
    print("Per-label AUC ROC scores:", auc_scores)
    print("Mean AUC ROC:", mean_auc)
    print("Median AUC ROC:", median_auc)
    print("Weighted AUC ROC:", weighted_auc)
    print("Micro-average AUC ROC:", micro_auc)
    print("Macro-average AUC ROC:", macro_auc)
else:
    print("No valid AUC ROC scores could be calculated.")
