"""

PCA- Visualisation of all three models

"""


#Importing required libraries
import pyrfume
import tensorflow as tf
import numpy as np
import seaborn as sns
import jax
import jax.numpy as jnp
import pandas as pd
import rdkit, rdkit.Chem, rdkit.Chem.rdDepictor, rdkit.Chem.Draw
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MultiLabelBinarizer
import haiku as hk
from sklearn.model_selection import train_test_split
import optax
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Flatten, Conv1D, Add
import warnings
import random


warnings.filterwarnings("ignore")
# Fix random seeds for reproducibility
random.seed(1)
np.random.seed(0)
tf.random.set_seed(0)


learning_rate = 1e-5
num_Dense_layers = 2
num_GNN_layers = 4
numEpochs = (
    1
)
steps_for_gradUpdate = 8
weights_stddevGNN = 1e-2
earlyStopping = True
earlyStopping_patience = 3
earlyStopping_minDelta = 0
regularizationStrength = 1e-6

node_feat_length = 256
message_feat_length = 256
graph_feat_length = 512
weights_stddevGNN = 0.01

scentdata = pd.read_csv("STRUCTURAL-SPECTRAL DATASET.csv")

# Perform an 80-20 split for train and test data
split_ratio = 0.8
split_point = int(len(scentdata) * split_ratio)
trainData = scentdata.iloc[:split_point]
testData = scentdata.iloc[split_point:]

# Shuffle the datasets
trainData = trainData.sample(frac=1, random_state=0).reset_index(drop=True)
testData = testData.sample(frac=1, random_state=0).reset_index(drop=True)

numMolecules = len(scentdata.odor_labels_filtered)
numClasses = 112 
scentClasses = pd.read_csv("scentClasses.csv")
scentClasses = scentClasses["Scent"].tolist()
moleculeScentList = []
for i in range(numMolecules):
    scentString = scentdata.odor_labels_filtered[i]
    temp = scentString.replace("[", "")
    temp = temp.replace("]", "")
    temp = temp.replace("'", "")
    temp = temp.replace(" ", "")
    scentList = temp.split(",")
    if "odorless" in scentList:
        scentList.remove("odorless")
    moleculeScentList.append(scentList)

numTrainMolecules = len(trainData.odor_labels_filtered)
moleculeScentList_train = []
for i in range(numTrainMolecules):
    scentString = trainData.odor_labels_filtered[i]
    temp = scentString.replace("[", "")
    temp = temp.replace("]", "")
    temp = temp.replace("'", "")
    temp = temp.replace(" ", "")
    scentList = temp.split(",")
    if "odorless" in scentList:
        scentList.remove("odorless")
    moleculeScentList_train.append(scentList)

numTestMolecules = len(testData.odor_labels_filtered)
moleculeScentList_test = []
for i in range(numTestMolecules):
    scentString = testData.odor_labels_filtered[i]
    temp = scentString.replace("[", "")
    temp = temp.replace("]", "")
    temp = temp.replace("'", "")
    temp = temp.replace(" ", "")
    scentList = temp.split(",")
    if "odorless" in scentList:
        scentList.remove("odorless")
    moleculeScentList_test.append(scentList)

# Generate graph data
def gen_smiles2graph(sml):
    m = rdkit.Chem.MolFromSmiles(sml)
    m = rdkit.Chem.AddHs(m)
    order_string = {
        rdkit.Chem.rdchem.BondType.SINGLE: 1,
        rdkit.Chem.rdchem.BondType.DOUBLE: 2,
        rdkit.Chem.rdchem.BondType.TRIPLE: 3,
        rdkit.Chem.rdchem.BondType.AROMATIC: 4,
    }
    N = len(list(m.GetAtoms()))
    nodes = np.zeros((N, node_feat_length))
    for i in m.GetAtoms():
        nodes[i.GetIdx(), i.GetAtomicNum()] = 1
        if i.IsInRing():
            nodes[i.GetIdx(), -1] = 1

    adj = np.zeros((N, N))
    for j in m.GetBonds():
        u = min(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        v = max(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        order = j.GetBondType()
        if order in order_string:
            order = order_string[order]
        else:
            raise Warning("Ignoring bond order" + order)
        adj[u, v] = 1
        adj[v, u] = 1
    adj += np.eye(N)
    return nodes, adj

# Create one-hot encoded label vectors
def createLabelVector(scentsList):
    labelVector = np.zeros(numClasses)
    for j in range(len(scentsList)):
        classIndex = scentClasses.index(scentsList[j])
        labelVector[classIndex] = 1
    return labelVector

# Generate training graphs
def generateGraphsTrain():
    for i in range(numTrainMolecules):
        graph = gen_smiles2graph(trainData.smiles[i])
        labels = createLabelVector(moleculeScentList_train[i])
        yield graph, labels

# Generate testing graphs
def generateGraphsTest():
    for i in range(numTestMolecules):
        graph = gen_smiles2graph(testData.smiles[i])
        labels = createLabelVector(moleculeScentList_test[i])
        yield graph, labels

# Generate all graphs
def generateGraphs():
    for i in range(numMolecules):
        graph = gen_smiles2graph(scentdata.smiles[i])
        labels = createLabelVector(moleculeScentList[i])
        yield graph, labels


# Create TensorFlow datasets
data = tf.data.Dataset.from_generator(
    generateGraphs,
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(
        (tf.TensorShape([None, node_feat_length]), tf.TensorShape([None, None])),
        tf.TensorShape([None]),
    ),
)


train_set = tf.data.Dataset.from_generator(
    generateGraphsTrain,
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(
        (tf.TensorShape([None, node_feat_length]), tf.TensorShape([None, None])),
        tf.TensorShape([None]),
    ),
)


test_set = tf.data.Dataset.from_generator(
    generateGraphsTest,
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(
        (tf.TensorShape([None, node_feat_length]), tf.TensorShape([None, None])),
        tf.TensorShape([None]),
    ),
)


train_N = len(trainData)
test_N = len(testData)

train_N = numTrainMolecules
test_N = numTestMolecules

# Define the GNN layer
class GNNLayer(hk.Module):
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, inputs):
        nodes, edges, features = inputs
        graph_feature_len = features.shape[-1] 
        node_feature_len = nodes.shape[-1] 
        message_feature_len = message_feat_length  

        w_init = hk.initializers.RandomNormal(stddev=weights_stddevGNN)

        we = hk.get_parameter(
            "we", shape=[node_feature_len, message_feature_len], init=w_init
        )

        b = hk.get_parameter("b", shape=[message_feature_len], init=w_init)

        wv = hk.get_parameter(
            "wv", shape=[message_feature_len, node_feature_len], init=w_init
        )

        wu = hk.get_parameter(
            "wu", shape=[node_feature_len, graph_feature_len], init=w_init
        )

        ek = jax.nn.leaky_relu(
            b
            + jnp.repeat(nodes[jnp.newaxis, ...], nodes.shape[0], axis=0)
            @ we
            * edges[..., None]
        )

        ebar = jnp.sum(ek, axis=1)

        new_nodes = jax.nn.leaky_relu(ebar @ wv) + nodes  
        new_nodes = hk.LayerNorm(
            axis=[0, 1], create_scale=False, create_offset=False, eps=1e-05
        )(new_nodes)

        global_node_features = jnp.sum(new_nodes, axis=0)

        new_features = (
            jax.nn.leaky_relu(global_node_features @ wu) + features
        )  
        return new_nodes, edges, new_features

# Define the model function
def model_fn(x):
    nodes, edges = x
    features = jnp.ones(graph_feat_length)
    x = nodes, edges, features
    
    x = GNNLayer(output_size=graph_feat_length)(x)
    x = GNNLayer(output_size=graph_feat_length)(x)
    x = GNNLayer(output_size=graph_feat_length)(x)
    x = GNNLayer(output_size=graph_feat_length)(x)

    # Saving the embeddings before the final linear layer
    embeddings = x[-1]  
    
    logits = hk.Linear(numClasses)(x[-1])
    logits = hk.Linear(numClasses)(logits)
    
    return logits, embeddings
 


# Initialize the model
model = hk.without_apply_rng(hk.transform(model_fn))

rng = jax.random.PRNGKey(0)
sampleData = data.take(1)
for dataVal in sampleData: 
    (nodes_i, edges_i), yi = dataVal
nodes_i = nodes_i.numpy()
edges_i = edges_i.numpy()

yi = yi.numpy()
xi = (nodes_i, edges_i)

params = model.init(rng, xi)

fileName = "optParams_dry-waterfall-17.npy"  
paramsArr = jnp.load(fileName, allow_pickle=True)
opt_params = {
    "gnn_layer": {
        "b": paramsArr[0],
        "we": paramsArr[1],
        "wu": paramsArr[2],
        "wv": paramsArr[3],
    },
    "gnn_layer_1": {
        "b": paramsArr[4],
        "we": paramsArr[5],
        "wu": paramsArr[6],
        "wv": paramsArr[7],
    },
    "gnn_layer_2": {
        "b": paramsArr[8],
        "we": paramsArr[9],
        "wu": paramsArr[10],
        "wv": paramsArr[11],
    },
    "gnn_layer_3": {
        "b": paramsArr[12],
        "we": paramsArr[13],
        "wu": paramsArr[14],
        "wv": paramsArr[15],
    },
    "linear": {"b": paramsArr[16], "w": paramsArr[17]},
    "linear_1": {"b": paramsArr[18], "w": paramsArr[19]},
}


test_yhat = np.empty(
    (test_N, numClasses)
)  #
test_y = np.empty((test_N, numClasses))




# Initialize lists to store embeddings and smiles for all molecules
all_embeddings = []
all_smiles = []


# Loop over the training set to store embeddings and corresponding smiles
for i, trainVal in enumerate(train_set):
    (nodes_i, edges_i), yi = trainVal
    nodes_i = nodes_i.numpy()
    edges_i = edges_i.numpy()
    yi = yi.numpy()
    xi = nodes_i, edges_i
    logits, embedding = model.apply(opt_params, xi)
    all_embeddings.append(embedding)
    all_smiles.append(trainData.smiles.iloc[i])  

# Loop over the test set to store embeddings and corresponding smiles
for i, testVal in enumerate(test_set):
    (nodes_i, edges_i), yi = testVal
    nodes_i = nodes_i.numpy()
    edges_i = edges_i.numpy()
    yi = yi.numpy()
    xi = nodes_i, edges_i
    logits, embedding = model.apply(opt_params, xi)
    all_embeddings.append(embedding)
    all_smiles.append(testData.smiles.iloc[i])  

# Convert the lists to Pandas DataFrames for easier manipulation and storage
embeddings_df = pd.DataFrame({'smiles': all_smiles, 'Embedding': all_embeddings})



#CNN-model



# Fix random seeds for reproducibility
random.seed(1)
np.random.seed(0)
tf.random.set_seed(0)

# Function to load and preprocess data
def load_and_preprocess_data(filepath, labels_file):
    df = pd.read_csv(filepath)
    df['Wave Numbers (cm^-1)'] = df['Wave Numbers (cm^-1)'].apply(eval)
    df['IR Intensity (km*mol^-1)'] = df['IR Intensity (km*mol^-1)'].apply(eval)

    labels_df = pd.read_csv(labels_file)
    labels_list = labels_df['Scent'].tolist()

    mlb = MultiLabelBinarizer(classes=labels_list)
    labels = mlb.fit_transform(df['odor_labels_filtered'].str.strip("[]").str.replace("'", "").str.split(", "))
    Y = pd.DataFrame(labels, columns=mlb.classes_)

    wave_numbers = pd.DataFrame(df['Wave Numbers (cm^-1)'].tolist())
    ir_intensity = pd.DataFrame(df['IR Intensity (km*mol^-1)'].tolist())

    X = pd.concat([wave_numbers, ir_intensity], axis=1)
    X = X.fillna(0)

    return X, Y

# Load data
X, Y = load_and_preprocess_data("STRUCTURAL-SPECTRAL DATASET.csv", "scentClasses.csv")

# Split 80% of the data into the training set and 20% into test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Build the CNN model
embedding_dim = 128
cnn_model = tf.keras.Sequential()
# Reshape input for CNN
cnn_model.add(tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)))
cnn_model.add(LeakyReLU(alpha=0.01))
# Dilated convolutions for expanding receptive field
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
cnn_model.add(tf.keras.layers.Conv1D(128, 1))
cnn_model.add(LeakyReLU(alpha=0.01))
cnn_model.add(tf.keras.layers.Conv1D(embedding_dim, 1))
cnn_model.add(LeakyReLU(alpha=0.01))
cnn_model.add(Flatten())
cnn_model.add(Dense(112, activation='relu'))
cnn_model.add(Dense(Y.shape[1], activation='sigmoid'))

# Compile the model
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = cnn_model.fit(X_train, Y_train, epochs=30, batch_size=32)

# Make predictions
Y_pred = cnn_model.predict(X_test)

# Initialize lists to store embeddings and smiles
all_embeddings_cnn = []
all_smiles_cnn = []

# Assuming your dataframe df has a column named 'smiles'
df = pd.read_csv("STRUCTURAL-SPECTRAL DATASET.csv")
train_smiles = df['smiles'].iloc[X_train.index.values]
test_smiles = df['smiles'].iloc[X_test.index.values]

# Create a new model to extract embeddings
embedding_model = tf.keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Get embeddings for training set
train_embeddings = embedding_model.predict(X_train)
train_embeddings_2d = train_embeddings.reshape(train_embeddings.shape[0], -1)
for idx, embedding in enumerate(train_embeddings_2d):
    all_embeddings_cnn.append(embedding)
    all_smiles_cnn.append(train_smiles.iloc[idx])

# Get embeddings for test set
test_embeddings = embedding_model.predict(X_test)
test_embeddings_2d = test_embeddings.reshape(test_embeddings.shape[0], -1)
for idx, embedding in enumerate(test_embeddings_2d):
    all_embeddings_cnn.append(embedding)
    all_smiles_cnn.append(test_smiles.iloc[idx])

# Convert to numpy array and DataFrame
all_embeddings_cnn = np.array(all_embeddings_cnn)
df_embeddings_cnn = pd.DataFrame(all_embeddings_cnn)
df_embeddings_cnn['smiles'] = all_smiles_cnn

# Convert list of all embeddings to a list of strings (or any other format as per your need)
all_embeddings_str = [str(embedding) for embedding in all_embeddings_cnn]

# Create DataFrame to visualize embeddings along with their corresponding smiles strings
embeddings_df = pd.DataFrame({'smiles': all_smiles_cnn, 'Embedding': all_embeddings_str})



#CONCATENATION


# Flatten all embeddings and make them 1D
all_embeddings = [emb.flatten() for emb in all_embeddings]
all_embeddings_cnn = [emb.flatten() for emb in all_embeddings_cnn]

# Convert these 1D numpy arrays to lists for DataFrame storage
all_embeddings = [emb.tolist() for emb in all_embeddings]
all_embeddings_cnn = [emb.tolist() for emb in all_embeddings_cnn]

# Create DataFrames
df_gnn_embeddings = pd.DataFrame({'smiles': all_smiles, 'GNN_Embedding': all_embeddings})
df_cnn_embeddings = pd.DataFrame({'smiles': all_smiles_cnn, 'CNN_Embedding': all_embeddings_cnn})

# Merge the DataFrames based on the smiles strings to align the embeddings
merged_df = pd.merge(df_gnn_embeddings, df_cnn_embeddings, on='smiles', how='inner')

# Save the merged DataFrame to a CSV file
merged_df.to_csv("EMBEDDINGS.csv", index=False)
# Check if all smiles strings have been aligned
if len(merged_df) == len(df_gnn_embeddings) and len(merged_df) == len(df_cnn_embeddings):
    print("All smiles strings aligned.")
else:
    print(f"Some smiles strings could not be aligned. Number of aligned embeddings: {len(merged_df)}")

# Convert the embeddings back to numpy arrays
gnn_embeddings_aligned = np.array(merged_df['GNN_Embedding'].tolist())
cnn_embeddings_aligned = np.array(merged_df['CNN_Embedding'].tolist())


scaler = StandardScaler()
gnn_embeddings_scaled = scaler.fit_transform(gnn_embeddings_aligned)
cnn_embeddings_scaled = scaler.fit_transform(cnn_embeddings_aligned)

# Concatenate the standardized embeddings
concatenated_embeddings_scaled = np.concatenate([gnn_embeddings_scaled, cnn_embeddings_scaled], axis=1)



#CSV MODIFICATION 



# Read the first CSV containing smiles and odor descriptors
df1 = pd.read_csv("smiles&labels.csv") 

# Read the second CSV containing only some smiles
df2 = pd.read_csv("EMBEDDINGS.csv") 

# Merge the two dataframes based on the 'smiles' column
merged_df = pd.merge(df2, df1, on='smiles', how='left')

# Drop the 'smiles' column from the merged dataframe
merged_df.drop('smiles', axis=1, inplace=True)

# Save the merged dataframe to a new CSV
merged_df.to_csv('embeddings&labels.csv', index=False)



#concatenated model


# Fix random seeds for reproducibility
random.seed(1)
np.random.seed(0)
tf.random.set_seed(0)

# Function to load and preprocess data
def load_and_preprocess_data(filepath, labels_file):
    df = pd.read_csv(filepath)
    df['GNN_Embedding'] = df['GNN_Embedding'].apply(eval)
    df['CNN_Embedding'] = df['CNN_Embedding'].apply(eval)

    labels_df = pd.read_csv(labels_file)
    labels_list = labels_df['Scent'].tolist()

    mlb = MultiLabelBinarizer(classes=labels_list)
    labels = mlb.fit_transform(df['odor_labels_filtered'].str.strip("[]").str.replace("'", "").str.split(", "))
    Y = pd.DataFrame(labels, columns=mlb.classes_)

    gnn = pd.DataFrame(df['GNN_Embedding'].tolist())
    cnn = pd.DataFrame(df['CNN_Embedding'].tolist())

    # Concatenate GNN and CNN embeddings
    X = pd.concat([gnn, cnn], axis=1)
    X = X.fillna(0)

    return X, Y

# Load data
X, Y = load_and_preprocess_data("embeddings&labels.csv", "scentClasses.csv")

# Split 80% of the data into the training set and 20% into test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Build the CNN model
embedding_dim = 128
con_model = tf.keras.Sequential()
con_model.add(tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)))
con_model.add(LeakyReLU(alpha=0.01))
con_model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=2, padding='causal'))
con_model.add(LeakyReLU(alpha=0.01))
con_model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=4, padding='causal'))
con_model.add(LeakyReLU(alpha=0.01))
con_model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=8, padding='causal'))
con_model.add(LeakyReLU(alpha=0.01))
con_model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=16, padding='causal'))
con_model.add(LeakyReLU(alpha=0.01))
con_model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=32, padding='causal'))
con_model.add(LeakyReLU(alpha=0.01))
con_model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=64, padding='causal'))
con_model.add(LeakyReLU(alpha=0.01))
con_model.add(tf.keras.layers.Conv1D(128, 1))
con_model.add(LeakyReLU(alpha=0.01))
con_model.add(tf.keras.layers.Conv1D(embedding_dim, 1))
con_model.add(LeakyReLU(alpha=0.01))
con_model.add(Flatten())
con_model.add(Dense(112, activation='relu'))
con_model.add(Dense(Y.shape[1], activation='sigmoid'))

# Compile the model
con_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = con_model.fit(X_train, Y_train, epochs=30, batch_size=32)


# Make predictions
Y_pred = con_model.predict(X_test)


# Initialize lists to store embeddings and smiles
all_embeddings_con = []

# Create a new model to extract embeddings
embedding_model = tf.keras.Model(inputs=con_model.input, outputs=con_model.layers[-2].output)

# Get embeddings for training set
train_embeddings_con = embedding_model.predict(X_train)
train_embeddings_2d_con = train_embeddings_con.reshape(train_embeddings_con.shape[0], -1)
for idx, embedding in enumerate(train_embeddings_2d_con):
    all_embeddings_con.append(embedding)

# Get embeddings for test set
test_embeddings_con = embedding_model.predict(X_test)
test_embeddings_2d_con = test_embeddings_con.reshape(test_embeddings_con.shape[0], -1)
for idx, embedding in enumerate(test_embeddings_2d_con):
    all_embeddings_con.append(embedding)

# Convert to numpy array and DataFrame
all_embeddings_con = np.array(all_embeddings_con)
df_embeddings_con = pd.DataFrame(all_embeddings_con)

# Convert list of all embeddings to a list of strings (or any other format as per your need)
all_embeddings_con_str = [str(embedding) for embedding in all_embeddings_con]

# Create DataFrame to visualize embeddings along with their corresponding smiles strings
embeddings_con_df = pd.DataFrame({'Embedding': all_embeddings_con_str})

# Combine train and test embeddings
all_embeddings_con = np.vstack((train_embeddings_2d_con, test_embeddings_2d_con))



##################################################
##############################################


# Compute PCA for all embeddings
pca = PCA(n_components=2)
all_embeddings_pca_gnn = pca.fit_transform(all_embeddings)
all_embeddings_pca_cnn = pca.fit_transform(all_embeddings_cnn)
reduced_embeddings_scaled = pca.fit_transform(concatenated_embeddings_scaled)
all_embeddings_pca_con = pca.fit_transform(all_embeddings_con)

# Find global min and max for the principal components
global_min_x = np.min([
    all_embeddings_pca_gnn[:, 0], 
    all_embeddings_pca_cnn[:, 0], 
    reduced_embeddings_scaled[:, 0],
    all_embeddings_pca_con[:, 0]
])

global_max_x = np.max([
    all_embeddings_pca_gnn[:, 0], 
    all_embeddings_pca_cnn[:, 0], 
    reduced_embeddings_scaled[:, 0],
    all_embeddings_pca_con[:, 0]
])

global_min_y = np.min([
    all_embeddings_pca_gnn[:, 1], 
    all_embeddings_pca_cnn[:, 1], 
    reduced_embeddings_scaled[:, 1],
    all_embeddings_pca_con[:, 1]
])

global_max_y = np.max([
    all_embeddings_pca_gnn[:, 1], 
    all_embeddings_pca_cnn[:, 1], 
    reduced_embeddings_scaled[:, 1],
    all_embeddings_pca_con[:, 1]
])

# Create a single figure
plt.figure(figsize=(20, 5))

# First Plot
plt.subplot(1, 4, 1)
plt.scatter(all_embeddings_pca_gnn[:, 0], all_embeddings_pca_gnn[:, 1])
plt.xlim(global_min_x, global_max_x)
plt.ylim(global_min_y, global_max_y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of GNN Embeddings')

# Second Plot
plt.subplot(1, 4, 2)
plt.scatter(all_embeddings_pca_cnn[:, 0], all_embeddings_pca_cnn[:, 1], alpha=0.6)
plt.xlim(global_min_x, global_max_x)
plt.ylim(global_min_y, global_max_y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of CNN Embeddings')

# Third Plot
plt.subplot(1, 4, 3)
plt.scatter(reduced_embeddings_scaled[:, 0], reduced_embeddings_scaled[:, 1], s=10)
plt.xlim(global_min_x, global_max_x)
plt.ylim(global_min_y, global_max_y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Scaled Concatenated')

# Fourth Plot
plt.subplot(1, 4, 4)
pca_train = all_embeddings_pca_con[:len(train_embeddings_2d_con), :]
pca_test = all_embeddings_pca_con[len(train_embeddings_2d_con):, :]
plt.scatter(pca_train[:, 0], pca_train[:, 1], c='b', label='Train')
plt.scatter(pca_test[:, 0], pca_test[:, 1], c='r', label='Test')
plt.xlim(global_min_x, global_max_x)
plt.ylim(global_min_y, global_max_y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of concatenated model embeddings')
plt.legend()

plt.show()
