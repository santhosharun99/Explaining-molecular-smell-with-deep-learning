"""

GNN MODEL - STRUCTURAL DATA

"""



# Imports required libraries
import tensorflow as tf
import numpy as np
import seaborn as sns
import jax
import jax.numpy as jnp
import haiku as hk
import pandas as pd
import rdkit, rdkit.Chem, rdkit.Chem.rdDepictor, rdkit.Chem.Draw
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.metrics import auc as sklearn_auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import optax
import sklearn.metrics
import warnings
import random
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
random.seed(1)
np.random.seed(0)
tf.random.set_seed(0)

# Hyperparameters and initial setup
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

# Initialize empty arrays to store test set true labels and predicted labels
test_y = np.empty((test_N, numClasses))
test_yhat = np.empty((test_N, numClasses))

# Iterate over the test set to populate test_y and test_yhat
for i, testVal in enumerate(test_set):
    (nodes_i, edges_i), yi = testVal
    nodes_i = nodes_i.numpy()
    edges_i = edges_i.numpy()
    yi = yi.numpy()
    xi = (nodes_i, edges_i)

    # Store true labels
    test_y[i, :] = yi

    logits, _ = model.apply(opt_params, xi)

    # Apply sigmoid to get predicted probabilities
    pred_probs = 1 / (1 + np.exp(-logits))

    # Store predicted labels
    test_yhat[i, :] = pred_probs

# Initialize a list to store AUC-ROC scores for each class
auc_scores = []

# Iterate over each class to compute AUC-ROC
for i in range(numClasses):
    
    y_true = test_y[:, i]
    y_score = test_yhat[:, i]
    
    unique_classes = np.unique(y_true)
    
    if len(unique_classes) < 2:
        print(f"Cannot calculate ROC AUC for descriptor {scentClasses[i]}: Only found {len(unique_classes)} unique class in y_true")
        auc = 0.5
    else:
        auc = roc_auc_score(y_true, y_score)
    
    auc_scores.append(auc)
    print(f"AUC-ROC for descriptor {scentClasses[i]}: {auc}")

#Calculate Mean AUC-ROC
mean_auc = np.mean(auc_scores)
print(f"Mean AUC-ROC: {mean_auc}")

#Calculate Median AUC-ROC
median_auc = np.median(auc_scores)
print(f"Median AUC-ROC: {median_auc}")

#compute the micro-average AUC-ROC
fpr, tpr, _ = roc_curve(test_y.ravel(), test_yhat.ravel())
micro_auc = sklearn_auc(fpr, tpr)
print(f"Micro-average AUC-ROC: {micro_auc}")

# Calculate Macro-average AUC-ROC
macro_auc = np.mean(auc_scores) 
print(f"Macro-average AUC-ROC: {macro_auc}")

# Calculate Weighted AUC-ROC
class_counts = np.sum(test_y, axis=0)
total_count = np.sum(class_counts)
weighted_auc = np.sum([auc * count for auc, count in zip(auc_scores, class_counts)]) / total_count
print(f"Weighted AUC-ROC: {weighted_auc}")


