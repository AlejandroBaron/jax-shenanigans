# %%
import numpy as np

data_adj = np.loadtxt("../data/ENZYMES_A.txt", delimiter=",").astype(int)
data_node_att = np.loadtxt("../data/ENZYMES_node_attributes.txt", delimiter=",")
data_node_label = np.loadtxt("../data/ENZYMES_node_labels.txt", delimiter=",").astype(
    int
)
data_graph_indicator = np.loadtxt(
    "../data/ENZYMES_graph_indicator.txt", delimiter=","
).astype(int)
data_graph_labels = np.loadtxt(
    "../data/ENZYMES_graph_labels.txt", delimiter=","
).astype(int)
