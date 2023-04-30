# %%
import jax.numpy as jnp
from tqdm import tqdm

from jax_shenanigans.graphs import Graph
from jax_shenanigans.utils.io import load_txt
from jax_shenanigans.utils.jax import is_between

data_adj = load_txt("../data/ENZYMES_A.txt", delimiter=",").astype(int)
data_node_att = load_txt("../data/ENZYMES_node_attributes.txt", delimiter=",")
data_node_label = load_txt("../data/ENZYMES_node_labels.txt", delimiter=",").astype(int)
data_graph_indicator = load_txt(
    "../data/ENZYMES_graph_indicator.txt", delimiter=","
).astype(int)
data_graph_labels = load_txt("../data/ENZYMES_graph_labels.txt", delimiter=",").astype(
    int
)


# %%
n_graphs = 5  # data_graph_indicator.max()
graph_nodes = {}
graph_node_features = {}
graph_node_labels = {}
graph_edges = {}

lgst_edge_node = jnp.where(
    condition=data_adj[:, 0] > data_adj[:, 1], x=data_adj[:, 0], y=data_adj[:, 1]
)

for gi in tqdm(range(1, n_graphs), "Preparing graphs"):
    gi_mask = data_graph_indicator == gi
    graph_nodes[gi] = jnp.where(gi_mask)[0] + 1
    graph_node_features[gi] = data_node_att[gi]
    graph_node_labels[gi] = data_node_label[gi]
    graph_edges[gi] = data_adj[
        is_between(lgst_edge_node, graph_nodes[gi].min(), graph_nodes[gi].max())
    ]


# %%
graphs = {
    gi: Graph(
        nodes=graph_nodes[gi],
        adj_sparse=graph_edges[gi],
        node_features=graph_node_features,
    )
    for gi in tqdm(range(1, n_graphs), "Instantiating graphs")
}

graphs[1]

# %%
