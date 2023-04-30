from typing import Union

import jax.numpy as jnp
import networkx as nx
from jax.experimental.sparse import BCOO
from jax.scipy.linalg import sqrtm

from jax_shenanigans.utils.jax import fill_diagonal
from jax_shenanigans.utils.typing import is_jax_array


class Graph:
    @staticmethod
    def edges_to_BCOO(edges, n_nodes):
        data = jnp.ones(len(edges))
        shape = (n_nodes, n_nodes)
        return BCOO((data, edges), shape=shape)

    @staticmethod
    def calc_adj_hat(A):
        A_self = fill_diagonal(A, 1)  # A with self connections
        D = jnp.zeros_like(A, dtype=A.dtype)
        D = fill_diagonal(D, A.sum(axis=1).flatten())
        D = sqrtm(D)
        return D @ A_self @ D

    def __init__(
        self,
        nodes: Union[jnp.ndarray, int],
        adj_sparse: Union[jnp.ndarray, BCOO] = None,
        node_features: jnp.ndarray = None,
        edge_features: jnp.ndarray = None,
        graph_features: jnp.ndarray = None,
    ):
        self.nodes = nodes
        self.adj_sparse = self._process_adj_sparse(adj_sparse)
        self.node_features = node_features
        self.edge_features = edge_features
        self.graph_features = graph_features

    @property
    def adj_dense(self):
        return self.adj_sparse.todense()

    @property
    def adj_hat(self):
        return Graph.calc_adj_hat(self.adj_dense)

    def _process_adj_sparse(self, adj_sparse):
        if is_jax_array(adj_sparse):
            return Graph.edges_to_BCOO(adj_sparse - adj_sparse.min(), n_nodes=len(self))
        return adj_sparse

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        A_np = self.adj_dense.to_py()
        A_np = A_np - A_np.min()
        nx_graph = nx.DiGraph(A_np)
        labels = {i: v for i, v in enumerate(self.nodes)}
        nx.draw(nx_graph, with_labels=True, labels=labels)
        return ""
