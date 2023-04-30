from typing import Union

import jax.numpy as jnp
import networkx as nx
from jax.experimental.sparse import BCOO
from jax.scipy.linalg import sqrtm

from jax_shenanigans.utils.jax import fill_diagonal
from jax_shenanigans.utils.typing import is_jax_array


class Graph:
    @staticmethod
    def sparse_E(edges, n_vertex):
        data = jnp.ones(len(edges))
        shape = (n_vertex, n_vertex)
        return BCOO((data, edges), shape=shape)

    @staticmethod
    def calc_A_hat(A):
        A_self = fill_diagonal(A, 1)  # A with self connections
        D = jnp.zeros_like(A, dtype=A.dtype)
        D = fill_diagonal(D, A.sum(axis=1).flatten())
        D = sqrtm(D)
        return D @ A_self @ D

    def __init__(
        self,
        V: Union[jnp.ndarray, int],
        E: Union[jnp.ndarray, BCOO] = None,
        U: jnp.ndarray = None,
        A: jnp.ndarray = None,
    ):
        self.V = V
        self.E = self._process_E(E) or BCOO.from_dense(A)
        self.U = U
        self.A = A or self.E.todense()
        self.A_hat = None

    def set_A_hat(self):
        self.A_hat = Graph.calc_A_hat(self.A)

    def _process_E(self, E):
        if is_jax_array(E):
            return Graph.sparse_E(E - E.min(), n_vertex=len(self))
        return E

    def __len__(self):
        return len(self.V)

    def __repr__(self):
        A_np = self.A.to_py()
        A_np = A_np - A_np.min()
        nx_graph = nx.DiGraph(A_np)
        labels = {i: v for i, v in enumerate(self.V)}
        nx.draw(nx_graph, with_labels=True, labels=labels)
        return ""
