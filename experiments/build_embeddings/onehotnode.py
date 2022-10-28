import numpy as np
from .build_embedding_base import BuildEmbeddingBase


class OneHotNodes(BuildEmbeddingBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, nodes: np.ndarray, edges: np.ndarray, features: np.ndarray):

        self.embedding = np.diag(np.ones(nodes.shape[0]))
        self._is_fitted = True

    def _transform(self, nodes: np.ndarray = None):
        return self.embedding[nodes] if nodes is not None else self.embedding
