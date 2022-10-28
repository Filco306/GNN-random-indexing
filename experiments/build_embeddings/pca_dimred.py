import numpy as np
from .build_embedding_base import BuildEmbeddingBase
import torch
from sklearn.decomposition import PCA
from utils import cuda_is_available


class PCADimRed(BuildEmbeddingBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.dim % 2 != 0:
            self.dim = int(self.dim - 1)
        # Default: Permute the vectors

    def permute(self, vectors: torch.Tensor, step: int):
        """ Shift everything one step. """
        return torch.roll(vectors, step, 1)

    def fit(self, nodes: np.ndarray, edges: np.ndarray, features: np.ndarray):
        if self.use_cuda is True:
            assert cuda_is_available(), "CUDA is not available."

        self.device = torch.device(
            "cuda:0" if cuda_is_available() and self.use_cuda else "cpu"
        )

        self._is_fitted = True
        self.index_vectors = PCA(self.dim).fit_transform(features)
        self.embedding = self.index_vectors

    def _transform(self, nodes: np.ndarray = None):
        return self.embedding[nodes] if nodes is not None else self.embedding
