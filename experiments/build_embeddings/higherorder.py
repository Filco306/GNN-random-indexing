import numpy as np
from .build_embedding_base import BuildEmbeddingBase
from helpers import create_sparse_adj_mat
import torch
from torch_sparse import SparseTensor
from utils import cuda_is_available

KWARGS = {
    "dim": 1000,
    "permute_vecs": True,
    "use_sign": False,
    "use_cuda": True,
    "features_as": "binary_variables",
    "use_both_one_m1": False,
    "nnz": 10,
    "is_directed": True,
}


class HigherOrder(BuildEmbeddingBase):
    """Baseline independent of the dimensionality. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.dim % 2 != 0:
            self.dim = int(self.dim - 1)

    def fit(self, nodes: np.ndarray, edges: np.ndarray, features: np.ndarray):
        self.features = (
            features.cpu().numpy() if isinstance(features, torch.Tensor) else features
        )
        if self.use_cuda is True:
            assert cuda_is_available(), "CUDA is not available."

        self.device = torch.device(
            "cuda:0" if cuda_is_available() and self.use_cuda else "cpu"
        )

        self.sparse_adj_mat = SparseTensor.from_scipy(
            create_sparse_adj_mat(nodes, edges, self.is_directed)
        )
        self.sparse_adj_mat_sq = self.sparse_adj_mat.spspmm(self.sparse_adj_mat)
        self.sparse_adj_mat_3 = self.sparse_adj_mat_sq.spspmm(self.sparse_adj_mat)

        fst_context = self.sparse_adj_mat.to_torch_sparse_coo_tensor()
        snd_context = self.sparse_adj_mat_sq.to_torch_sparse_coo_tensor()
        trd_context = self.sparse_adj_mat_3.to_torch_sparse_coo_tensor()

        self.embedding = (
            1.0 * fst_context + 0.1 * snd_context + 0.1 * trd_context
        ).to_dense()

        self.embedding = self.embedding.cpu().numpy()
        self._is_fitted = True

    def _transform(self, nodes: np.ndarray = None):
        return self.embedding[nodes] if nodes is not None else self.embedding
