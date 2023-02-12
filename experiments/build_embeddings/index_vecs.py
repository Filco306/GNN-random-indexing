import numpy as np
from .build_embedding_base import BuildEmbeddingBase
from random_indexing import generate_index_vectors
from helpers import create_feature_nodes_edges
import torch
from utils import cuda_is_available

KWARGS = {
    "dim": 250,
    "use_sign": False,
    "features_as": "initialization_as_context",
    "use_cuda": True,
    "permute_vecs": True,
    "use_both_one_m1": True,
    "nnz": 2,
    "is_directed": False,
    "zeroth_order": 1.0,
    "fst_order": 0.1,
    "snd_order": 0.1,
    "trd_order": 0.1,
}


BEST_KWARGS_CITESEER = {
    "dim": 1500,
    "use_sign": False,
    "features_as": "initialization_as_context",
    "use_cuda": True,
    "permute_vecs": True,
    "use_both_one_m1": True,
    "nnz": 2,
    "is_directed": False,
    "zeroth_order": 1.0,
    "fst_order": 1.0,
    "snd_order": 0.1,
    "trd_order": 0.01,
}


class IndexVecs(BuildEmbeddingBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.dim % 2 != 0:
            self.dim = int(self.dim - 1)
        # Default: Permute the vectors

    def fit(self, nodes: np.ndarray, edges: np.ndarray, features: np.ndarray):
        if self.use_cuda is True:
            assert cuda_is_available(), "CUDA is not available."

        self.device = torch.device(
            "cuda:0" if cuda_is_available() and self.use_cuda else "cpu"
        )

        self._is_fitted = True
        self.index_vectors, _, _ = generate_index_vectors(
            nodes=nodes,
            edges=edges,
            features=features,
            dim=self.dim,
            nnz=int(min(self.dim, (self.nnz // 2) * 2)),
            features_as=self.features_as,
            use_cuda=self.use_cuda,
            use_both_one_m1=self.use_both_one_m1,
        )
        if self.features_as == "graph":
            feature_nodes, feature_edges = create_feature_nodes_edges(features)
            nodes = np.concatenate([nodes, feature_nodes], axis=0)
            edges = np.concatenate([edges, feature_edges], axis=0)
        self.embedding = (
            self.index_vectors
            if hasattr(self, "use_sign") is False or self.use_sign is False
            else np.sign(self.index_vectors)
        )
        if self.use_both_one_m1 is False:
            self.embedding = np.abs(self.embedding)

    def _transform(self, nodes: np.ndarray = None):
        return self.embedding[nodes] if nodes is not None else self.embedding
