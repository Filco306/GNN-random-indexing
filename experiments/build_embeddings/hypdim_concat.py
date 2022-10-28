import numpy as np
from .build_embedding_base import BuildEmbeddingBase
from random_indexing import generate_index_vectors
from helpers import create_sparse_adj_mat, create_feature_nodes_edges
import torch
from torch_sparse import SparseTensor
from utils import cuda_is_available

KWARGS = {
    "dim": 2500,
    "use_sign": False,
    "features_as": "initialization",
    "use_cuda": True,
    "use_both_one_m1": True,
    "nnz": 10,
    "num_contexts": 4,
    "is_directed": False,
}


class HypDimCompConcat(BuildEmbeddingBase):
    def __init__(self, **kwargs):
        super().__init__(**KWARGS)
        if self.dim % 2 != 0:
            self.dim = int(self.dim - 1)
        # Default: Permute the vectors

    def permute(self, vectors: torch.Tensor, step: int):
        """ Shift everything one step. """
        return torch.roll(vectors, step, 1)

    def generate_context_vectors(self):
        index_vectors = torch.Tensor(self.index_vectors)
        fst_context = self.sparse_adj_mat.spmm(index_vectors)
        snd_context = self.sparse_adj_mat_sq.spmm(index_vectors)
        trd_context = self.sparse_adj_mat_3.spmm(index_vectors)
        # import ipdb; ipdb.sset_trace()
        # NOTE: We do not use scalars here
        # because we just want to see how good the representation is.
        contexts = [index_vectors, fst_context, snd_context, trd_context]
        self.embedding = (
            torch.sign(
                torch.cat([contexts[i] for i in range(self.num_contexts)], dim=1)
            )
            if self.use_sign
            else (torch.cat([contexts[i] for i in range(self.num_contexts)], dim=1))
        )

        # import ipdb; ipdb.sset_trace()
        # Take the others randomly.
        # self.embedding[self.embedding == 0] = torch.sign(
        #     torch.randn((self.embedding == 0).nonzero().shape[0])
        # )
        self.embedding = self.embedding.long().cpu().numpy()

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
        self.sparse_adj_mat = SparseTensor.from_scipy(
            create_sparse_adj_mat(nodes, edges, self.is_directed)
        )
        self.sparse_adj_mat_sq = self.sparse_adj_mat.spspmm(self.sparse_adj_mat)
        self.sparse_adj_mat_3 = self.sparse_adj_mat_sq.spspmm(self.sparse_adj_mat)

        self.generate_context_vectors()

    def _transform(self, nodes: np.ndarray = None):
        return self.embedding[nodes] if nodes is not None else self.embedding
