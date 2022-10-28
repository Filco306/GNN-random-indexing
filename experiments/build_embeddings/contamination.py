import numpy as np
from .build_embedding_base import BuildEmbeddingBase
from helpers import create_sparse_adj_mat
from random_indexing import generate_index_vectors
import time
from torch_sparse import SparseTensor
import torch

"""KWARGS = {
  'dim': 100,
  'nnz': 2,
  'use_cuda': False,
  'use_sign': True,
  'features_as': 'initialization_as_context',
  'is_directed': True,
  'use_both_one_m1': True,
  'n_epochs': 93,
  'discount_degrees': True,
  'subtract_oneself': False,
  'contaminating_function': 'log_decreasing_discount',
  'contaminating_function_args' : {'C': 19.637931201399656}
  }"""


def constant_discount(C, epoch):
    return C


def decreasing_discount(C, epoch):
    return float(1 / max(1, (C * epoch + 1)))


def log_decreasing_discount(C, epoch):
    return float(1 / (C * np.log1p(epoch + np.e)))


def exponential_decay(epoch, C):
    return np.exp(-C * epoch)


class Contamination(BuildEmbeddingBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # import ipdb

        # ipdb.sset_trace()
        self._is_fitted = False

    def fit(self, nodes: np.ndarray, edges: np.ndarray, features: np.ndarray):
        """ In this fit function, we build the model. """
        assert self.features_as in [
            "binary_variables",
            "random_indexing",
            "excluded",
            "graph",
            "initialization",
            "initialization_as_context",
        ]

        self.features = features

        # Generate the graph, either with features or without
        print("Building embeddings.")
        start = time.time()
        self.index_vectors, nodes, edges = generate_index_vectors(
            nodes=nodes,
            edges=edges,
            features=self.features,
            dim=int(self.dim),
            nnz=self.nnz,
            features_as=self.features_as,
            use_cuda=self.use_cuda,
            use_both_one_m1=self.use_both_one_m1,
        )
        print(f"Building the embeddings took {time.time() - start} seconds.")
        # Create the sparse adjacency matrix

        self.sparse_adj_mat = SparseTensor.from_scipy(
            create_sparse_adj_mat(nodes, edges, is_directed=self.is_directed)
        )
        self.degrees = self.sparse_adj_mat.sum(dim=1)
        self.sparse_adj_mat = self.sparse_adj_mat.to_torch_sparse_coo_tensor()
        # To prevent division by 0
        self.degrees[self.degrees == 0] = 1
        self.embedding = self.index_vectors

        self._is_fitted = True
        self._run_epochs()
        return self

    def _one_epoch(self, epoch: int, subtract_oneself: bool):
        device = torch.device("cpu")

        if isinstance(self.sparse_adj_mat, SparseTensor):
            self.sparse_adj_mat = self.sparse_adj_mat.to_torch_sparse_coo_tensor().to(
                device
            )
        else:
            self.sparse_adj_mat = self.sparse_adj_mat.to(device)
        old_context_vectors = torch.Tensor(self.embedding).to(device)
        if self.discount_degrees:
            degrees = torch.Tensor(1 / self.degrees).view(-1, 1).to(device)
            # import ipdb; ipdb.sset_trace()
            context_vecs_mult = old_context_vectors * degrees
        else:
            context_vecs_mult = old_context_vectors
        # sparse_adj_mat = self.sparse_adj_mat.to(device)

        new_context_vectors_to_add = torch.mm(self.sparse_adj_mat, context_vecs_mult)
        # import ipdb; ipdb.sset_trace()
        new_context_vectors = (
            old_context_vectors.contiguous()
            + eval(self.contaminating_function)(
                **{**self.contaminating_function_args, **{"epoch": epoch}}
            )
            * new_context_vectors_to_add.contiguous()
        )
        ret = new_context_vectors.cpu().numpy()
        if (
            np.any(np.isposinf(ret))
            or np.any(np.isneginf(ret))
            or np.any(np.isnan(ret))
        ):
            self.done = True
            return old_context_vectors
            raise ValueError("Infinity in context vectors")
        if subtract_oneself is True and epoch != 0:
            ret -= (
                eval(self.contaminating_function)(
                    **{**self.contaminating_function_args, **{"epoch": epoch}}
                )
                * self.index_vectors
            )
        return ret

    def _run_epochs(self):
        self.use_wandb = False
        self.done = False
        for epoch in range(self.n_epochs):
            self.embedding = self._one_epoch(epoch, self.subtract_oneself)
            if self.done:
                break

    def _transform(self, nodes: np.ndarray):
        return self.embedding[nodes]


if __name__ == "__main__":
    pass
