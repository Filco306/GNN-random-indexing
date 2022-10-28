import torch
from torch_sparse import SparseTensor
import numpy as np
from helpers import create_feature_nodes_edges, get_features_as_index_vectors


def _sort_out_dups(x):
    """
        Sort out duplicate entries in random indexing
        sampling.
    """
    y, indices = x.sort(dim=-1)
    y[:, 1:] *= ((y[:, 1:] - y[:, :-1]) != 0).long()
    indices = indices.sort(dim=-1)[1]
    result = torch.gather(y, 1, indices)
    return result


def create_index_vectors(
    nrows, ncols, nnz, use_both_one_m1, return_numpy: bool = False, device="cuda:0",
):
    nnz_indices = (torch.rand(nrows, nnz, device=device) * ncols).ceil().long()
    nnz_indices = torch.unique(nnz_indices, dim=1).sort(dim=1)[0]
    nnz_indices = _sort_out_dups(nnz_indices)
    # Eliminate all duplicate entries. Might slow down the procedure but totally worth it.
    while (nnz_indices == 0).any():
        nnz_indices[nnz_indices == 0] = (
            (torch.rand((nnz_indices == 0).sum(), device=device) * ncols).ceil().long()
        )
    nnz_indices = nnz_indices - 1
    more_inds = torch.cat([torch.arange(nrows).reshape(-1, 1).long()] * nnz, dim=1).to(
        device
    )
    nnz_indices_ = torch.cat([more_inds.unsqueeze(2), nnz_indices.unsqueeze(2)], dim=2)
    nnz_indices_ = nnz_indices_.view(-1, 2)
    assert nnz % 2 == 0, "please choose an even number of nnz-entries"

    half = nnz // 2
    vals = (
        torch.cat(
            [
                torch.ones(nrows, half),
                (-(1 ** int(use_both_one_m1))) * torch.ones(nrows, half),
            ],
            dim=1,
        )
        .long()
        .to(device)
        .view(-1)
    )

    nnz_indices_.shape[0] == vals.shape[0]
    nnz_indices_.t().shape
    index_vectors = torch.sparse_coo_tensor(
        indices=nnz_indices_.t(), values=vals, size=(nrows, ncols)
    )

    return (
        index_vectors
        if return_numpy is False
        else index_vectors.to_dense().cpu().numpy()
    )


def build_context_vectors_random_walks(
    walks: np.ndarray,
    n_nodes: int,
    dim: int,
    nnz: int,
    n_restarts: int = None,
    one_per_node: bool = False,
    device="cuda:0",
):

    walks = torch.LongTensor(walks.astype("int32")).to(device)
    if one_per_node is False:
        index_vectors = create_index_vectors(
            walks.shape[0], dim, nnz, use_both_one_m1=True
        )
    else:
        assert n_restarts is not None
        index_vectors = create_index_vectors(n_nodes, dim, nnz, use_both_one_m1=True)
        # index_vectors = index_vectors_nodes.cpu().to_dense().repeat(n_restarts, 1).to_sparse()

    context_vectors = torch.zeros(n_nodes, dim).to(device)
    index_vecs_tsp = SparseTensor.from_torch_sparse_coo_tensor(index_vectors).to(device)
    for node_i in range(n_nodes):
        inc_ = (walks == node_i).sum(axis=1)
        included_in = torch.where(inc_ > 0)[0]
        context_vectors[node_i] += (
            index_vecs_tsp[
                torch.div(
                    included_in,
                    (n_nodes if one_per_node is True else 1),
                    rounding_mode="floor",
                )
            ]
            * inc_[included_in][:, None]
        ).sum(dim=0)
    return context_vectors.cpu().numpy()


def get_density(arr):
    return (arr != 0).sum() / np.prod(arr.shape)


def generate_index_vectors(
    nodes: np.ndarray,
    edges: np.ndarray,
    features: np.ndarray,
    dim: int,
    nnz: int,
    features_as: str,
    use_cuda: bool,
    use_both_one_m1: bool,
):
    assert features_as in [
        "binary_variables",
        "random_indexing",
        "excluded",
        "graph",
        "initialization",
        "initialization_as_context",
    ]
    if features is None or features_as == "random_indexing":
        features_as = "excluded"
    if features_as == "graph":
        feature_nodes, feature_edges = create_feature_nodes_edges(features)
        nodes = np.concatenate([nodes, feature_nodes], axis=0)
        edges = np.concatenate([edges, feature_edges], axis=0)
    if features_as in ["graph", "excluded", "binary_variables"]:

        index_vectors = create_index_vectors(
            nrows=len(nodes),
            ncols=dim,
            nnz=nnz,
            use_both_one_m1=use_both_one_m1,
            return_numpy=True,
            device="cuda:0" if use_cuda is True else "cpu",
        )
    elif features_as == "initialization":
        index_vectors = get_features_as_index_vectors(
            features=features,
            feature_index_vectors=create_index_vectors(
                nrows=features.shape[1],
                ncols=dim,
                nnz=nnz,
                use_both_one_m1=use_both_one_m1,
                return_numpy=True,
            ),
            n_nodes=nodes.shape[0],
            dim=dim,
        )
    elif features_as == "initialization_as_context":
        index_vectors = get_features_as_index_vectors(
            features=features,
            feature_index_vectors=create_index_vectors(
                nrows=features.shape[1],
                ncols=dim,
                nnz=nnz,
                use_both_one_m1=use_both_one_m1,
                return_numpy=True,
            ),
            n_nodes=nodes.shape[0],
            dim=dim,
        )
    if features_as == "binary_variables":
        index_vectors = np.concatenate([index_vectors, features], axis=1)

    return index_vectors, nodes, edges
