from scipy.sparse import dok_matrix
import numpy as np


def create_sparse_adj_mat(
    nodes: np.array, edges: np.array, is_directed: bool = False
) -> dok_matrix:

    sparse_mat = dok_matrix((len(nodes), len(nodes)), dtype=np.float32)
    for edge in edges:
        sparse_mat[edge[0], edge[1]] = 1
        if is_directed is False:
            sparse_mat[edge[1], edge[0]] = 1
    return sparse_mat


def get_features_as_index_vectors(
    features: np.ndarray, feature_index_vectors, n_nodes: int, dim: int
) -> np.ndarray:
    nodes, feats = np.where(features > 0)

    initial_index_vectors = np.zeros((n_nodes, dim))
    print(f"Generating, {nodes.shape[0]}")
    # import ipdb; ipdb.sset_trace()
    for ii in range(nodes.shape[0]):
        initial_index_vectors[nodes[ii]] += (
            feature_index_vectors[feats[ii]] * features[nodes[ii], feats[ii]]
        )
    print("Generated index vectors")
    return initial_index_vectors


def get_feature_context_vectors_as_index(
    features: np.ndarray, node_index_vectors: np.ndarray, dim: int
):
    initial_feature_index_vectors = np.zeros((features.shape[1], dim))
    # Give all index
    nodes, feats = np.where(features == 1)
    for ii in range(nodes.shape[0]):
        initial_feature_index_vectors[feats[ii]] += node_index_vectors[nodes[ii]]
    return get_features_as_index_vectors(
        features, initial_feature_index_vectors, n_nodes=features.shape[0], dim=dim
    )


def create_feature_nodes_edges(features) -> tuple:
    # Features is a NxF matrix
    # N is the number of nodes
    # F is the number of features
    n_nodes = features.shape[0]
    n_feature_nodes = features.shape[1]

    feature_edges = np.concatenate([np.where(features == 1)], axis=1)
    feature_nodes = n_nodes + np.arange(n_feature_nodes)
    feature_edges[1, :] += n_nodes

    return feature_nodes, np.roll(feature_edges, 1).T
