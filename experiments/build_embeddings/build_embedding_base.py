import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class BuildEmbeddingBase(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.dim = int(self.dim)
        if int(self.dim) % 2 == 1:
            self.dim = int(self.dim) + 1
        self._is_fitted = False

    def fit(self, nodes: np.ndarray, edges: np.ndarray, features: np.ndarray):
        self._is_fitted = True
        pass

    def _transform(self, nodes: np.ndarray):
        pass

    def transform(self, nodes: np.ndarray):
        assert self._is_fitted, "Please fit the model first! "
        return self._transform(nodes)

    def fit_transform(self, nodes: np.ndarray, edges: np.ndarray, features: np.ndarray):
        self.fit(nodes, edges, features)
        # First check if it even exists for backward compatibility
        if "downscale" in dir(self) and getattr(self, "downscale") is True:
            # Then we assume that downscaled_dim and scale_features are set
            # and exist.
            self.pca = PCA(n_components=self.downscaled_dim)
            self._large_embeddings = self.transform(nodes)
            if self.scale_features is True:
                sc = StandardScaler()

                self._large_embeddings = sc.fit_transform(self._large_embeddings)

            self.embedding = self.pca.fit_transform(self._large_embeddings)
        return self.transform(nodes)
