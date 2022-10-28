import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from helpers import create_sparse_adj_mat
from torch_sparse import SparseTensor
import numpy as np
from sparsegcn import SparseGCNConv
from dotenv import load_dotenv
from utils import cuda_is_available, get_numdims

assert load_dotenv()


class RILayer(torch.nn.Module):
    def __init__(self, dataset, depth: int):
        super().__init__()
        self.is_directed = True
        self.permute_vecs = True
        self.use_sign = False
        self.depth = depth

        try:
            self.num_nodes = dataset.data.x.shape[0]
        except AttributeError:
            self.num_nodes = dataset.x.shape[0]
        nodes = np.arange(self.num_nodes)
        try:
            edgs = (
                dataset.data.edge_label_index[:, dataset.data.edge_label == 1]
                .cpu()
                .numpy()
            )
        except AttributeError:
            edgs = dataset.edge_label_index[:, dataset.edge_label == 1].cpu().numpy()
        self.sparse_adj_mat = SparseTensor.from_scipy(
            create_sparse_adj_mat(nodes, edgs.T, self.is_directed)
        )
        device = torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
        if self.depth >= 2:
            self.sparse_adj_mat_sq = self.sparse_adj_mat.spspmm(self.sparse_adj_mat)
            self.sparse_adj_mat_sq = self.sparse_adj_mat_sq.fill_diag(0)
        if self.depth >= 3:
            self.sparse_adj_mat_3 = self.sparse_adj_mat_sq.spspmm(self.sparse_adj_mat)
            self.sparse_adj_mat_3 = self.sparse_adj_mat_3.fill_diag(0)
            self.sparse_adj_mat_3 = self.sparse_adj_mat_3.coalesce().to(device)

        if self.depth >= 2:
            self.sparse_adj_mat_sq = self.sparse_adj_mat_sq.coalesce().to(device)
        self.sparse_adj_mat = self.sparse_adj_mat.coalesce().to(device)
        self.reset_parameters()

    def initialize_weights(self):
        if self.depth >= 1:
            self.zeroth_order = torch.nn.Embedding(1, 1).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
            self.zeroth_order.weight.data.fill_(1.0)
        else:
            self.zeroth_order = torch.ones((1, 1)).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        if self.depth >= 1:
            self.fst_order = torch.nn.Embedding(1, 1).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        else:
            self.fst_order = torch.zeros((1, 1)).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        if self.depth >= 2:

            self.snd_order = torch.nn.Embedding(1, 1).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        else:
            self.snd_order = torch.zeros((1, 1)).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        if self.depth >= 3:
            self.trd_order = torch.nn.Embedding(1, 1).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        else:
            self.trd_order = torch.zeros((1, 1)).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )

    def reset_parameters(self):
        self.initialize_weights()

    def permute(self, vectors: torch.Tensor, step: int):
        """ Shift everything one step. """
        return torch.roll(vectors, step, 1)

    def get_orders(self):
        idx = torch.LongTensor([0]).to(torch.device("cuda:1"))
        zeroth = (
            self.zeroth_order(idx)
            if isinstance(self.zeroth_order, torch.nn.Embedding)
            else self.zeroth_order
        )
        fst = (
            self.fst_order(idx)
            if isinstance(self.fst_order, torch.nn.Embedding)
            else self.fst_order
        )
        snd = (
            self.snd_order(idx)
            if isinstance(self.snd_order, torch.nn.Embedding)
            else self.snd_order
        )
        trd = (
            self.trd_order(idx)
            if isinstance(self.trd_order, torch.nn.Embedding)
            else self.trd_order
        )
        ret1, ret2, ret3, ret4 = (
            torch.nn.functional.relu(zeroth),
            torch.nn.functional.relu(fst),
            torch.nn.functional.relu(snd),
            torch.nn.functional.relu(trd),
        )
        tot = ret1 + ret2 + ret3 + ret4 + 1e-6  # prevent division by 0
        ret1, ret2, ret3, ret4 = ret1 / tot, ret2 / tot, ret3 / tot, ret4 / tot
        return ret1, ret2, ret3, ret4

    def forward(self, index_vectors):

        # index_vectors = torch.Tensor(self.index_vectors)

        fst_context = (
            self.sparse_adj_mat.spmm(index_vectors) if (self.depth >= 1) else 0.0
        )
        # import ipdb; ipdb.sset_trace()
        if self.depth >= 2:
            snd_context = self.sparse_adj_mat_sq.spmm(
                index_vectors
                if self.permute_vecs is False
                else self.permute(index_vectors, 1)
            )
        else:
            snd_context = 0.0

        if self.depth >= 3:
            trd_context = self.sparse_adj_mat_3.spmm(
                index_vectors
                if self.permute_vecs is False
                else self.permute(index_vectors, 2)
            )
        else:
            trd_context = 0.0
        # import ipdb; ipdb.sset_trace()
        # NOTE: We do not use scalars here because
        # we just want to see how good the representation is.
        zeroth, fst, snd, trd = self.get_orders()
        embedding = (
            torch.sign(
                (zeroth) * index_vectors
                + (fst) * fst_context
                + (snd) * snd_context
                + (trd) * trd_context
            )
            if self.use_sign
            else (
                (zeroth) * index_vectors
                + (fst) * fst_context
                + (snd) * snd_context
                + (trd) * trd_context
            )
        )
        return embedding


class RICRD(torch.nn.Module):
    def __init__(
        self,
        dataset,
        d_in,
        d_out,
        p,
        use_sign: bool = False,
        permute_vecs: bool = True,
        is_directed: bool = False,
        depth: int = 3,
        use_sparse: bool = False,
    ):
        super(RICRD, self).__init__()
        self.conv = (
            GCNConv(d_in, d_out, cached=True)
            if use_sparse is False
            else SparseGCNConv(d_in, d_out, cached=True)
        )
        self.depth = depth
        self.initialize_weights()
        self.p = p
        self.use_sign = use_sign
        self.permute_vecs = True
        self.is_directed = is_directed

        try:
            self.num_nodes = dataset.data.x.shape[0]
        except AttributeError:
            self.num_nodes = dataset.x.shape[0]
        nodes = np.arange(self.num_nodes)
        try:
            edgs = dataset.data.edge_index.cpu().numpy()
        except AttributeError:
            edgs = dataset.edge_index.cpu().numpy()

        self.sparse_adj_mat = SparseTensor.from_scipy(
            create_sparse_adj_mat(nodes, edgs.T, self.is_directed)
        ).to(torch.device("cuda:1") if cuda_is_available() else torch.device("cpu"))
        if self.depth >= 2:
            self.sparse_adj_mat_sq = self.sparse_adj_mat.spspmm(self.sparse_adj_mat)
            self.sparse_adj_mat_sq = self.sparse_adj_mat_sq.fill_diag(0)
        if self.depth >= 3:
            self.sparse_adj_mat_3 = self.sparse_adj_mat_sq.spspmm(self.sparse_adj_mat)
            self.sparse_adj_mat_3 = self.sparse_adj_mat_3.fill_diag(0)

    def initialize_weights(self):
        if self.depth >= 1:
            self.zeroth_order = torch.nn.Embedding(1, 1).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
            self.zeroth_order.weight.data.fill_(1.0)
        else:
            self.zeroth_order = torch.ones((1, 1)).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        if self.depth >= 1:
            self.fst_order = torch.nn.Embedding(1, 1).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        else:
            self.fst_order = torch.zeros((1, 1)).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        if self.depth >= 2:

            self.snd_order = torch.nn.Embedding(1, 1).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        else:
            self.snd_order = torch.zeros((1, 1)).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        if self.depth >= 3:
            self.trd_order = torch.nn.Embedding(1, 1).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )
        else:
            self.trd_order = torch.zeros((1, 1)).to(
                torch.device("cuda:1") if cuda_is_available() else torch.device("cpu")
            )

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.initialize_weights()

    def permute(self, vectors: torch.Tensor, step: int):
        """ Shift everything one step. """
        return torch.roll(vectors, step, 1)

    def get_orders(self):
        idx = torch.LongTensor([0]).to(torch.device("cuda:1"))
        zeroth = (
            self.zeroth_order(idx)
            if isinstance(self.zeroth_order, torch.nn.Embedding)
            else self.zeroth_order
        )
        fst = (
            self.fst_order(idx)
            if isinstance(self.fst_order, torch.nn.Embedding)
            else self.fst_order
        )
        snd = (
            self.snd_order(idx)
            if isinstance(self.snd_order, torch.nn.Embedding)
            else self.snd_order
        )
        trd = (
            self.trd_order(idx)
            if isinstance(self.trd_order, torch.nn.Embedding)
            else self.trd_order
        )
        ret1, ret2, ret3, ret4 = (
            torch.nn.functional.relu(zeroth),
            torch.nn.functional.relu(fst),
            torch.nn.functional.relu(snd),
            torch.nn.functional.relu(trd),
        )
        tot = ret1 + ret2 + ret3 + ret4 + 1e-6  # prevent division by 0
        ret1, ret2, ret3, ret4 = ret1 / tot, ret2 / tot, ret3 / tot, ret4 / tot
        return ret1, ret2, ret3, ret4

    def generate_context_vectors(self, index_vectors, edges):

        # index_vectors = torch.Tensor(self.index_vectors)

        fst_context = (
            self.sparse_adj_mat.spmm(index_vectors) if (self.depth >= 1) else 0.0
        )
        # import ipdb; ipdb.sset_trace()
        if self.depth >= 2:
            snd_context = self.sparse_adj_mat_sq.spmm(
                index_vectors
                if self.permute_vecs is False
                else self.permute(index_vectors, 1)
            )
        else:
            snd_context = 0.0

        if self.depth >= 3:
            trd_context = self.sparse_adj_mat_3.spmm(
                index_vectors
                if self.permute_vecs is False
                else self.permute(index_vectors, 2)
            )
        else:
            trd_context = 0.0
        # import ipdb; ipdb.sset_trace()
        # NOTE: We do not use scalars here because
        # we just want to see how good the representation is.
        zeroth, fst, snd, trd = self.get_orders()
        embedding = (
            torch.sign(
                (zeroth) * index_vectors
                + (fst) * fst_context
                + (snd) * snd_context
                + (trd) * trd_context
            )
            if self.use_sign
            else (
                (zeroth) * index_vectors
                + (fst) * fst_context
                + (snd) * snd_context
                + (trd) * trd_context
            )
        )
        return embedding

    def forward(self, x, edge_index, mask=None, x_features=None):
        x = self.generate_context_vectors(x, edge_index)
        # import ipdb; ipdb.sset_trace()
        if x_features is not None:
            x = torch.cat([x, x_features], dim=1)
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class RICLS(torch.nn.Module):
    def __init__(self, d_in, d_out, use_sparse: bool = False):
        super(RICLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)
        # if use_sparse is False
        # else SparseGCNConv(d_in, d_out, cached=True)
        # )

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class RINet(torch.nn.Module):
    def __init__(
        self,
        dataset,
        hidden,
        dropout,
        use_sign: bool = False,
        permute_vecs: bool = True,
        is_directed: bool = False,
        depth: int = 3,
        use_sparse: bool = False,
        use_feats_too: bool = False,
    ):
        super(RINet, self).__init__()

        # import ipdb; ipdb.sset_trace()
        try:
            num_classes = dataset.num_classes
        except AttributeError:
            num_classes = (
                (dataset.y.max() + 1).item()
                if len(dataset.y.shape) == 1
                else dataset.y.shape[1]
            )
            if len(dataset.y.shape) == 2:
                assert dataset.y.sum(dim=1).max() == 1

        self.crd = RICRD(
            dataset=dataset,
            d_in=get_numdims(dataset, use_features_too=use_feats_too),
            d_out=hidden,
            p=dropout,
            use_sign=use_sign,
            permute_vecs=permute_vecs,
            is_directed=is_directed,
            depth=depth,
            use_sparse=use_sparse,
        )
        self.cls = RICLS(hidden, num_classes, use_sparse=use_sparse)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x


class JointRIGCN(RINet):
    def __init__(
        self,
        dataset,
        hidden,
        dropout,
        use_sign: bool = False,
        permute_vecs: bool = True,
        is_directed: bool = False,
        depth: int = 3,
        use_sparse: bool = False,
    ):
        super(JointRIGCN, self).__init__(
            dataset,
            hidden,
            dropout,
            use_sign,
            permute_vecs,
            is_directed,
            depth,
            use_sparse,
            use_feats_too=True,
        )

    def forward(self, data):
        x, x_features, edge_index = data.x, data.x_features, data.edge_index

        x = self.crd(x, edge_index, x_features=x_features, mask=data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x
