import os.path as osp
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
import torch_geometric as tg
from build_embeddings import (  # noqa: F401
    IndexVecs,
    OneHotNodes,
)
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit
from typing import Tuple
import numpy as np


class ModifiedPlanetoidDataset(InMemoryDataset):
    def __init__(self, planetoiddata, features):
        self.name = planetoiddata.name

        super().__init__(
            planetoiddata.root, planetoiddata.transform, planetoiddata.pre_transform
        )

        self.data, self.slices = (
            tg.data.Data(
                x=features.float(),
                y=planetoiddata[0].y,
                edge_index=planetoiddata[0].edge_index,
                train_mask=planetoiddata[0].train_mask,
                val_mask=planetoiddata[0].val_mask,
                test_mask=planetoiddata[0].test_mask,
                num_classes=torch.unique(planetoiddata[0].y).shape[0]
                if len(planetoiddata[0].y.shape) == 1
                else planetoiddata[0].y.shape[1],
            ),
            planetoiddata.slices,
        )
        self.data.x = features
        self.split = planetoiddata.split
        assert self.split in ["public", "full", "random"]

        if planetoiddata.split == "full":
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif planetoiddata.split == "random":
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[
                    torch.randperm(idx.size(0))[: planetoiddata.num_train_per_class]
                ]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[: planetoiddata.num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[
                remaining[
                    planetoiddata.num_val : planetoiddata.num_val
                    + planetoiddata.num_test
                ]
            ] = True

            self.data, self.slices = self.collate([data])


def get_planetoid_dataset(
    name,
    features_as=None,
    params=None,
    normalize_features=False,
    transform=None,
    split="public",
):
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", name)

    if split == "complete":
        dataset = Planetoid(path, name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[: dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[
            dataset[0].num_nodes - 1000 : dataset[0].num_nodes - 500
        ] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500 :] = 1
    else:
        dataset = Planetoid(path, name, split=split)
    if features_as not in ["Baseline", None]:
        # assert params is not None
        embedfit = eval(features_as)(**(params if params is not None else {}))

        features = embedfit.fit_transform(
            nodes=np.arange(dataset[0].num_nodes),
            edges=dataset[0].edge_index.T.numpy(),
            features=dataset[0].x.numpy(),
        )
        feats = (
            torch.from_numpy(features).float()
            if isinstance(features, np.ndarray)
            else features.float()
        )
        dataset = ModifiedPlanetoidDataset(dataset, feats)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


def load_node_classification(
    dataset: str = "Cora", n_perc_train: Tuple[float, float] = None
):
    if dataset.lower() in ["cora", "citeseer", "pubmed"]:
        return get_planetoid_dataset(name=dataset, split="complete")
    data = AttributedGraphDataset(root="./data", name=dataset)

    splitter = RandomNodeSplit()
    dataset = splitter(data.data)

    return dataset


def load_dataset(
    dataset_name: str,
    features_as=None,
    params=None,
    normalize_features=False,
    transform=None,
    split="public",
):
    if dataset_name.lower() in ["pubmed", "citeseer", "cora"]:
        assert split is not None
        dataset = get_planetoid_dataset(
            dataset_name,
            features_as=features_as,
            params=params,
            normalize_features=normalize_features,
            transform=transform,
            split=split,
        )
        return dataset
    # Otherwise just load an attributed graph dataset
    dataset = load_node_classification(dataset_name)
    original_features = dataset.x
    if features_as not in ["Baseline", None]:
        embedfit = eval(features_as)(**(params if params is not None else {}))

        features = embedfit.fit_transform(
            nodes=np.arange(dataset.x.shape[0]),
            edges=dataset.edge_index.T.numpy(),
            features=dataset.x.numpy(),
        )
        feats = (
            torch.from_numpy(features).float()
            if isinstance(features, np.ndarray)
            else features.float()
        )
        dataset.x = torch.Tensor(feats)
    dataset.x_features = original_features
    return dataset


def load_link_dataset(
    dataset_name: str,
    features_as=None,
    params=None,
    normalize_features=False,
    transform=None,
    split="public",
):

    data = load_dataset(
        dataset_name,
        features_as=None,
        params=None,
        normalize_features=normalize_features,
        transform=transform,
        split=split,
    )
    tfs = RandomLinkSplit()
    try:
        ddd = data.data
    except AttributeError:
        ddd = data
    train, val, test = tfs(ddd)
    original_features = train.x
    if features_as not in ["Baseline", None]:
        # assert params is not None
        embedfit = eval(features_as)(**(params if params is not None else {}))

        features = embedfit.fit_transform(
            nodes=np.arange(train.x.shape[0]),
            edges=train.edge_label_index[:, train.edge_label == 1].T.numpy(),
            features=train.x.numpy(),
        )
        feats = (
            torch.from_numpy(features).float()
            if isinstance(features, np.ndarray)
            else features.float()
        )

        train.x = torch.Tensor(feats)
        val.x = torch.Tensor(feats)
        test.x = torch.Tensor(feats)
    train.x_features = original_features
    val.x_features = original_features
    test.x_features = original_features
    return train, val, test


if __name__ == "__main__":
    lst_names = ["Cora", "CiteSeer", "PubMed"]
    for name in lst_names:
        dataset = get_planetoid_dataset(name)
        print(f"dataset: {name}")
        print(f"num_nodes: {dataset[0]['x'].shape[0]}")
        print(f"num_edges: {dataset[0]['edge_index'].shape[1]}")
        print(f"num_classes: {dataset.num_classes}")
        print(f"num_features: {dataset.num_node_features}")
