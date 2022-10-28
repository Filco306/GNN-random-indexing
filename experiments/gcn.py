import argparse
import numpy as np
from sparsegcn import SparseNet
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from datasets import load_dataset
from train_eval import run
from str2bool import str2bool
from utils import seed_everything
from gcn_random_index import RINet, JointRIGCN
import wandb
from dotenv import load_dotenv

assert load_dotenv()


class Net_orig(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_orig, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class Net(torch.nn.Module):
    def __init__(self, dataset, hidden, dropout):
        super(Net, self).__init__()

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
        self.crd = CRD(dataset.num_features, hidden, dropout)
        self.cls = CLS(hidden, num_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x


def run_experiment(args):
    seed_everything(42)

    use_wandb = args.use_wandb
    if use_wandb is True:
        wandb.init(project="GCN-random-indexing", entity="kth-gale")
        wandb.config.update({"Task": "Node classification"})
    for key, value in args.__dict__.items():
        print(key, value)
        if use_wandb is True and key != "name":
            wandb.config.update({key: value})
        elif key == "name" and use_wandb is True:
            wandb.config.update({"Experiment name": value})

    if args.features_as == "IndexVecs":
        featparams = {
            "features_as": args.params_features_as,
            "use_cuda": True,
            "use_both_one_m1": True,
            "dim": args.params_dim,
            "nnz": args.params_nnz,
        }
    else:
        featparams = {"dim": -1}

    dataset = load_dataset(
        dataset_name=args.dataset,
        features_as=args.features_as,
        normalize_features=args.normalize_features,
        split=args.split,
        params=featparams,
    )
    if args.model == "Net" and args.use_sparse is False:
        model = Net(dataset, args.hidden, args.dropout)
    elif args.model == "SparseNet":
        model = SparseNet(dataset, args.hidden, args.dropout)
    elif args.model == "RINet":
        model = RINet(
            dataset,
            args.hidden,
            args.dropout,
            depth=args.depth,
            use_sparse=args.use_sparse,
        )
    elif args.model == "JointRIGCN":
        model = JointRIGCN(
            dataset,
            args.hidden,
            args.dropout,
            depth=args.depth,
            use_sparse=args.use_sparse,
        )
    else:
        raise ValueError(f"Model {args.model} not found. ")
    name = f"dataset_{args.dataset}|features_as_{args.features_as}|model_{model.__class__.__name__}|depth_{args.depth}"  # noqa: E501
    if use_wandb is True:
        without_features = (featparams.get("features_as", "") == "excluded") or (
            args.features_as == "OneHotNodes"
        )
        with_features = without_features is False
        wandb.config.update({"name": name, "With features": with_features})

    kwargs = {
        "dataset": dataset,
        "model": model,
        "str_optimizer": args.optimizer,
        "str_preconditioner": args.preconditioner,
        "runs": args.runs,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "early_stopping": args.early_stopping,
        "logger": args.logger,
        "momentum": args.momentum,
        "eps": args.eps,
        "update_freq": args.update_freq,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "hyperparam": args.hyperparam,
        "use_wandb": use_wandb,
    }

    if args.hyperparam == "eps":
        for param in np.logspace(-3, 0, 10, endpoint=True):
            print(f"{args.hyperparam}: {param}")
            kwargs[args.hyperparam] = param
            run(**kwargs)
    elif args.hyperparam == "update_freq":
        for param in [4, 8, 16, 32, 64, 128]:
            print(f"{args.hyperparam}: {param}")
            kwargs[args.hyperparam] = param
            run(**kwargs)
    elif args.hyperparam == "gamma":
        for param in np.linspace(1.0, 10.0, 10, endpoint=True):
            print(f"{args.hyperparam}: {param}")
            kwargs[args.hyperparam] = param
            run(**kwargs)
    else:
        run(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="public")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--normalize_features", type=str2bool, default=False)
    parser.add_argument("--logger", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--preconditioner", type=str, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--update_freq", type=int, default=50)
    parser.add_argument("--features_as", type=str, default="Baseline")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--hyperparam", type=str, default=None)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--use_sparse", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=True)
    parser.add_argument("--model", type=str, default="Net")
    parser.add_argument(
        "--params_features_as",
        type=str,
        default="initialization_as_context",
        choices=["excluded", "initialization_as_context"],
    )
    parser.add_argument("--params_dim", type=int, default=500)
    parser.add_argument("--params_nnz", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--name", type=str, default="Unknown")
    args = parser.parse_args()
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        run_experiment(args)
