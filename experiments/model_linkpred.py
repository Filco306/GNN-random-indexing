from datasets import load_link_dataset
import torch.nn as nn
import argparse
import torch
import torch.nn.functional as F
from train_eval_linkpred import run_linkpred
from str2bool import str2bool
from utils import seed_everything
from gcn_random_index import RILayer
import wandb
from dotenv import load_dotenv
import torch_geometric.nn as gnn
from utils import get_numdims
from sparsegcn import SparseGCNConv

assert load_dotenv()


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, use_sparse: bool):
        super().__init__()
        self.gcn1 = (
            gnn.GCNConv(in_channels, hidden_channels)
            if use_sparse is False
            else SparseGCNConv(in_channels, hidden_channels)
        )
        self.gcn2 = (
            gnn.GCNConv(hidden_channels, out_channels)
            if use_sparse is False
            else SparseGCNConv(hidden_channels, out_channels=out_channels)
        )

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x


class Decoder(nn.Module):
    def __init__(self, use_sigmoid=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid

    def reset_parameters(self):
        return None

    def forward(self, z):
        return torch.sigmoid(z @ z.t()) if self.use_sigmoid else (z @ z.t())


class LinkPredNet(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels, use_sparse: bool):
        super().__init__()
        dim = get_numdims(data)
        self.encoder = Encoder(
            in_channels=dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            use_sparse=use_sparse,
        )
        self.decoder = Decoder(use_sigmoid=True)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, x_features=None):
        z = self.encoder(x, edge_index)
        adj_pred = self.decoder(z)
        return adj_pred, z


class RILinkPredNet(torch.nn.Module):
    def __init__(
        self,
        data,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        use_sparse: bool,
        use_feats_too: bool = False,
    ):
        super().__init__()
        dim = get_numdims(data, use_features_too=use_feats_too)
        self.rilayer = RILayer(data, depth)
        self.encoder = Encoder(
            in_channels=dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            use_sparse=use_sparse,
        )
        self.decoder = Decoder(use_sigmoid=True)

    def reset_parameters(self):
        self.rilayer.reset_parameters()
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, x_features=None):
        x_ = self.rilayer(x)
        z = self.encoder(x_, edge_index)
        adj_pred = self.decoder(z)
        return adj_pred, z


class JointRILinkPredNet(RILinkPredNet):
    def __init__(
        self,
        data,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        use_sparse: bool,
    ):
        super().__init__(
            data, hidden_channels, out_channels, depth, use_sparse, use_feats_too=True
        )
        self.lin = nn.Parameter(
            torch.diag(torch.ones(data.x_features.shape[1])), requires_grad=False
        )

    def forward(self, x, edge_index, x_features):
        x_ = self.rilayer(x)
        x_features_ = x_features @ self.lin
        x_ = torch.cat((x_, x_features_), dim=1)
        z = self.encoder(x_, edge_index)
        adj_pred = self.decoder(z)
        return adj_pred, z


def run_experiment(args):
    seed_everything(42)
    use_wandb = args.use_wandb
    if use_wandb is True:
        wandb.init(project="GCN-random-indexing", entity="kth-gale")
        wandb.config.update({"Task": "Link prediction"})
    for key, value in args.__dict__.items():
        print(key, value)
        if use_wandb is True and key != "name":
            wandb.config.update({key: value})

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

    train, valid, test = load_link_dataset(
        dataset_name=args.dataset,
        features_as=args.features_as,
        normalize_features=args.normalize_features,
        split=args.split,
        params=featparams,
    )

    # Animesh Code
    if args.model == "LinkPredNet":
        model = LinkPredNet(train, args.hidden, args.out, use_sparse=args.use_sparse)
    elif args.model == "RILinkPredNet":
        model = RILinkPredNet(
            train, args.hidden, args.out, args.depth, use_sparse=args.use_sparse
        )
    elif args.model == "JointRILinkPredNet":
        model = JointRILinkPredNet(
            train, args.hidden, args.out, args.depth, use_sparse=args.use_sparse
        )
    else:
        raise ValueError(f"Model {args.model} not supported")
    # Animesh code ends
    name = f"dataset_{args.dataset}|features_as_{args.features_as}|model_{model.__class__.__name__}|depth_{args.depth}"  # noqa: E501
    if use_wandb is True:
        without_features = (featparams.get("features_as", "") == "excluded") or (
            args.features_as == "OneHotNodes"
        )
        with_features = without_features is False
        wandb.config.update({"name": name, "With features": with_features})

    kwargs = {
        "train": train,
        "valid": valid,
        "test": test,
        "model": model,
        "str_optimizer": args.optimizer,
        "str_preconditioner": args.preconditioner,
        "runs": args.runs,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "early_stopping": args.early_stopping,
        "momentum": args.momentum,
        "eps": args.eps,
        "update_freq": args.update_freq,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "hyperparam": args.hyperparam,
        "use_wandb": use_wandb,
    }
    run_linkpred(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="public")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out", type=int, default=64)  # Animesh Code
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
    parser.add_argument("--model", type=str, default="LinkPredNet")
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
    run_experiment(args)
