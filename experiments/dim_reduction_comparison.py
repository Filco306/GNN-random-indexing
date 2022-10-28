import argparse
import numpy as np
from str2bool import str2bool
from datasets import load_dataset, load_link_dataset
from train_eval_linkpred import run_linkpred
from gcn_random_index import RINet
from model_linkpred import LinkPredNet, RILinkPredNet
from gcn import Net
from train_eval import run
import pickle
from dotenv import load_dotenv
from datetime import datetime

assert load_dotenv()
# import ipdb; ipdb.sset_trace()


RESULTS = []


def run_one_run(
    dataset,
    args,
    depth: int,
    is_baseline: bool = False,
    use_sparse: bool = False,
    task: str = "nodeclassification",
):

    if is_baseline is True:
        model = (
            Net(dataset, args["hidden"], args["dropout"])
            if task == "nodeclassification"
            else LinkPredNet(
                dataset[0],
                hidden_channels=args["hidden"],
                out_channels=args["out"],
                use_sparse=args["use_sparse"],
            )
        )
    else:
        model = (
            RINet(
                dataset,
                args["hidden"],
                args["dropout"],
                permute_vecs=args["permute_vecs"],
                depth=depth,
                use_sparse=use_sparse,
            )
            if task == "linkpred"
            else RILinkPredNet(
                dataset[0],
                hidden_channels=args["hidden"],
                out_channels=args["out"],
                depth=depth,
                use_sparse=args["use_sparse"],
            )
        )
    kwargs = (
        {
            "dataset": dataset,
            "model": model,
            "str_optimizer": args["optimizer"],
            "str_preconditioner": args["preconditioner"],
            "runs": args["runs"],
            "epochs": args["epochs"],
            "lr": args["lr"],
            "weight_decay": args["weight_decay"],
            "early_stopping": args["early_stopping"],
            "logger": args["logger"],
            "momentum": args["momentum"],
            "eps": args["eps"],
            "update_freq": args["update_freq"],
            "gamma": args["gamma"],
            "alpha": args["alpha"],
            "hyperparam": args["hyperparam"],
            "use_wandb": False,
        }
        if task == "nodeclassification"
        else {
            "train": dataset[0],
            "valid": dataset[1],
            "test": dataset[2],
            "model": model,
            "str_optimizer": args["optimizer"],
            "str_preconditioner": args["preconditioner"],
            "runs": args["runs"],
            "epochs": args["epochs"],
            "lr": args["lr"],
            "weight_decay": args["weight_decay"],
            "early_stopping": args["early_stopping"],
            "momentum": args["momentum"],
            "eps": args["eps"],
            "update_freq": args["update_freq"],
            "gamma": args["gamma"],
            "alpha": args["alpha"],
            "hyperparam": args["hyperparam"],
            "use_wandb": False,
        }
    )
    if task == "nodeclassification":

        val_loss, acc_mean, acc_stdev, n_params = run(**kwargs)
        print(f"{val_loss} {acc_mean} {acc_stdev} {n_params}")
        return {
            "Validation loss": val_loss,
            "Accuracy mean": acc_mean,
            "Accuracy stdev": acc_stdev,
            "Number of parameters": n_params,
        }
    else:
        assert task == "linkprediction"
        valloss, roc, roc_std, nparams = run_linkpred(**kwargs)
        return {
            "Validation loss": valloss,
            "ROC": roc,
            "ROC stdev": roc_std,
            "Number of parameters": nparams,
        }


def run_dim_experiment(
    dataset_name: str,
    args,
    split: str = "complete",
    preprocessmodel: str = "IndexVecs",
    nnz: int = 2,
    use_sparse: bool = False,
    feats_as: str = "initialization_as_context",
    task: str = "nodeclassification",
):
    dataset = (
        load_dataset(dataset_name, features_as=None, split=split)
        if task == "nodeclassification"
        else load_link_dataset(dataset_name, features_as=None, split=split)
    )
    # import ipdb; ipdb.sset_trace()
    try:
        if feats_as == "initialization_as_context":
            maxdim = dataset.data.x.shape[1]
        else:
            maxdim = dataset.data.x.shape[0]
    except AttributeError:
        if isinstance(dataset, tuple):
            maxdim = dataset[0].x.shape[1]
        elif feats_as == "initialization_as_context":
            maxdim = dataset.x.shape[1]
        else:
            maxdim = dataset.x.shape[0]
    depth = args["depth"]
    RESULTS.append(
        {
            **run_one_run(
                dataset, args, depth=None, is_baseline=True, use_sparse=None, task=task
            ),
            **{
                "Dataset": dataset_name,
                "Split": split,
                "Model": "Baseline",
                "dim": maxdim,
                "features_as": feats_as,
                "Permute vectors": None,
                "Depth": None,
                "task": task,
            },
        }
    )
    currtime = datetime.now().strftime("%Y-%m-%d-%H:%M")
    dims = np.arange(50, maxdim, (maxdim - 50) // 15).tolist() + [maxdim]
    for dim in dims:
        print(f"{dataset} {dim}")
        if nnz is not None:
            nnz = max(2, min(dim // 20, 10))
        dataset = (
            load_dataset(
                dataset_name,
                features_as=preprocessmodel,
                params={
                    "dim": dim,
                    "use_cuda": True,
                    "nnz": nnz,
                    "features_as": feats_as,
                    "use_both_one_m1": True,
                },
                split=split,
            )
            if task == "nodeclassification"
            else load_link_dataset(
                dataset_name,
                features_as=preprocessmodel,
                params={
                    "dim": dim,
                    "use_cuda": True,
                    "nnz": nnz,
                    "features_as": feats_as,
                    "use_both_one_m1": True,
                },
                split=split,
            )
        )

        RESULTS.append(
            {
                **run_one_run(
                    dataset, args, depth=depth, use_sparse=args["use_sparse"], task=task
                ),
                **{
                    "Dataset": dataset_name,
                    "Split": split,
                    "Model": "RINet",
                    "Preprocessmodel": preprocessmodel,
                    "nnz": nnz,
                    "dim": dim,
                    "features_as": feats_as,
                    "Sparse": use_sparse,
                    "Permute vectors": args["permute_vecs"],
                    "Depth": args["depth_propagate"],
                },
            }
        )
        with open(f"results/exps/dim_comparison_results_{currtime}.pickle", "wb",) as f:
            pickle.dump(RESULTS, f)


# import ipdb

# with ipdb.launch_ipdb_on_exception():
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-propagate", type=int, required=False)
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--split", type=str, default="public")
    parser.add_argument(
        "--task",
        type=str,
        default="nodeclassification",
        choices=["nodeclassification", "linkprediction"],
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--depth", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--normalize_features", type=str2bool, default=False)
    parser.add_argument("--logger", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--preconditioner", type=str, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--update_freq", type=int, default=50)
    parser.add_argument("--features_as", type=str, default="Baseline")
    parser.add_argument("--processmodel", type=str, default="IndexVecs")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--hyperparam", type=str, default=None)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--use_sparse", type=str2bool, default=False)
    parser.add_argument("--params_nnz", type=int, default=None)
    parser.add_argument("--permute_vecs", type=str2bool, default=True)
    parser.add_argument(
        "--params_features_as",
        type=str,
        default="initialization_as_context",
        choices=["excluded", "initialization_as_context"],
    )
    parser.add_argument("--model", type=str, default="Net")
    args_ = parser.parse_args()
    # print(args.depth)
    assert args_.normalize_features is False
    args = args_.__dict__

    run_dim_experiment(
        dataset_name=args["dataset"],
        args=args,
        split=args["split"],
        preprocessmodel=args["features_as"],
        nnz=args["params_nnz"],
        use_sparse=args["use_sparse"],
        feats_as=args["params_features_as"],
        task=args["task"],
    )
