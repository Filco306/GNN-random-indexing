from __future__ import division

import time
import scipy.stats as st
from wandb import wandb
import torch
import torch.nn.functional as F
from torch import tensor
from tqdm import tqdm
from dotenv import load_dotenv
from utils import cuda_is_available, calculate_confidence_interval
from sklearn.metrics import average_precision_score, roc_auc_score

assert load_dotenv()


device = torch.device("cuda:1" if cuda_is_available() else "cpu")

path_runs = "runs"


def forward_edge_index(z, edge_index, sigmoid=True):
    value = z[edge_index[0], edge_index[1]]
    return torch.sigmoid(value) if sigmoid else value


def get_alphas(model):
    zeroth = (
        model.crd.zeroth_order.weight.data
        if isinstance(model.crd.zeroth_order, torch.nn.Embedding)
        else model.crd.zeroth_order
    )
    fst = (
        model.crd.fst_order.weight.data
        if isinstance(model.crd.fst_order, torch.nn.Embedding)
        else model.crd.fst_order
    )
    snd = (
        model.crd.snd_order.weight.data
        if isinstance(model.crd.snd_order, torch.nn.Embedding)
        else model.crd.snd_order
    )
    trd = (
        model.crd.trd_order.weight.data
        if isinstance(model.crd.trd_order, torch.nn.Embedding)
        else model.crd.trd_order
    )
    return zeroth, fst, snd, trd


def run_linkpred(
    train,
    valid,
    test,
    model,
    str_optimizer,
    str_preconditioner,
    runs,
    epochs,
    lr,
    weight_decay,
    early_stopping,
    momentum,
    eps,
    update_freq,
    gamma,
    alpha,
    hyperparam,
    use_wandb: bool,
):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model.__class__.__name__)
    print("NUMBER OF PARAMETERS:", num_params)

    print(
        "NUMBER OF FEATURES:",
        (
            train.num_features
            if model.__class__.__name__ != "JointRILinkPredNet"
            else train.num_features + train.x_features.shape[1]
        ),
    )
    if use_wandb:
        wandb.log(
            {
                "Number of parameters": num_params,
                "Number of features": train.num_features,
            }
        )
    val_losses, rocs, durations = [], [], []
    torch.manual_seed(42)
    for i_run in range(runs):
        data = train
        data = data.to(device)

        model.to(device).reset_parameters()
        preconditioner = None
        criterion = None

        if str_optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif str_optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,)

        if cuda_is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float("inf")
        test_roc = 0
        val_loss_history = []
        for epoch in tqdm(range(1, epochs + 1)):
            lam = (float(epoch) / float(epochs)) ** gamma if gamma is not None else 0.0
            _ = train_epoch(
                model, optimizer, criterion, data, preconditioner, lam
            )  # Animesh Code added

            eval_info = evaluate(model, train, valid, test)  # Animesh Code added

            eval_info["epoch"] = int(epoch)
            eval_info["run"] = int(i_run + 1)
            eval_info["time"] = time.perf_counter() - t_start
            eval_info["eps"] = eps
            eval_info["update-freq"] = update_freq
            if gamma is not None:
                eval_info["gamma"] = gamma

            if alpha is not None:
                eval_info["alpha"] = alpha

            if eval_info["val loss"] < best_val_loss:
                best_val_loss = eval_info["val loss"]
                # val_roc = eval_info["val roc"]
                test_roc = eval_info["test roc"]

            val_loss_history.append(eval_info["val loss"])

            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1) : -1])
                if eval_info["val loss"] > tmp.mean().item():
                    break
            if use_wandb is True:
                wandb.log(
                    {
                        "rolling test roc": test_roc,
                        "rolling val loss": eval_info["val loss"],
                    }
                )
        if cuda_is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        val_losses.append(best_val_loss)
        rocs.append(test_roc)
        durations.append(t_end - t_start)
        print(f"ROC for model {model.__class__.__name__}: {test_roc}")

    loss, roc, duration = tensor(val_losses), tensor(rocs), tensor(durations)
    print(
        "Val Loss: {:.4f}, Test ROC: {:.2f} ± {:.2f}, Duration: {:.3f} \n".format(
            loss.mean().item(),
            100 * roc.mean().item(),
            100 * roc.std().item(),
            duration.mean().item(),
        )
    )
    dof = roc.shape[0] - 1
    scale = st.sem(roc.cpu().numpy())
    avg, n95conf = calculate_confidence_interval(roc.mean().item(), dof, scale)
    if use_wandb:
        wandb.log(
            {
                "Val Loss": loss.mean().item(),
                "Test ROC mean": roc.mean().item(),
                "Test ROC 95 %": n95conf,
                "Test ROC lower 95 %": avg - n95conf,
                "Test ROC upper 95 %": avg + n95conf,
                "Duration": duration.mean().item(),
                "Val Loss std": loss.std().item(),
                "Test ROC std": roc.std().item(),
                "Duration std": duration.std().item(),
            }
        )

    return (
        loss.mean().item(),
        100 * roc.mean().item(),
        100 * roc.std().item(),
        num_params,
    )


# Animesh Code Below
# 3functions ->train,test,evaluate
def train_epoch(model, optimizer, criterion, data, preconditioner=None, lam=0.0):
    model.train()
    optimizer.zero_grad()
    out, adj_recon = model(
        data.x,
        data.edge_label_index[:, data.edge_label == 1],
        x_features=data.x_features,
    )

    loss = F.binary_cross_entropy(
        out[data.edge_label_index[0, :], data.edge_label_index[1, :]], data.edge_label
    )
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_epoch(model, data, train):
    outs = {}
    out, _ = model(
        train.x,
        train.edge_label_index[:, train.edge_label == 1],
        x_features=train.x_features,
    )
    y = data.edge_label
    pos_edge_index = data.edge_label_index[:, y == 1]
    neg_edge_index = data.edge_label_index[:, y == 0]
    pos_pred = forward_edge_index(out, pos_edge_index, sigmoid=True)
    neg_pred = forward_edge_index(out, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    loss = F.binary_cross_entropy(pred, y.to(pred.device))
    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
    ra, ap = roc_auc_score(y, pred), average_precision_score(y, pred)

    outs["loss"] = loss.item()
    outs["roc"] = ra
    outs["aps"] = ap
    return outs


def evaluate(model, train, val, test_):
    # out_train = test_epoch(model, train, train)
    out_val = test_epoch(model, val, train)
    out_test = test_epoch(model, test_, train)

    outs = {}
    for tvt, out in zip(["val", "test"], [out_val, out_test]):
        outs["{} loss".format(tvt)] = out["loss"]
        outs["{} aps".format(tvt)] = out["aps"]
        outs["{} roc".format(tvt)] = out["roc"]
    return outs
