from __future__ import division

import time
import os

from wandb import wandb
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
import utils as ut
import psgd
from tqdm import tqdm
from dotenv import load_dotenv
from utils import cuda_is_available

assert load_dotenv()

device = torch.device("cuda:1" if cuda_is_available() else "cpu")

path_runs = "runs"


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


def run(
    dataset,
    model,
    str_optimizer,
    str_preconditioner,
    runs,
    epochs,
    lr,
    weight_decay,
    early_stopping,
    logger,
    momentum,
    eps,
    update_freq,
    gamma,
    alpha,
    hyperparam,
    use_wandb: bool,
):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("NUMBER OF PARAMETERS:", num_params)
    print(
        "NUMBER OF FEATURES:",
        (
            dataset.num_features
            if model.__class__.__name__ != "JointRIGCN"
            else dataset.num_features + dataset.x_features
        ),
    )
    if use_wandb:
        wandb.log(
            {
                "Number of parameters": num_params,
                "Number of features": dataset.num_features,
            }
        )
    if logger is not None:
        if hyperparam:
            logger += f"-{hyperparam}{eval(hyperparam)}"
        path_logger = os.path.join(path_runs, logger)
        print(f"path logger: {path_logger}")

        ut.empty_dir(path_logger)
        logger = (
            SummaryWriter(log_dir=os.path.join(path_runs, logger))
            if logger is not None
            else None
        )

    val_losses, accs, durations = [], [], []
    torch.manual_seed(42)
    for i_run in range(runs):
        try:
            data = dataset[0]
        except KeyError:
            data = dataset
        data = data.to(device)

        model.to(device).reset_parameters()
        if str_preconditioner == "KFAC":

            preconditioner = psgd.KFAC(
                model,
                eps,
                sua=False,
                pi=False,
                update_freq=update_freq,
                alpha=alpha if alpha is not None else 1.0,
                constraint_norm=False,
            )
        else:
            preconditioner = None

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
        test_acc = 0
        val_loss_history = []

        for epoch in tqdm(range(1, epochs + 1)):
            lam = (float(epoch) / float(epochs)) ** gamma if gamma is not None else 0.0
            _ = train(model, optimizer, data, preconditioner, lam)
            eval_info = evaluate(model, data)
            eval_info["epoch"] = int(epoch)
            eval_info["run"] = int(i_run + 1)
            eval_info["time"] = time.perf_counter() - t_start
            eval_info["eps"] = eps
            eval_info["update-freq"] = update_freq
            if gamma is not None:
                eval_info["gamma"] = gamma

            if alpha is not None:
                eval_info["alpha"] = alpha

            if logger is not None:
                for k, v in eval_info.items():
                    logger.add_scalar(k, v, global_step=epoch)

            if eval_info["val loss"] < best_val_loss:
                best_val_loss = eval_info["val loss"]
                # val_acc = eval_info["val acc"]
                test_acc = eval_info["test acc"]

            val_loss_history.append(eval_info["val loss"])

            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1) : -1])
                if eval_info["val loss"] > tmp.mean().item():
                    break
            if use_wandb is True:
                wandb.log({"test acc": test_acc, "val loss": eval_info["val loss"]})
        if cuda_is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        if model.__class__.__name__ == "RINet":
            zeroth, fst, snd, trd = get_alphas(model)
            print(f"zeroth: {zeroth} | fst: {fst} | snd: {snd} | trd: {trd}")
        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
        print(f"Acc for model {model.__class__.__name__}: {test_acc}")

    if logger is not None:
        logger.close()
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    print(
        "Val Loss: {:.4f}, Test Accuracy: {:.2f} ± {:.2f}, Duration: {:.3f} \n".format(
            loss.mean().item(),
            100 * acc.mean().item(),
            100 * acc.std().item(),
            duration.mean().item(),
        )
    )
    if use_wandb:
        wandb.log(
            {
                "Val Loss": loss.mean().item(),
                "Test Accuracy": acc.mean().item(),
                "Duration": duration.mean().item(),
                "Val Loss std": loss.std().item(),
                "Test Accuracy std": acc.std().item(),
                "Duration std": duration.std().item(),
            }
        )

    return (
        loss.mean().item(),
        100 * acc.mean().item(),
        100 * acc.std().item(),
        num_params,
    )


def train(model, optimizer, data, preconditioner=None, lam=0.0):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    label = out.max(1)[1]
    label[data.train_mask] = data.y[data.train_mask]
    label.requires_grad = False

    loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
    loss += lam * F.nll_loss(out[~data.train_mask], label[~data.train_mask])

    loss.backward(retain_graph=True)

    if preconditioner:
        preconditioner.step(lam=lam)
    optimizer.step()
    return loss.item()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    for key in ["train", "val", "test"]:
        mask = data["{}_mask".format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs["{} loss".format(key)] = loss
        outs["{} acc".format(key)] = acc

    return outs
