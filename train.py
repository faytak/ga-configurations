import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from math import cos, pi

import random
import numpy as np

torch.cuda.empty_cache()

def add_repo_parent_to_path():
    """Ensure `glgenn` package imports work when running this script from repo root.

    When executing `python glgenn/train.py` from the parent directory of the repo,
    Python can import `glgenn` directly. But when running from inside the repo root
    (this file's directory), we need to add the parent directory to `sys.path`.
    """
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent  # path to `glgenn/`
    parent_of_repo = repo_root.parent
    if str(parent_of_repo) not in sys.path:
        sys.path.append(str(parent_of_repo))


add_repo_parent_to_path()

from engineer.trainer.trainer import Trainer  # noqa: E402
from models.hull_glgmlp import ConvexHullGLGMLP  # noqa: E402
from models.nbody_glgenn_gnn import NBodyGNN_GA  # noqa: E402
from models.on_glg import OnGLGMLP  # noqa: E402
# from models.o5_glgmlp import O5GLGMLP  # noqa: E402
from models.lorentz_cggnn import LorentzCGGNN  # noqa: E402
from data.on_hull import ConvexHullDataset  # noqa: E402
from data.on_nbody import NBody  # noqa: E402
from data.on_regression import OnDataset  # noqa: E402
from data.top_tagging import TopTaggingDataset  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ConvexHullGLGMLP with Lightning Trainer wrapper")

    # Selection
    parser.add_argument("--dataset", type=str, default="convex_hull",
                        choices=["convex_hull", "nbody", "on_regression", "top_tagging"],
                        help="Which dataset to use")
    parser.add_argument("--model", type=str, default="convex_hull_glgmlp",
                        choices=["convex_hull_glgmlp", "nbody_glgenn_gnn", "on_glg", "o5_glgmlp", "lorentz_cggnn"],
                        help="Which model to use")
    parser.add_argument("--subspace_type", type=str, default="Q", #choices=["P", "Q", "A", "B", "Triple", "LG"],
                        help="Subspace to use")

    # Data
    parser.add_argument("--dataroot", type=str, default=None,
                        help="Path to data root. Required for convex_hull and nbody datasets")
    parser.add_argument("--num_samples", type=int, default=256, help="Number of training samples to load (if applicable)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    # Model
    parser.add_argument("--n", type=int, default=None, help="Clifford algebra dimension n (models that need it)")
    parser.add_argument("--in_features", type=int, default=16)
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--out_features", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=4)

    # Model-specific optional args
    parser.add_argument("--on_output_qtgp", type=int, default=8, help="OnGLG output channels of QTGP")
    parser.add_argument("--on_hidden_mlp_1", type=int, default=580)
    parser.add_argument("--on_hidden_mlp_2", type=int, default=580)
    parser.add_argument("--on_use_mlp", action="store_true", help="Use MLP head in OnGLG")
    parser.add_argument("--nb_hidden_features", type=int, default=28, help="Hidden size for NBodyGLGGNN")
    parser.add_argument("--nb_n_layers", type=int, default=3, help="Number of EGCL layers for NBodyGLGGNN")
    # Top tagging dataset sizes
    parser.add_argument("--tt_num_train", type=int, default=1024, help="TopTaggingDataset train samples")
    parser.add_argument("--tt_num_val", type=int, default=1024, help="TopTaggingDataset val samples")
    parser.add_argument("--tt_num_test", type=int, default=1024, help="TopTaggingDataset test samples")
    parser.add_argument("--CA_type", type=str, default="QT", help="TopTagging solver type")

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Scheduler (opt-in; default: None)
    parser.add_argument("--scheduler", type=str, default=None, choices=[None, "cosine", "steplr"], help="LR scheduler to use (default: none)")
    parser.add_argument("--step_lr_step", type=int, default=0, help="Step size for StepLR (used when --scheduler=steplr)")
    parser.add_argument("--step_lr_gamma", type=float, default=0.1, help="Gamma for StepLR (used when --scheduler=steplr)")

    # Trainer controls
    parser.add_argument("--max_steps", type=int, default=0, help="0 means unlimited in our wrapper (Lightning None)")
    parser.add_argument("--max_time", type=str, default=None, help="Max time as string e.g. '0-01:30:00' or '01:30:00'")
    # parser.add_argument("--val_check_interval", type=int, default=10)
    # parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--limit_val_batches", type=float, default=float("inf"))
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--eval_test_during_val", action="store_true", help="Also run test loader during validation and log as test_eval/*")
    parser.add_argument("--val_n_interval", type=int, default=50)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor per worker (if num_workers>0)")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="TensorBoard log directory")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for the TensorBoard logger")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint for the same run_name")

    return parser

def set_seed(seed: int):
    seed = int(seed)
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

def build_cosine_warmup_scheduler(optimizer: torch.optim.Optimizer, total_steps: int,
                                  warmup_ratio: float = 1.0 / 64.0,
                                  decay_ratio: float = 1.0 / 4.0):
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    decay_steps = max(1, int(decay_ratio * total_steps))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        elif step < warmup_steps + decay_steps:
            t = (step - warmup_steps) / float(decay_steps)
            return 0.5 * (1.0 + cos(pi * t))
        else:
            return 0.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def maybe_build_scheduler(optimizer: torch.optim.Optimizer, args, total_steps: int):
    # Default: None (match original behavior)
    if args.scheduler is None:
        return None
    if args.scheduler == "steplr":
        step_size = max(1, int(args.step_lr_step))
        gamma = float(args.step_lr_gamma)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if args.scheduler == "cosine":
        return build_cosine_warmup_scheduler(optimizer, total_steps)
    return None


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = build_arg_parser()
        args = parser.parse_args()

        # Derive reasonable intervals. Avoid zeros when max_steps is 0/unlimited.
        if args.max_steps and args.max_steps > 0:
            derived = max(1, args.max_steps // args.val_n_interval)
        else:
            derived = 64
        args.val_check_interval = derived
        args.log_interval = derived

    # Seed
    set_seed(args.seed)

    # Distributed setup: determine available GPUs and per-process batch size
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        per_process_batch_size = max(1, args.batch_size // num_gpus)
    else:
        per_process_batch_size = args.batch_size

    # Data root for dataset (only for datasets that need disk data)
    if args.dataset in ("convex_hull", "nbody", "top_tagging"):
        if not args.dataroot:
            raise ValueError("--dataroot is required for datasets: convex_hull, nbody, top_tagging")
        dataroot = args.dataroot
        if args.dataset == "nbody" and not dataroot.endswith(os.sep):
            dataroot = dataroot + os.sep  # NBody code concatenates strings
        os.environ["DATAROOT"] = dataroot

    # Data
    if args.dataset == "convex_hull":
        dataset = ConvexHullDataset(num_samples=args.num_samples, batch_size=per_process_batch_size)
        dataset_ctx = {}
        train_ds = dataset.train_dataset
        val_ds = dataset.val_dataset
        test_ds = dataset.test_dataset
    elif args.dataset == "nbody":
        dataset = NBody(num_samples=args.num_samples, batch_size=per_process_batch_size)
        dataset_ctx = {}
        train_ds = dataset.train_dataset
        val_ds = dataset.valid_dataset
        test_ds = dataset.test_dataset
    elif args.dataset == "on_regression":
        if args.n is None:
            raise ValueError("--n is required for on_regression dataset")
        dataset = OnDataset(n=args.n, num_train_samples=args.num_samples, num_test_samples=max(1, args.num_samples // 4), batch_size=per_process_batch_size)
        # dataset_ctx = {"ymean": dataset.ymean, "ystd": dataset.ystd}
        dataset_ctx = {"ymean": 0, "ystd": 1}
        train_ds = dataset.train_dataset
        val_ds = dataset.val_dataset
        test_ds = dataset.test_dataset
    elif args.dataset == "top_tagging":
        # Use provided class that constructs loaders internally
        dataset_ctx = {}
        top_ds = TopTaggingDataset(
            batch_size=per_process_batch_size,
            num_train=args.tt_num_train,
            num_val=args.tt_num_val,
            num_test=args.tt_num_test,
            num_workers = args.num_workers
        )
        train_loader = top_ds.train_loader()
        val_loader = top_ds.val_loader()
        test_loader = top_ds.test_loader()

        args.num_samples = args.tt_num_train
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    pin_memory = torch.cuda.is_available()
    common_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0 and args.prefetch_factor is not None and args.prefetch_factor > 0:
        common_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    if args.dataset != "top_tagging":
        train_loader = DataLoader(train_ds, batch_size=per_process_batch_size, shuffle=True, drop_last=True, **common_loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=per_process_batch_size, shuffle=False, drop_last=False, **common_loader_kwargs)
        test_loader = DataLoader(test_ds, batch_size=per_process_batch_size, shuffle=False, drop_last=False, **common_loader_kwargs)

    # Model
    if args.model == "convex_hull_glgmlp":
        if args.n is None:
            raise ValueError("--n is required for convex_hull_glgmlp")
        model = ConvexHullGLGMLP(
            n=args.n,
            in_features=args.in_features,
            hidden_features=args.hidden_features,
            out_features=args.out_features,
            num_layers=args.num_layers,
            subspace_type=args.subspace_type
        )
    elif args.model == "nbody_glgenn_gnn":
        model = NBodyGNN_GA(
            hidden_features=args.nb_hidden_features,
            n_layers=args.nb_n_layers,
            subspace_type=args.subspace_type
        )
    elif args.model == "on_glg":
        if args.n is None:
            raise ValueError("--n is required for on_glg")
        if "ymean" not in dataset_ctx:
            raise ValueError("on_glg model requires on_regression dataset to provide ymean/ystd")
        model = OnGLGMLP(
            n=args.n,
            ymean=dataset_ctx["ymean"],
            ystd=dataset_ctx["ystd"],
            subspace_type=args.subspace_type,
            output_qtgp=args.on_output_qtgp,
            hidden_mlp_1=args.on_hidden_mlp_1,
            hidden_mlp_2=args.on_hidden_mlp_2,
            if_mlp=args.on_use_mlp,
        )
    elif args.model == "o5_glgmlp":
        if "ymean" not in dataset_ctx:
            raise ValueError("o5_glgmlp model requires on_regression dataset to provide ymean/ystd")
        model = O5GLGMLP(ymean=dataset_ctx["ymean"], ystd=dataset_ctx["ystd"])
    elif args.model == "lorentz_cggnn":
        model = LorentzCGGNN(subspace_type = args.subspace_type)
    else:
        raise ValueError(f"Unknown model {args.model}")

    # Optimizer & optional scheduler
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters())
    # Build scheduler (default: cosine with warmup/decay based on total steps)
    total_steps = args.max_steps if (args.max_steps and args.max_steps > 0) else len(train_loader)
    scheduler = maybe_build_scheduler(optimizer, args, total_steps)

    # Trainer wrapper (Lightning under the hood)
    # Logger (TensorBoard)
    run_name = f"{args.dataset}_{args.model}_Ss{args.subspace_type}_nsm{args.num_samples}_nst{args.max_steps}_bs{args.batch_size}"
    if args.run_name is not None:
        run_name = f"{args.run_name}_{run_name}"
    
    # Optionally resume from the last checkpoint for the same run_name
    resume_ckpt_path = None
    logger_version = None
    if getattr(args, "resume", False):
        run_root = os.path.join(args.log_dir, run_name)
        if os.path.isdir(run_root):
            # find latest version_* folder
            versions = []
            for entry in os.listdir(run_root):
                if entry.startswith("version_"):
                    try:
                        versions.append(int(entry.split("_")[-1]))
                    except Exception:
                        pass
            if len(versions) > 0:
                latest_version = max(versions)
                candidate_dir = os.path.join(run_root, f"version_{latest_version}", "checkpoints")
                last_ckpt = os.path.join(candidate_dir, "last.ckpt")
                if os.path.exists(last_ckpt):
                    resume_ckpt_path = last_ckpt
                    logger_version = latest_version
                else:
                    # fallback: pick most recent .ckpt file if present
                    try:
                        ckpts = [
                            os.path.join(candidate_dir, f)
                            for f in os.listdir(candidate_dir)
                            if f.endswith(".ckpt")
                        ] if os.path.isdir(candidate_dir) else []
                        if len(ckpts) > 0:
                            ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                            resume_ckpt_path = ckpts[0]
                            logger_version = latest_version
                    except Exception:
                        pass
        if resume_ckpt_path is None:
            print(f"[resume] No checkpoint found for run '{run_name}'. Starting fresh.")

    if logger_version is not None:
        tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=run_name, version=logger_version)
    else:
        tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=run_name)

    trainer = Trainer(
        scheduler=scheduler,
        logger=tb_logger,
        max_steps=args.max_steps,
        max_time=args.max_time,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        print_interval=32,
        fast_dev_run=args.fast_dev_run,
        wandb=None,
        callbacks=list(),
        log_interval=args.log_interval,
        checkpoint=None,
        test_only=args.test_only,
        monitor_metric=f'val/{model.monitor_metric}',
        monitor_mode=model.monitor_mode,
        eval_test_during_val=getattr(args, "eval_test_during_val", False)
    )

    # Fit + optional test
    trainer.fit(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        ckpt_path=resume_ckpt_path if getattr(args, "resume", False) else None
        # test_loader=val_loader
    )


if __name__ == "__main__":
    main()


