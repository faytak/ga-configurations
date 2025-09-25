#!/usr/bin/env python3
"""
Plotting and aggregation utilities extracted from drow_plots.ipynb.

This module provides:
- collect_experiments: build a tidy DataFrame of final metric values per run/version
- add_subspace_column: parse subspace tokens from run names
- set_paper_style: apply publication-friendly Matplotlib style
- plot_by_subspace: metric vs num_samples, grouped by subspace
- plot_metric_vs_params_grouped_by_nsm: metric vs params, grouped by num_samples
- plot_by_subspace_per_step: metric/max_steps vs max_steps, grouped by subspace

The module dynamically loads analyzer helpers from the colocated
analyze_experiments.py to avoid import path issues.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from cycler import cycler

# Dynamically load analyzer helpers that live in the same directory
import importlib.util as _ilu

_THIS_DIR = Path(__file__).resolve().parent
_ANALYZER_PATH = _THIS_DIR / "analyze_experiments.py"
_spec = _ilu.spec_from_file_location("analyze_experiments_mod", str(_ANALYZER_PATH))
_an = _ilu.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_an)  # type: ignore

# Re-exported analyzer helpers
list_run_dirs = _an.list_run_dirs
parse_run_name = _an.parse_run_name
list_version_dirs = _an.list_version_dirs
extract_metrics_from_tensorboard_version = _an.extract_metrics_from_tensorboard_version
extract_metrics_from_csv_version = _an.extract_metrics_from_csv_version
calculate_statistics = _an.calculate_statistics


# -----------------------------------------------------------------------------
# Publication-style settings and palette
# -----------------------------------------------------------------------------

# Saving controls (can be overridden by callers)
SAVE_FIGS: bool = True
SAVE_FORMAT: str = "pdf"  # "pdf" | "png" | "eps"
SAVE_DPI: int = 300
SAVE_DIR: Path = _THIS_DIR.parent / "figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def set_paper_style() -> None:
    rcParams.update({
        # Sizing
        'figure.figsize': (6.0, 4.0),
        'figure.dpi': 150,
        'savefig.dpi': SAVE_DPI,
        'savefig.format': SAVE_FORMAT,
        'savefig.bbox': 'tight',
        # Fonts
        'font.size': 9,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Computer Modern Roman', 'DejaVu Serif', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'text.usetex': False,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        # Lines/markers
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'errorbar.capsize': 2.0,
        # Axes and ticks
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.facecolor': '#EEF1F6',
        'axes.edgecolor': '#E0E0E0',
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        # Legends
        'legend.frameon': False,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,
        'legend.borderaxespad': 0.4,
        'legend.facecolor': 'white',
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#DDDDDD',
        # Embed editable text in vector outputs
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    # Colorblind-safe default cycle (Okabe–Ito)
    rcParams['axes.prop_cycle'] = cycler(color=[
        '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#000000'
    ])


# Subspace style maps
SUBSPACE_PALETTE: Dict[str, str] = {
    'A': '#0072B2',      # blue
    'B': '#D55E00',      # vermillion
    'LG': '#009E73',     # green-ish
    r"$\Gamma$": '#009E73',
    'P': '#CC79A7',      # pink
    'Q': '#000000',      # black
    'Triple': '#E69F00', # orange
    'Tr-LG-Tr-LG-Tr': '#FF0000',
    'Triple-$\\Gamma$': '#FF0000',
}

SUBSPACE_MARKERS: Dict[str, str] = {
    'A': 'o',
    'B': 's',
    'LG': '^',
    r"$\Gamma$": '^',
    'P': 'D',
    'Q': 'v',
    'Triple': 'P',
}

SUBSPACE_LINESTYLES: Dict[str, Any] = {
    'A': '-',
    'B': '--',
    'Tr-LG-Tr-LG-Tr': '--',
    'Triple-$\\Gamma$': '--',
    'LG': '-.',
    r"$\Gamma$": '-.',
    'P': ':',
    'Q': (0, (3, 1, 1, 1)),   # dash-dot-dot
    'Triple': (0, (5, 2)),    # long dash
}

subspaces_to_exclude: List[str] = [
    'Q-A-A-A-A-Q-Q-Q-Q-Q',
    'Q-A-A-A-A-P-P-P-P-Q',
    'Q-A-A-A-A-Tr-Tr-Tr-Tr-Q',
    'A-A-Q-Tr-Q',
    'A-Q-A-Q-A',
    'Q-Tr-Q-Tr-Q',
    'Tr-Q-Tr-Q-Tr',
]


renamed_entities: Dict[str, str] = {
    'test/loss': 'Test Loss',
    'test/accuracy': 'Test Accuracy',
    'nbody_glgenn_gnn': 'O(6,0)-N-Body',
    'convex_hull_glgmlp': 'O(6,0)-Convex Hull',
    'on_glg': 'O(8,0)-Regression',
    'o5_glgmlp': 'O5 Experiment',
    'lorentz_cggnn': 'O(1,3)-Top Tagging',
    'LG': r"$\Gamma$",
}


# -----------------------------------------------------------------------------
# Data aggregation
# -----------------------------------------------------------------------------

def collect_experiments(
    log_dir: str,
    model_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    source: str = 'auto',
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
) -> pd.DataFrame:
    run_names = list_run_dirs(log_dir)
    if include_regex:
        try:
            rx_include = re.compile(include_regex)
            run_names = [r for r in run_names if rx_include.search(r)]
        except re.error:
            pass

    if exclude_regex:
        try:
            rx_exclude = re.compile(exclude_regex)
            run_names = [r for r in run_names if not rx_exclude.search(r)]
        except re.error:
            pass

    rows: List[Dict[str, Any]] = []
    for run_name in run_names:
        params = parse_run_name(run_name)
        if model_filter and params.get('model') != model_filter:
            continue
        if dataset_filter and params.get('dataset') != dataset_filter:
            continue

        run_path = os.path.join(log_dir, run_name)
        versions = list_version_dirs(run_path)
        if not versions:
            versions = [run_path]

        for version_dir in versions:
            metrics_this: Dict[str, List[float]] = {}

            if source in ('auto', 'tb'):
                metrics_this = extract_metrics_from_tensorboard_version(version_dir)
            if (not metrics_this) and source in ('auto', 'csv'):
                metrics_this = extract_metrics_from_csv_version(version_dir)

            if not metrics_this:
                continue

            for metric_name, values in metrics_this.items():
                if not values:
                    continue
                rows.append({
                    'run_name': run_name,
                    'version_dir': version_dir,
                    'metric': metric_name,
                    'final_value': float(values[-1]),
                    'num_samples': params.get('num_samples'),
                    'max_steps': params.get('max_steps'),
                    'batch_size': params.get('batch_size'),
                    'dataset': params.get('dataset'),
                    'model': params.get('model'),
                })

    df = pd.DataFrame(rows)
    return df


SUBSPACE_REGEX = re.compile(r"_Ss(?P<subspace>[^_]+)_nsm")


def add_subspace_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def extract_subspace(name: Any) -> Optional[str]:
        if not isinstance(name, str):
            return None
        m = SUBSPACE_REGEX.search(name)
        if not m:
            return None
        return m.group('subspace')

    df['subspace'] = df['run_name'].apply(extract_subspace)
    return df


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, base_name: str) -> None:
    if not SAVE_FIGS:
        return
    out_main = SAVE_DIR / f"{base_name}.{SAVE_FORMAT}"
    fig.savefig(out_main, bbox_inches='tight', transparent=(SAVE_FORMAT == 'pdf'))
    # Always also create EPS for compatibility
    out_eps = SAVE_DIR / f"{base_name}.eps"
    fig.savefig(out_eps, bbox_inches='tight', transparent=False)


def plot_by_subspace(df: pd.DataFrame, metric: str, model: str, normalize_by_nsm: bool = False) -> None:
    sub = df[(df['metric'] == metric) & (df['model'] == model)].copy()
    sub = sub.dropna(subset=['num_samples', 'subspace'])
    if sub.empty:
        print(f'No data for metric={metric}, model={model} after filtering for subspace and num_samples')
        return

    if normalize_by_nsm:
        sub['y'] = sub['final_value'] / sub['num_samples'].astype(float)
        y_label = renamed_entities.get(metric, metric)
    else:
        sub['y'] = sub['final_value']
        y_label = renamed_entities.get(metric, metric)

    agg = (
        sub.groupby(['subspace', 'num_samples'], dropna=False)['y']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    agg['num_samples'] = agg['num_samples'].astype(int)

    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    def get_style_for_subspace(s: str):
        color = SUBSPACE_PALETTE.get(s, default_colors[hash(s) % len(default_colors)] if default_colors else None)
        marker = SUBSPACE_MARKERS.get(s, 'o')
        linestyle = SUBSPACE_LINESTYLES.get(s, '-')
        label = s[2:] if s.startswith('Ss') else s
        if s == 'LG':
            label = r"$\Gamma$"
        elif s == 'Tr-LG-Tr-LG-Tr':
            label = r"Triple-$\Gamma$"
        label = f'{label}-configuration'
        return color, marker, linestyle, label

    subspaces = sorted(agg['subspace'].unique(), key=lambda x: (x != 'LG', x), reverse=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    present_x_values: set[int] = set()

    for s in subspaces:
        if s in subspaces_to_exclude:
            continue
        cur = agg[agg['subspace'] == s].sort_values('num_samples')
        x = cur['num_samples'].values
        y = cur['mean'].values
        present_x_values.update(x.tolist())
        yerr = cur['std'].values
        y = np.maximum(y, 1e-20)
        yerr = np.minimum(yerr, 0.99 * y)
        color, marker, linestyle, label = get_style_for_subspace(s)
        ax.errorbar(x, y, yerr=yerr, marker=marker, color=color, capsize=2, linestyle=linestyle, label=label)

    ax.set_title(f"{renamed_entities.get(model, model)}")
    ax.set_xlabel('Training set size')
    ax.set_ylabel(y_label)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Only show ticks for x values that are present in the data
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, NullFormatter
    present_x_sorted = sorted(present_x_values)
    ax.xaxis.set_major_locator(FixedLocator(present_x_sorted))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in present_x_sorted]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0, labelOnlyBase=False))

    if model in ('convex_hull', 'convex_hull_glgmlp', 'lorentz_cggnn'):
        from matplotlib.ticker import FuncFormatter, NullFormatter
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[i / 10.0 for i in range(1, 10)]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.2f}"))

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', width=0.8, length=3)
    ax.grid(which='major', linestyle='--', alpha=0.3)

    ax.legend(loc='best', frameon=True, handlelength=1.5)
    fig.tight_layout()

    base = f"{metric.replace('/', '_')}_vs_nsm_{model}{'_per_nsm' if normalize_by_nsm else ''}"
    _save_figure(fig, base)
    plt.show()


# Model params map used for params plots
model_params_by_subspace: Dict[str, Dict[str, Dict[str, int]]] = {
    "A": {
        "ConvexHullGLGMLP": {"total_params": 13697, "trainable_params": 13697},
        "NBodyGNN_GA": {"total_params": 49141, "trainable_params": 49141},
        "OnGLGMLP": {"total_params": 881, "trainable_params": 881},
        "LorentzCGGNN": {"total_params": 253010, "trainable_params": 253010},
    },
    "B": {
        "ConvexHullGLGMLP": {"total_params": 13697, "trainable_params": 13697},
        "NBodyGNN_GA": {"total_params": 49141, "trainable_params": 49141},
        "OnGLGMLP": {"total_params": 881, "trainable_params": 881},
        "LorentzCGGNN": {"total_params": 253010, "trainable_params": 253010},
    },
    "P": {
        "ConvexHullGLGMLP": {"total_params": 13185, "trainable_params": 13185},
        "NBodyGNN_GA": {"total_params": 47797, "trainable_params": 47797},
        "OnGLGMLP": {"total_params": 753, "trainable_params": 753},
        "LorentzCGGNN": {"total_params": 250962, "trainable_params": 250962},
    },
    "Q": {
        "ConvexHullGLGMLP": {"total_params": 28289, "trainable_params": 28289},
        "NBodyGNN_GA": {"total_params": 102621, "trainable_params": 102621},
        "OnGLGMLP": {"total_params": 2257, "trainable_params": 2257},
        "LorentzCGGNN": {"total_params": 268770, "trainable_params": 268770},
    },
    "Triple": {
        "ConvexHullGLGMLP": {"total_params": 21889, "trainable_params": 21889},
        "NBodyGNN_GA": {"total_params": 78233, "trainable_params": 78233},
        "OnGLGMLP": {"total_params": 1793, "trainable_params": 1793},
        "LorentzCGGNN": {"total_params": 263194, "trainable_params": 263194},
    },
    "LG": {
        "ConvexHullGLGMLP": {"total_params": 97761, "trainable_params": 97761},
        "NBodyGNN_GA": {"total_params": 185025, "trainable_params": 185025},
        "OnGLGMLP": {"total_params": 7905, "trainable_params": 7905},
        "LorentzCGGNN": {"total_params": 320594, "trainable_params": 320594},
    },
    "Tr-LG-Tr-LG-Tr": {
        "ConvexHullGLGMLP": {"total_params": 49537, "trainable_params": 49537},
    },
}


MODEL_KEY_MAP: Dict[str, str] = {
    'nbody_glgenn_gnn': 'NBodyGNN_GA',
    'convex_hull_glgmlp': 'ConvexHullGLGMLP',
    'on_glg': 'OnGLGMLP',
    'lorentz_cggnn': 'LorentzCGGNN',
}

NSM_LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]


def _attach_params_by_subspace(df: pd.DataFrame, model_key: str, use_trainable: bool = True) -> pd.DataFrame:
    df = df.copy()

    def get_params(s: Optional[str]) -> Optional[int]:
        if s is None or not isinstance(s, str):
            return None
        sub_dict = model_params_by_subspace.get(s)
        if not sub_dict:
            return None
        model_info = sub_dict.get(model_key)
        if not model_info:
            return None
        key = 'trainable_params' if use_trainable else 'total_params'
        return int(model_info.get(key, 0))

    df['params'] = df['subspace'].apply(get_params)
    return df


def plot_metric_vs_params_grouped_by_nsm(df: pd.DataFrame, metric: str, model: str, normalize_by_nsm: bool = False) -> None:
    from matplotlib import cm
    from matplotlib.lines import Line2D

    model_key = MODEL_KEY_MAP.get(model)
    if not model_key:
        print(f'Unknown model mapping for {model}')
        return

    sub = df[(df['metric'] == metric) & (df['model'] == model)].copy()
    sub = sub.dropna(subset=['num_samples', 'subspace'])
    if sub.empty:
        print(f'No data for metric={metric}, model={model}')
        return

    if normalize_by_nsm:
        sub['y'] = sub['final_value'] / sub['num_samples'].astype(float)
        y_label = f"{renamed_entities.get(metric, metric)}"
    else:
        sub['y'] = sub['final_value']
        y_label = f"{renamed_entities.get(metric, metric)}"

    sub = _attach_params_by_subspace(sub, model_key, use_trainable=True)
    sub = sub.dropna(subset=['params'])
    sub['params'] = sub['params'].astype(int)

    agg = (
        sub.groupby(['subspace', 'num_samples', 'params'], dropna=False)['y']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )

    unique_nsm = sorted(agg['num_samples'].unique())
    # Use a perceptually uniform colormap for num_samples groups
    # cmap = plt.colormaps.get_cmap('nipy_spectral')
    cmap = cm.get_cmap('nipy_spectral', len(unique_nsm)+6)
    # nsm_to_color = {nsm: cmap(i / max(1, len(unique_nsm) - 1)) for i, nsm in enumerate(unique_nsm)}
    # nsm_to_ls = {nsm: NSM_LINESTYLES[i % len(NSM_LINESTYLES)] for i, nsm in enumerate(unique_nsm)}
    nsm_to_color = {nsm: cmap(i) for i, nsm in enumerate(unique_nsm)}
    nsm_to_ls = {nsm: NSM_LINESTYLES[i % len(NSM_LINESTYLES)] for i, nsm in enumerate(unique_nsm)}

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    for nsm in unique_nsm:
        cur = agg[agg['num_samples'] == nsm].sort_values('params')
        if cur.empty:
            continue
        x = cur['params'].values
        y = cur['mean'].values
        yerr = cur['std'].values
        color = nsm_to_color[nsm]
        ls = nsm_to_ls[nsm]
        ax.errorbar(x, y, yerr=yerr, color=color, linestyle=ls, marker=None, label=f'n={int(nsm)}', capsize=2)

    # Overlay markers by subspace
    marker_map = {'A': 'o', 'B': 's', 'LG': '^', 'P': 'D', 'Q': 'v', 'Triple': 'P'}
    for _, row in agg.iterrows():
        ax.scatter(row['params'], row['mean'], s=35, marker=marker_map.get(row['subspace'], 'o'),
                   facecolor='white', edgecolor='black', linewidths=0.7, zorder=3)

    ax.set_title(f"{renamed_entities.get(model, model)}")
    ax.set_xlabel('Trainable parameters')
    ax.set_ylabel(y_label)

    ax.set_xscale('log')
    from matplotlib.ticker import AutoLocator, ScalarFormatter
    ax.yaxis.set_major_locator(AutoLocator())
    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(sf)

    if model == 'on_glg':
        from matplotlib.ticker import FuncFormatter
        ax.set_yscale('log')
        ax.set_xticks([2e3, 8e3], minor=True)
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, pos: '$2 \\times 10^3$' if np.isclose(x, 2000.0) else ('$8 \\times 10^3$' if np.isclose(x, 8000.0) else '')))
        ax.tick_params(axis='x', which='minor', labelsize=8.75, pad=4)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', width=0.8, length=3)
    ax.grid(which='major', linestyle='--', alpha=0.3)

    line_handles = [
        Line2D([0], [0], color=nsm_to_color[nsm], linestyle=nsm_to_ls[nsm], label=f'n={int(nsm)}')
        for nsm in unique_nsm
    ]

    legend_loc = 'best'
    if model in ('convex_hull_glgmlp', 'on_glg'):
        legend_loc = 'upper right'
        ax.legend(handles=line_handles, title='num_samples', loc=legend_loc, frameon=True, bbox_to_anchor=(0.95, 1))
    else:
        ax.legend(handles=line_handles, title='num_samples', loc=legend_loc, frameon=True)

    # Top x-axis labels with subspace names at their parameter counts
    param_to_subspaces = (
        agg.groupby('params')['subspace'].apply(lambda s: sorted(set(s))).to_dict()
    )
    abbrev = {'Triple': 'Triple', 'LG': '$\\Gamma$', 'Tr-LG-Tr-LG-Tr': 'Triple-$\\Gamma$'}
    ticks_top = sorted(param_to_subspaces.keys())
    labels_top = ["\n".join([abbrev.get(s, s) for s in param_to_subspaces[p]]) for p in ticks_top]
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax.get_xlim())
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, NullLocator
    ax_top.xaxis.set_major_locator(FixedLocator(ticks_top))
    ax_top.xaxis.set_major_formatter(FixedFormatter(labels_top))
    ax_top.xaxis.set_minor_locator(NullLocator())
    ax_top.xaxis.set_minor_formatter(NullFormatter())
    ax_top.tick_params(axis='x', labelsize=9, pad=2)
    for _lbl in ax_top.get_xticklabels():
        _lbl.set_multialignment('center')

    fig.tight_layout()
    base = f"{metric.replace('/', '_')}_vs_params_{model}{'_per_nsm' if normalize_by_nsm else ''}"
    _save_figure(fig, base)
    plt.show()


def plot_by_subspace_per_step(df: pd.DataFrame, metric: str, model: str) -> None:
    sub = df[(df['metric'] == metric) & (df['model'] == model)].copy()
    sub = sub.dropna(subset=['subspace', 'max_steps'])
    if sub.empty:
        print(f'No data for metric={metric}, model={model} after filtering for subspace and max_steps')
        return

    sub['y'] = sub['final_value'] / sub['max_steps'].astype(float)
    y_label = renamed_entities.get(metric, metric)

    agg = (
        sub.groupby(['subspace', 'max_steps'], dropna=False)['y']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    agg['max_steps'] = agg['max_steps'].astype(int)

    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    def get_style_for_subspace(s: str):
        color = SUBSPACE_PALETTE.get(s, default_colors[hash(s) % len(default_colors)] if default_colors else None)
        marker = SUBSPACE_MARKERS.get(s, 'o')
        linestyle = SUBSPACE_LINESTYLES.get(s, '-')
        label = s[2:] if s.startswith('Ss') else s
        if s == 'LG':
            label = "$\\Gamma$"
        elif s == 'Tr-LG-Tr-LG-Tr':
            label = "Tr-$\\Gamma$"
        label = f'{label}-configuration'
        return color, marker, linestyle, label

    subspaces = sorted(agg['subspace'].unique(), key=lambda x: (x != 'LG', x), reverse=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for s in subspaces:
        if s in subspaces_to_exclude:
            continue
        cur = agg[agg['subspace'] == s].sort_values('max_steps')
        x = cur['max_steps'].values
        y = cur['mean'].values
        yerr = cur['std'].values
        y = np.maximum(y, 1e-20)
        yerr = np.minimum(yerr, 0.99 * y)
        color, marker, linestyle, label = get_style_for_subspace(s)
        ax.errorbar(x, y, yerr=yerr, marker=marker, color=color, capsize=2, linestyle=linestyle, label=label)

    ax.set_title(f"{renamed_entities.get(model, model)}")
    ax.set_xlabel('Max training steps')
    ax.set_ylabel(y_label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0, labelOnlyBase=False))

    if model in ('convex_hull', 'convex_hull_glgmlp', 'lorentz_cggnn'):
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.2f}"))

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', width=0.8, length=3)
    ax.grid(which='major', linestyle='--', alpha=0.3)

    ax.legend(loc='best', frameon=True, handlelength=1.5)
    fig.tight_layout()

    base = f"{metric.replace('/', '_')}_per_step_vs_maxsteps_{model}"
    _save_figure(fig, base)
    plt.show()


__all__ = [
    'SAVE_FIGS', 'SAVE_FORMAT', 'SAVE_DPI', 'SAVE_DIR',
    'set_paper_style',
    'collect_experiments', 'add_subspace_column',
    'plot_by_subspace', 'plot_metric_vs_params_grouped_by_nsm', 'plot_by_subspace_per_step',
    'SUBSPACE_PALETTE', 'SUBSPACE_MARKERS', 'SUBSPACE_LINESTYLES',
    'renamed_entities', 'model_params_by_subspace',
]


