import argparse
import json
import math
import os
import os.path as osp
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_run_paths(experiment_root: str, run_name: str) -> Dict[str, str]:
    run_root = osp.join(experiment_root, run_name)
    return {
        'root': run_root,
        'metrics_baseline': osp.join(run_root, 'metrics', 'baseline_results_summary.json'),
        'metrics_improved': osp.join(run_root, 'metrics', 'asr_jsar_results_summary.json'),
        'face_size': osp.join(run_root, 'face_size_analysis', 'face_size_distribution_analysis.json'),
        'jsar': osp.join(run_root, 'jsar_assignment', 'jsar_assignment_summary.json'),
        'hard_subset': osp.join(run_root, 'hard_subset_analysis', 'hard_subset_analysis.json'),
        'scale_hist': osp.join(run_root, 'scale_prob_history', 'scale_prob_history_summary.json'),
    }


def compute_scale_stats(scale_candidates: List[float], scale_probs: List[float]) -> Dict[str, float]:
    weighted_inv_scale = float(sum((p / s) for s, p in zip(scale_candidates, scale_probs)))
    weighted_scale = float(sum((p * s) for s, p in zip(scale_candidates, scale_probs)))
    zoom_in_mass = float(sum(p for s, p in zip(scale_candidates, scale_probs) if s < 1.0))
    zoom_out_mass = float(sum(p for s, p in zip(scale_candidates, scale_probs) if s > 1.0))
    extreme_zoom_in_mass = float(sum(p for s, p in zip(scale_candidates, scale_probs) if s <= 0.6))
    extreme_zoom_out_mass = float(sum(p for s, p in zip(scale_candidates, scale_probs) if s >= 2.0))
    return {
        'expected_inverse_scale': weighted_inv_scale,
        'expected_scale': weighted_scale,
        'zoom_in_mass': zoom_in_mass,
        'zoom_out_mass': zoom_out_mass,
        'extreme_zoom_in_mass': extreme_zoom_in_mass,
        'extreme_zoom_out_mass': extreme_zoom_out_mass,
        'min_scale': float(min(scale_candidates)),
        'max_scale': float(max(scale_candidates)),
        'max_magnification': float(max(1.0 / s for s in scale_candidates)),
        'min_magnification': float(min(1.0 / s for s in scale_candidates)),
    }


def collect_run_summary(run_name: str, paths: Dict[str, str]) -> Dict:
    baseline_metrics = load_json(paths['metrics_baseline'])
    improved_metrics = load_json(paths['metrics_improved'])
    face_size = load_json(paths['face_size'])
    jsar = load_json(paths['jsar'])
    hard_subset = load_json(paths['hard_subset']) if osp.exists(paths['hard_subset']) else None
    scale_hist = load_json(paths['scale_hist'])

    baseline_scale_stats = compute_scale_stats(
        face_size['metadata']['baseline_scale_candidates'],
        face_size['metadata']['baseline_scale_probs'])
    improved_scale_stats = compute_scale_stats(
        face_size['metadata']['improved_scale_candidates'],
        face_size['metadata']['improved_scale_probs'])

    return {
        'name': run_name,
        'baseline_metrics': baseline_metrics,
        'improved_metrics': improved_metrics,
        'face_size': face_size,
        'jsar': jsar,
        'hard_subset': hard_subset,
        'scale_hist': scale_hist,
        'baseline_scale_stats': baseline_scale_stats,
        'improved_scale_stats': improved_scale_stats,
    }


def plot_metric_bars(runs: List[Dict], out_path: str):
    metrics = ['easy_AP', 'medium_AP', 'hard_AP', 'mAP']
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.5))

    series = [
        ('Default baseline', runs[0]['baseline_metrics'], -1.5 * width, '#4c78a8'),
        ('Default ASR+JSAR', runs[0]['improved_metrics'], -0.5 * width, '#f58518'),
        ('Paper SR12 baseline', runs[1]['baseline_metrics'], 0.5 * width, '#54a24b'),
        ('Paper SR12 ASR+JSAR', runs[1]['improved_metrics'], 1.5 * width, '#e45756'),
    ]
    for label, source, offset, color in series:
        vals = [source[m] for m in metrics]
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.65, 0.93)
    ax.set_ylabel('AP')
    ax.set_title('WIDERFace metrics across scale sets')
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_asr_delta(runs: List[Dict], out_path: str):
    metrics = ['easy_AP', 'medium_AP', 'hard_AP', 'mAP']
    labels = ['easy', 'medium', 'hard', 'mAP']
    default_delta = [runs[0]['improved_metrics'][m] - runs[0]['baseline_metrics'][m] for m in metrics]
    paper_delta = [runs[1]['improved_metrics'][m] - runs[1]['baseline_metrics'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.32
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, default_delta, width, label='Default scale set', color='#4c78a8')
    bars2 = ax.bar(x + width / 2, paper_delta, width, label='Paper SR12 scale set', color='#e45756')
    for bars in (bars1, bars2):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + (0.001 if val >= 0 else -0.001),
                    f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('ASR+JSAR - Baseline')
    ax.set_title('Metric deltas from ASR+JSAR')
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scale_pool_geometry(runs: List[Dict], out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, run in enumerate(runs):
        meta = run['face_size']['metadata']
        base_candidates = np.array(meta['baseline_scale_candidates'], dtype=float)
        base_probs = np.array(meta['baseline_scale_probs'], dtype=float)
        imp_candidates = np.array(meta['improved_scale_candidates'], dtype=float)
        imp_probs = np.array(meta['improved_scale_probs'], dtype=float)

        ax = axes[0, col]
        ax.plot(base_candidates, 1.0 / base_candidates, marker='o', label='candidate magnification 1/scale')
        ax.set_title(f"{run['name']}: candidate pool geometry")
        ax.set_xlabel('scale candidate')
        ax.set_ylabel('relative face magnification')
        ax.grid(True, linestyle='--', alpha=0.25)

        ax2 = axes[1, col]
        width = 0.035 if len(base_candidates) <= 10 else 0.03
        ax2.bar(base_candidates - width / 2, base_probs, width, label='baseline uniform', color='#b0b0b0')
        ax2.bar(imp_candidates + width / 2, imp_probs, width, label='ASR mean prob', color='#4c78a8')
        ax2.set_title(f"{run['name']}: scale probability profile")
        ax2.set_xlabel('scale candidate')
        ax2.set_ylabel('probability')
        ax2.grid(True, linestyle='--', alpha=0.25)
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_distribution_pressure(runs: List[Dict], out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    run_labels = ['Default baseline', 'Default ASR+JSAR', 'Paper baseline', 'Paper ASR+JSAR']

    tiny_ratios = [
        runs[0]['face_size']['baseline_summary']['bin_ratios'][0],
        runs[0]['face_size']['improved_summary']['bin_ratios'][0],
        runs[1]['face_size']['baseline_summary']['bin_ratios'][0],
        runs[1]['face_size']['improved_summary']['bin_ratios'][0],
    ]
    p50s = [
        runs[0]['face_size']['baseline_summary']['p50'],
        runs[0]['face_size']['improved_summary']['p50'],
        runs[1]['face_size']['baseline_summary']['p50'],
        runs[1]['face_size']['improved_summary']['p50'],
    ]
    promote_tiny = [
        runs[0]['face_size']['baseline_summary']['promote_tiny_to_ge16_ratio'],
        runs[0]['face_size']['improved_summary']['promote_tiny_to_ge16_ratio'],
        runs[1]['face_size']['baseline_summary']['promote_tiny_to_ge16_ratio'],
        runs[1]['face_size']['improved_summary']['promote_tiny_to_ge16_ratio'],
    ]
    inv_scale = [
        runs[0]['baseline_scale_stats']['expected_inverse_scale'],
        runs[0]['improved_scale_stats']['expected_inverse_scale'],
        runs[1]['baseline_scale_stats']['expected_inverse_scale'],
        runs[1]['improved_scale_stats']['expected_inverse_scale'],
    ]

    def bar_plot(ax, values, title, ylabel, color):
        x = np.arange(len(run_labels))
        bars = ax.bar(x, values, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=20, ha='right')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8)

    bar_plot(axes[0, 0], tiny_ratios, 'Tiny-face ratio after SR simulation', 'ratio', '#4c78a8')
    bar_plot(axes[0, 1], p50s, 'Median face size after SR simulation', 'pixels', '#f58518')
    bar_plot(axes[1, 0], promote_tiny, 'Tiny -> >=16 px promotion ratio', 'ratio', '#54a24b')
    bar_plot(axes[1, 1], inv_scale, 'Expected magnification E[1/scale]', 'relative factor', '#e45756')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_jsar_tiny_supervision(runs: List[Dict], out_path: str):
    labels = ['Default', 'Paper SR12']
    before = [runs[0]['jsar']['final_before_per_gt'][0], runs[1]['jsar']['final_before_per_gt'][0]]
    after = [runs[0]['jsar']['final_after_per_gt'][0], runs[1]['jsar']['final_after_per_gt'][0]]
    boost = [runs[0]['jsar']['final_boost_ratio'][0], runs[1]['jsar']['final_boost_ratio'][0]]

    x = np.arange(len(labels))
    width = 0.3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    bars1 = axes[0].bar(x - width / 2, before, width, label='before JSAR', color='#b0b0b0')
    bars2 = axes[0].bar(x + width / 2, after, width, label='after JSAR', color='#4c78a8')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('positives per tiny GT')
    axes[0].set_title('Tiny-face supervision density')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.25)
    for bars in (bars1, bars2):
        for bar in bars:
            val = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, val, f'{val:.2f}',
                         ha='center', va='bottom', fontsize=8)

    bars3 = axes[1].bar(x, boost, width=0.42, color='#e45756')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('after / before')
    axes[1].set_title('Tiny-face JSAR boost ratio')
    axes[1].grid(axis='y', linestyle='--', alpha=0.25)
    for bar in bars3:
        val = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, val, f'{val:.3f}',
                     ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hard_size_gain(run: Dict, out_path: str, title: str):
    if not run.get('hard_subset'):
        return
    bins = run['hard_subset']['size_bins']
    labels = [item['size_bin'] for item in bins]
    deltas = [item['recall_proxy_delta'] for item in bins]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(x, deltas, color=['#e45756' if d < 0 else '#54a24b' for d in deltas])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('recall proxy delta')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (0.005 if val >= 0 else -0.005),
                f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hard_size_gain_compare(runs: List[Dict], out_path: str):
    if not all(run.get('hard_subset') for run in runs):
        return
    labels = [item['size_bin'] for item in runs[0]['hard_subset']['size_bins']]
    default_deltas = [item['recall_proxy_delta'] for item in runs[0]['hard_subset']['size_bins']]
    paper_deltas = [item['recall_proxy_delta'] for item in runs[1]['hard_subset']['size_bins']]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars1 = ax.bar(x - width / 2, default_deltas, width, label='Default scale set', color='#4c78a8')
    bars2 = ax.bar(x + width / 2, paper_deltas, width, label='Paper SR12 scale set', color='#e45756')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('recall proxy delta')
    ax.set_title('Hard-subset recall gain by face-size bin')
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend()
    for bars in (bars1, bars2):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + (0.005 if val >= 0 else -0.005),
                    f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_box(ax, x, y, w, h, text, facecolor='#f8f8f8', edgecolor='#444'):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.02',
                           linewidth=1.2, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=11, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=14,
                            linewidth=1.4, color='#333')
    ax.add_patch(arrow)


def plot_asr_schematic(out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={'width_ratios': [1.2, 1.0]})
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    draw_box(ax, 0.02, 0.66, 0.24, 0.24, 'Epoch statistics\n$G_b, P_b, L^{cls}_b, L^{box}_b$', '#eef4fb')
    draw_box(ax, 0.34, 0.62, 0.28, 0.32,
             "Difficulty terms\n"
             r"$r_b=\max(0,1-\min(\frac{P_b}{\max(G_b,1)},1))$" "\n"
             r"$c_b=\frac{L^{cls}_b}{\max(P_b,1)}$" "\n"
             r"$u_b=\frac{L^{box}_b}{\max(P_b,1)}$",
             '#fdf4e7')
    draw_box(ax, 0.70, 0.62, 0.26, 0.32,
             "Bin difficulty\n"
             r"$d_b \propto \mathrm{norm}(r_b)$" "\n"
             r"$\quad +\ \mathrm{norm}(c_b)$" "\n"
             r"$\quad +\ \mathrm{norm}(u_b)$",
             '#eef8e8')
    draw_box(ax, 0.16, 0.18, 0.32, 0.24,
             "Scale support prior\n"
             r"$S(s,b)=\exp(-|\log s-\log \mu_b| / 0.45)$",
             '#eef8e8')
    draw_box(ax, 0.56, 0.16, 0.38, 0.28,
             "Probability update\n"
             r"$q(s)\propto \sum_b S(s,b)\,d_b$" "\n"
             r"$q'(s)=q(s)(1-Km)+m$" "\n"
             r"$p_{t+1}=\alpha p_t+(1-\alpha)q'$",
             '#fbeaea')
    draw_arrow(ax, 0.26, 0.78, 0.34, 0.78)
    draw_arrow(ax, 0.62, 0.78, 0.70, 0.78)
    draw_arrow(ax, 0.80, 0.62, 0.78, 0.44)
    draw_arrow(ax, 0.48, 0.30, 0.56, 0.30)
    ax.text(
        0.52,
        0.04,
        "ASR dùng phản hồi theo size bin để học lại phân phối scale. "
        "Nếu tiny/small bins vẫn khó, xác suất được dịch về các crop ratio "
        "có support lớn hơn cho các bin đó.",
        ha='center',
        va='bottom',
        fontsize=10,
        wrap=True,
    )

    ax2 = axes[1]
    scale_candidates = np.array([0.35, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], dtype=float)
    target_scale = np.array([1.8, 1.35, 1.0, 0.72], dtype=float)
    support = np.exp(-np.abs(np.log(scale_candidates[:, None]) - np.log(target_scale[None, :])) / 0.45)
    support[scale_candidates >= 1.0, 0] *= 1.15
    support[scale_candidates >= 1.0, 1] *= 1.05
    support[scale_candidates <= 1.0, 2] *= 1.05
    support[scale_candidates <= 1.0, 3] *= 1.15
    im = ax2.imshow(support, cmap='YlGnBu', aspect='auto')
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(['tiny', 'small', 'medium', 'large'])
    ax2.set_yticks(np.arange(scale_candidates.shape[0]))
    ax2.set_yticklabels([f'{s:.2f}' for s in scale_candidates])
    ax2.set_xlabel('face-size bin')
    ax2.set_ylabel('scale candidate')
    ax2.set_title('ASR support matrix $S(s,b)$')
    for i in range(support.shape[0]):
        for j in range(support.shape[1]):
            ax2.text(j, i, f'{support[i, j]:.2f}', ha='center', va='center', fontsize=8)
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label('support strength')
    ax2.text(
        -0.02,
        -0.22,
        "Trục dọc là scale candidate. Trục ngang là size bin. "
        "Giá trị lớn nghĩa là scale đó được ưu tiên hơn nếu bin tương ứng đang khó.",
        transform=ax2.transAxes,
        fontsize=9.5,
        va='top',
        wrap=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_jsar_schematic(out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={'width_ratios': [1.0, 1.15]})

    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    gt = Rectangle((0.38, 0.28), 0.24, 0.24, linewidth=2, edgecolor='#333', facecolor='none')
    ax.add_patch(gt)
    ax.text(0.50, 0.55, 'tiny GT', ha='center', va='bottom', fontsize=12)

    base_pos = [(0.42, 0.34), (0.58, 0.44)]
    added_pos = [(0.31, 0.38), (0.68, 0.35), (0.50, 0.22)]
    neutral = [(0.25, 0.55), (0.75, 0.55), (0.22, 0.24), (0.78, 0.22), (0.50, 0.68)]

    for x, y in neutral:
        ax.add_patch(Circle((x, y), 0.022, facecolor='#d8d8d8', edgecolor='#666', linewidth=1))
    for x, y in base_pos:
        ax.add_patch(Circle((x, y), 0.028, facecolor='#4c78a8', edgecolor='white', linewidth=1.5))
    for x, y in added_pos:
        ax.add_patch(Circle((x, y), 0.028, facecolor='#f58518', edgecolor='white', linewidth=1.5))

    ax.text(0.14, 0.87, 'ATSS positives', color='#4c78a8', fontsize=11, fontweight='bold')
    ax.text(0.14, 0.81, 'JSAR added positives', color='#f58518', fontsize=11, fontweight='bold')
    ax.text(0.50, 0.08,
            'JSAR giữ lại positives gốc của ATSS (xanh) và bổ sung thêm\n'
            'một số anchor gần tiny GT (cam) khi GT đó đang thiếu supervision.',
            ha='center', va='bottom', fontsize=10, wrap=True)

    ax2 = axes[1]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    draw_box(ax2, 0.04, 0.66, 0.40, 0.26,
             "Threshold relaxation\n"
             r"$\tau_g = \mu_g + \sigma_g - \Delta_g$" "\n"
             r"$\Delta_g=\delta_{tiny},\ \delta_{small},\ 0$",
             '#eef4fb')
    draw_box(ax2, 0.54, 0.66, 0.40, 0.26,
             "Center gating\n"
             r"$c_g=\mathrm{clip}(\frac{\mathrm{size}_g}{T_{tiny}},1,\rho)$" "\n"
             r"$\mathrm{valid}(a,g): d_{min}(a,g)\cdot c_g > t_{center}$",
             '#fdf4e7')
    draw_box(ax2, 0.04, 0.28, 0.40, 0.26,
             "Fallback score\n"
             r"$\mathrm{score}(a,g)=\mathrm{IoU}(a,g)$" "\n"
             r"$\qquad -\ 0.05\cdot\frac{\mathrm{dist}(a,g)}{\max(\mathrm{size}_g,1)}$",
             '#eef8e8')
    draw_box(ax2, 0.54, 0.28, 0.40, 0.26,
             "Hybrid fallback rule\n"
             r"$N_{pos}(g) < N_{min}$  $\Rightarrow$  add top-$k$" "\n"
             "background anchors by score",
             '#fbeaea')
    draw_arrow(ax2, 0.44, 0.79, 0.54, 0.79)
    draw_arrow(ax2, 0.24, 0.66, 0.24, 0.54)
    draw_arrow(ax2, 0.74, 0.66, 0.74, 0.54)
    ax2.text(
        0.50,
        0.07,
        "JSAR trong code hiện tại = nới threshold cho tiny/small GT + center gating size-aware\n"
        "và nếu vẫn thiếu positives thì lấy thêm top-k anchors nền gần nhất theo score.",
        ha='center',
        va='bottom',
        fontsize=10,
        wrap=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary_json(runs: List[Dict], out_path: str):
    payload = {
        'default_source_scale_set': {
            'baseline_metrics': runs[0]['baseline_metrics'],
            'improved_metrics': runs[0]['improved_metrics'],
            'baseline_scale_stats': runs[0]['baseline_scale_stats'],
            'improved_scale_stats': runs[0]['improved_scale_stats'],
            'face_size_baseline': runs[0]['face_size']['baseline_summary'],
            'face_size_improved': runs[0]['face_size']['improved_summary'],
            'jsar': runs[0]['jsar'],
        },
        'paper_sr12_scale_set': {
            'baseline_metrics': runs[1]['baseline_metrics'],
            'improved_metrics': runs[1]['improved_metrics'],
            'baseline_scale_stats': runs[1]['baseline_scale_stats'],
            'improved_scale_stats': runs[1]['improved_scale_stats'],
            'face_size_baseline': runs[1]['face_size']['baseline_summary'],
            'face_size_improved': runs[1]['face_size']['improved_summary'],
            'jsar': runs[1]['jsar'],
        },
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build report assets that compare default SCRFD scale set and paper SR12 scale set.')
    parser.add_argument(
        '--experiment-root',
        required=True,
        help='Root directory that contains default_source_scale_set and paper_sr12_scale_set.')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Directory for generated report assets. Defaults to <experiment_root>/report_assets.')
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_root = osp.abspath(args.experiment_root)
    out_dir = osp.abspath(args.out_dir or osp.join(experiment_root, 'report_assets'))
    ensure_dir(out_dir)

    default_run = collect_run_summary('default_source_scale_set', get_run_paths(experiment_root, 'default_source_scale_set'))
    paper_run = collect_run_summary('paper_sr12_scale_set', get_run_paths(experiment_root, 'paper_sr12_scale_set'))
    runs = [default_run, paper_run]

    plot_metric_bars(runs, osp.join(out_dir, 'metrics_by_scale_set.png'))
    plot_asr_delta(runs, osp.join(out_dir, 'asr_delta_by_scale_set.png'))
    plot_scale_pool_geometry(runs, osp.join(out_dir, 'scale_pool_geometry.png'))
    plot_distribution_pressure(runs, osp.join(out_dir, 'distribution_pressure.png'))
    plot_jsar_tiny_supervision(runs, osp.join(out_dir, 'jsar_tiny_supervision.png'))
    plot_hard_size_gain(default_run, osp.join(out_dir, 'default_hard_size_gain.png'),
                        'Default scale set: hard-subset recall gain by face-size bin')
    plot_hard_size_gain(paper_run, osp.join(out_dir, 'paper_hard_size_gain.png'),
                        'Paper SR12 scale set: hard-subset recall gain by face-size bin')
    plot_hard_size_gain_compare(runs, osp.join(out_dir, 'hard_size_gain_by_scale_set.png'))
    plot_asr_schematic(osp.join(out_dir, 'asr_schematic.png'))
    plot_jsar_schematic(osp.join(out_dir, 'jsar_schematic.png'))
    write_summary_json(runs, osp.join(out_dir, 'report_summary.json'))

    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    safe_message = f'Wrote report assets to {out_dir}'.encode(encoding, errors='backslashreplace').decode(encoding)
    print(safe_message)


if __name__ == '__main__':
    main()
