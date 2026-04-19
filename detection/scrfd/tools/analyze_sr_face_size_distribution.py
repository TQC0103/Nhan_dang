import argparse
import ast
import csv
import glob
import json
import math
import os
import os.path as osp
import runpy

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from mmcv import Config  # type: ignore
except ImportError:  # pragma: no cover - optional fallback for offline analysis
    Config = None

from scale_prob_history_utils import average_scale_probs, load_scale_prob_records, normalize_probs


DEFAULT_BIN_EDGES = [0.0, 16.0, 32.0, 96.0, float('inf')]
DEFAULT_BIN_NAMES = ['tiny', 'small', 'medium', 'large']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze how SR / ASR changes the face-size distribution seen during SCRFD training.')
    parser.add_argument(
        '--ann-file',
        default='data/retinaface/train/labelv2.txt',
        help='RetinaFace/SCRFD labelv2 annotation file')
    parser.add_argument(
        '--baseline-config',
        default='configs/scrfd/scrfd_2.5g_80e_baseline.py',
        help='Baseline SCRFD config')
    parser.add_argument(
        '--improved-config',
        default='configs/scrfd/scrfd_2.5g_80e_asr_jsar.py',
        help='Improved SCRFD config')
    parser.add_argument(
        '--improved-state',
        required=True,
        help='ASR/online-scheduler work_dir, adaptive_sr dir, scale_prob_history.jsonl, or train log file')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Output directory')
    parser.add_argument(
        '--resize-size',
        type=float,
        default=640.0,
        help='Final square resize size used in training')
    parser.add_argument(
        '--bin-edges',
        type=float,
        nargs='+',
        default=DEFAULT_BIN_EDGES,
        help='Face-size bin edges in pixels')
    parser.add_argument(
        '--prob-mode',
        choices=['latest', 'mean_epochs'],
        default='mean_epochs',
        help='How to summarize adaptive scale probabilities')
    parser.add_argument(
        '--log-bins',
        type=int,
        default=80,
        help='Number of log-space bins for plotted histograms')
    parser.add_argument(
        '--min-plot-size',
        type=float,
        default=2.0,
        help='Minimum face size shown in plots')
    parser.add_argument(
        '--max-plot-size',
        type=float,
        default=512.0,
        help='Maximum face size shown in plots')
    args = parser.parse_args()
    return args


def load_config(config_path):
    if Config is not None:
        cfg = Config.fromfile(config_path)
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        return cfg
    return load_python_config_fallback(config_path)


def read_base_refs(config_path):
    with open(config_path, 'r', encoding='utf-8') as infile:
        source = infile.read()
    tree = ast.parse(source, filename=config_path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '_base_':
                return ast.literal_eval(node.value)
    return None


def normalize_base_list(base_refs):
    if base_refs is None:
        return []
    if isinstance(base_refs, (list, tuple)):
        return list(base_refs)
    return [base_refs]


def merge_namespaces(dst, src):
    for key, value in src.items():
        if key.startswith('__') and key.endswith('__'):
            continue
        dst[key] = value
    return dst


def load_python_config_fallback(config_path):
    config_path = osp.abspath(config_path)
    namespace = {}
    for base_ref in normalize_base_list(read_base_refs(config_path)):
        base_path = base_ref if osp.isabs(base_ref) else osp.normpath(osp.join(osp.dirname(config_path), base_ref))
        base_ns = load_python_config_fallback(base_path)
        merge_namespaces(namespace, base_ns)
    current_ns = runpy.run_path(config_path, init_globals=dict(namespace))
    merge_namespaces(namespace, current_ns)
    namespace['_filename'] = config_path
    return namespace


def get_cfg_value(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def find_random_square_crop(pipeline):
    for item in pipeline:
        item_type = item['type'] if isinstance(item, dict) else getattr(item, 'type', None)
        if item_type == 'RandomSquareCrop':
            return item
    raise ValueError('Could not find RandomSquareCrop in the train pipeline.')


def get_pipeline_crop_policy(cfg):
    data_cfg = get_cfg_value(cfg, 'data', None)
    if data_cfg is not None:
        train_cfg = get_cfg_value(data_cfg, 'train', None)
        pipeline = get_cfg_value(train_cfg, 'pipeline', None) if train_cfg is not None else None
    else:
        pipeline = None
    if pipeline is None:
        pipeline = get_cfg_value(cfg, 'train_pipeline', None)
    if pipeline is None:
        raise ValueError('Could not find train pipeline in config.')
    crop_cfg = find_random_square_crop(pipeline)
    crop_choice = [float(x) for x in crop_cfg['crop_choice']]
    crop_weights = crop_cfg.get('crop_choice_weights', None)
    if crop_weights is None:
        crop_weights = [1.0 / len(crop_choice) for _ in crop_choice]
    else:
        crop_weights = normalize_probs(crop_weights)
    redistribution_cfg = crop_cfg.get('adaptive_sr', get_cfg_value(cfg, 'redistribution_cfg', None))
    warmup_epochs = 0
    if redistribution_cfg is not None:
        warmup_epochs = int(redistribution_cfg.get('ADAPTIVE_SR_WARMUP_EPOCHS', 0))
    return {
        'crop_choice': crop_choice,
        'crop_weights': crop_weights,
        'warmup_epochs': warmup_epochs,
        'redistribution_cfg': redistribution_cfg,
    }


def parse_labelv2(ann_file):
    images = []
    current = None
    with open(ann_file, 'r', encoding='utf-8') as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('#'):
                parts = line[1:].strip().split()
                if len(parts) < 3:
                    raise ValueError('Malformed label header: {}'.format(line))
                current = {
                    'filename': parts[0],
                    'width': float(parts[1]),
                    'height': float(parts[2]),
                    'face_sizes': [],
                }
                images.append(current)
                continue
            if current is None:
                raise ValueError('Found bbox line before image header in {}'.format(ann_file))
            values = [float(x) for x in line.split()]
            if len(values) < 4:
                continue
            if len(values) == 5 and int(values[4]) == 1:
                continue
            width = max(values[2] - values[0], 1e-6)
            height = max(values[3] - values[1], 1e-6)
            current['face_sizes'].append(math.sqrt(width * height))
    filtered = [item for item in images if item['face_sizes']]
    if not filtered:
        raise ValueError('No valid faces found in {}'.format(ann_file))
    return filtered


def load_improved_probs(state_path, fallback_candidates, mode, warmup_epochs):
    records, resolved_source = load_scale_prob_records(
        state_path,
        fallback_candidates=fallback_candidates,
    )
    if mode == 'mean_epochs':
        candidates, probs, usable_records = average_scale_probs(
            records,
            warmup_epochs=warmup_epochs,
        )
        if [float(v) for v in candidates] != [float(v) for v in fallback_candidates]:
            raise ValueError('Resolved scale candidates do not match the improved config crop_choice.')
        return np.asarray(probs, dtype=np.float64), {
            'source': 'train_log_mean',
            'resolved_source': resolved_source,
            'num_records': len(usable_records),
            'warmup_epochs_skipped': warmup_epochs,
        }

    epoch_records = [record for record in records if int(record.get('epoch', 0)) > int(warmup_epochs)]
    if not epoch_records:
        epoch_records = records
    epoch_records = sorted(epoch_records, key=lambda item: (int(item.get('epoch', 0)), int(item.get('iteration', 0))))
    latest = epoch_records[-1]
    candidates = latest.get('scale_candidates', fallback_candidates)
    if [float(v) for v in candidates] != [float(v) for v in fallback_candidates]:
        raise ValueError('Resolved scale candidates do not match the improved config crop_choice.')
    probs = np.asarray(latest.get('scale_probs', None), dtype=np.float64)
    probs = probs / probs.sum()
    return probs, {
        'source': 'train_log_latest',
        'resolved_source': resolved_source,
        'epoch': int(latest.get('epoch', 0)),
        'iteration': int(latest.get('iteration', 0)),
    }


def assign_bins(values, bin_edges):
    values = np.asarray(values, dtype=np.float64)
    bin_indices = np.digitize(values, bin_edges[1:-1], right=False)
    return np.clip(bin_indices, 0, len(bin_edges) - 2)


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return float(values[-1])
    threshold = quantile * cumulative[-1]
    idx = np.searchsorted(cumulative, threshold, side='left')
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def build_distribution(images, scale_candidates, scale_probs, resize_size, bin_edges):
    values = []
    weights = []
    orig_bin_weights = np.zeros((len(bin_edges) - 1, ), dtype=np.float64)
    aug_bin_weights = np.zeros((len(bin_edges) - 1, ), dtype=np.float64)
    transition = np.zeros((len(bin_edges) - 1, len(bin_edges) - 1), dtype=np.float64)
    original_tiny_total = 0.0
    original_tiny_promoted = 0.0
    original_small_total = 0.0
    original_small_promoted = 0.0

    scale_candidates = np.asarray(scale_candidates, dtype=np.float64)
    scale_probs = np.asarray(scale_probs, dtype=np.float64)

    for image in images:
        short_side = max(min(float(image['width']), float(image['height'])), 1.0)
        face_sizes = np.asarray(image['face_sizes'], dtype=np.float64)
        orig_bins = assign_bins(face_sizes, bin_edges)
        resize_factors = resize_size / (scale_candidates * short_side)
        augmented = face_sizes[:, None] * resize_factors[None, :]
        augmented_flat = augmented.reshape(-1)
        weights_flat = np.tile(scale_probs, face_sizes.shape[0])
        aug_bins = assign_bins(augmented_flat, bin_edges)

        values.append(augmented_flat)
        weights.append(weights_flat)

        orig_bin_weights += np.bincount(orig_bins, minlength=len(bin_edges) - 1)
        aug_bin_weights += np.bincount(
            aug_bins,
            weights=weights_flat,
            minlength=len(bin_edges) - 1)

        for orig_bin in range(len(bin_edges) - 1):
            mask = orig_bins == orig_bin
            if not np.any(mask):
                continue
            local_aug = augmented[mask].reshape(-1)
            local_weights = np.tile(scale_probs, int(mask.sum()))
            local_bins = assign_bins(local_aug, bin_edges)
            transition[orig_bin] += np.bincount(
                local_bins,
                weights=local_weights,
                minlength=len(bin_edges) - 1)

        tiny_mask = face_sizes < bin_edges[1]
        if np.any(tiny_mask):
            tiny_aug = augmented[tiny_mask]
            original_tiny_total += float(tiny_mask.sum())
            original_tiny_promoted += float((tiny_aug >= bin_edges[1]).astype(np.float64).dot(scale_probs).sum())

        small_mask = face_sizes < bin_edges[2]
        if np.any(small_mask):
            small_aug = augmented[small_mask]
            original_small_total += float(small_mask.sum())
            original_small_promoted += float((small_aug >= bin_edges[2]).astype(np.float64).dot(scale_probs).sum())

    values = np.concatenate(values, axis=0)
    weights = np.concatenate(weights, axis=0)
    weights = weights / weights.sum()

    summary = {
        'count_faces': int(sum(len(item['face_sizes']) for item in images)),
        'count_weighted_samples': float(weights.size),
        'bin_weights': aug_bin_weights.tolist(),
        'bin_ratios': (aug_bin_weights / max(aug_bin_weights.sum(), 1e-12)).tolist(),
        'transition_matrix': transition.tolist(),
        'promote_tiny_to_ge16_ratio': (
            float(original_tiny_promoted / original_tiny_total)
            if original_tiny_total > 0 else None),
        'promote_lt32_to_ge32_ratio': (
            float(original_small_promoted / original_small_total)
            if original_small_total > 0 else None),
        'p10': weighted_quantile(values, weights, 0.10),
        'p25': weighted_quantile(values, weights, 0.25),
        'p50': weighted_quantile(values, weights, 0.50),
        'p75': weighted_quantile(values, weights, 0.75),
        'p90': weighted_quantile(values, weights, 0.90),
        'mean': float(np.sum(values * weights)),
    }
    return values, weights, summary


def build_raw_distribution(images, bin_edges):
    values = np.concatenate([
        np.asarray(image['face_sizes'], dtype=np.float64)
        for image in images
    ], axis=0)
    weights = np.full((values.shape[0], ), 1.0 / max(values.shape[0], 1), dtype=np.float64)
    bins = assign_bins(values, bin_edges)
    bin_weights = np.bincount(bins, minlength=len(bin_edges) - 1).astype(np.float64)
    summary = {
        'count_faces': int(values.shape[0]),
        'bin_weights': bin_weights.tolist(),
        'bin_ratios': (bin_weights / max(bin_weights.sum(), 1e-12)).tolist(),
        'p10': weighted_quantile(values, weights, 0.10),
        'p25': weighted_quantile(values, weights, 0.25),
        'p50': weighted_quantile(values, weights, 0.50),
        'p75': weighted_quantile(values, weights, 0.75),
        'p90': weighted_quantile(values, weights, 0.90),
        'mean': float(np.sum(values * weights)),
    }
    return values, weights, summary


def save_json(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)


def write_summary_csv(path, summaries, bin_names):
    fieldnames = ['distribution', 'mean', 'p10', 'p25', 'p50', 'p75', 'p90',
                  'promote_tiny_to_ge16_ratio', 'promote_lt32_to_ge32_ratio']
    fieldnames.extend(['bin_{}'.format(name) for name in bin_names])
    with open(path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for name, summary in summaries.items():
            row = {
                'distribution': name,
                'mean': summary.get('mean'),
                'p10': summary.get('p10'),
                'p25': summary.get('p25'),
                'p50': summary.get('p50'),
                'p75': summary.get('p75'),
                'p90': summary.get('p90'),
                'promote_tiny_to_ge16_ratio': summary.get('promote_tiny_to_ge16_ratio'),
                'promote_lt32_to_ge32_ratio': summary.get('promote_lt32_to_ge32_ratio'),
            }
            for idx, bin_name in enumerate(bin_names):
                row['bin_{}'.format(bin_name)] = summary['bin_ratios'][idx]
            writer.writerow(row)


def write_transition_csv(path, matrix, bin_names):
    with open(path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['orig_bin'] + list(bin_names))
        for idx, row in enumerate(matrix):
            total = sum(row)
            normalized = [value / total if total > 0 else 0.0 for value in row]
            writer.writerow([bin_names[idx]] + normalized)


def plot_histogram(path, distributions, min_plot_size, max_plot_size, log_bins, bin_edges):
    edges = np.geomspace(min_plot_size, max_plot_size, log_bins)
    plt.figure(figsize=(10, 6))
    for name, color, values, weights in distributions:
        plt.hist(
            values,
            bins=edges,
            weights=weights,
            histtype='step',
            linewidth=2.0,
            label=name,
            color=color,
            density=False)
    for edge in bin_edges[1:-1]:
        if np.isfinite(edge):
            plt.axvline(edge, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    plt.xscale('log')
    plt.xlabel('Face size (sqrt(area) in pixels)')
    plt.ylabel('Weighted count')
    plt.title('Face-size distribution before / after SR policy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cdf(path, distributions, min_plot_size, max_plot_size):
    grid = np.geomspace(min_plot_size, max_plot_size, 400)
    plt.figure(figsize=(10, 6))
    for name, color, values, weights in distributions:
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        cdf = np.cumsum(weights)
        y = np.interp(grid, values, cdf, left=0.0, right=1.0)
        plt.plot(grid, y, linewidth=2.0, label=name, color=color)
    plt.xscale('log')
    plt.xlabel('Face size (sqrt(area) in pixels)')
    plt.ylabel('CDF')
    plt.title('CDF of face-size distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_bin_bars(path, bin_names, raw_summary, baseline_summary, improved_summary):
    x = np.arange(len(bin_names))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, raw_summary['bin_ratios'], width=width, label='Original')
    plt.bar(x, baseline_summary['bin_ratios'], width=width, label='Baseline SR')
    plt.bar(x + width, improved_summary['bin_ratios'], width=width, label='ASR+JSAR')
    plt.xticks(x, bin_names)
    plt.ylabel('Ratio')
    plt.title('Tiny / Small / Medium / Large distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_scale_probs(path, baseline_candidates, baseline_probs, improved_candidates, improved_probs):
    plt.figure(figsize=(10, 6))
    plt.plot(
        baseline_candidates,
        baseline_probs,
        marker='o',
        linewidth=2.0,
        label='Baseline SR')
    plt.plot(
        improved_candidates,
        improved_probs,
        marker='o',
        linewidth=2.0,
        label='ASR+JSAR')
    ticks = sorted(set([float(x) for x in baseline_candidates] + [float(x) for x in improved_candidates]))
    plt.xticks(ticks, ['{:.2f}'.format(v) for v in ticks], rotation=45)
    plt.ylabel('Probability')
    plt.xlabel('RandomSquareCrop scale candidate')
    plt.title('Scale sampling probabilities')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_transition_heatmap(path, matrix, bin_names, title):
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, np.maximum(row_sums, 1e-12))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(normalized, cmap='YlOrRd', vmin=0.0, vmax=max(normalized.max(), 1e-6))
    ax.set_xticks(np.arange(len(bin_names)))
    ax.set_xticklabels(bin_names)
    ax.set_yticks(np.arange(len(bin_names)))
    ax.set_yticklabels(bin_names)
    ax.set_xlabel('Augmented bin')
    ax.set_ylabel('Original bin')
    ax.set_title(title)
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, '{:.2f}'.format(normalized[i, j]), ha='center', va='center', color='black')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_markdown(path, metadata, raw_summary, baseline_summary, improved_summary, bin_names):
    lines = [
        '# SR Face-Size Distribution Analysis',
        '',
        'This analysis uses the original `labelv2` face boxes and models the training-time size change as:',
        '',
        '`augmented_face_size ~= original_face_size * resize_size / (crop_scale * min(image_w, image_h))`',
        '',
        'It does not simulate image content or face dropping from crop position. The goal is to compare the *size distribution pressure* induced by each sampling policy.',
        '',
        '## Inputs',
        '',
        '- Annotation file: `{}`'.format(metadata['ann_file']),
        '- Baseline config: `{}`'.format(metadata['baseline_config']),
        '- Improved config: `{}`'.format(metadata['improved_config']),
        '- Improved scale source: `{}`'.format(metadata['improved_scale_source']),
        '',
        '## Key Numbers',
        '',
        '| Distribution | mean | p50 | p90 | tiny | small | medium | large | tiny->>=16 | <32->>=32 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]

    def add_row(name, summary):
        lines.append(
            '| {} | {:.2f} | {:.2f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} | {} |'.format(
                name,
                summary['mean'],
                summary['p50'],
                summary['p90'],
                summary['bin_ratios'][0],
                summary['bin_ratios'][1],
                summary['bin_ratios'][2],
                summary['bin_ratios'][3],
                '-' if summary.get('promote_tiny_to_ge16_ratio') is None else '{:.4f}'.format(summary['promote_tiny_to_ge16_ratio']),
                '-' if summary.get('promote_lt32_to_ge32_ratio') is None else '{:.4f}'.format(summary['promote_lt32_to_ge32_ratio']),
            ))

    add_row('Original', raw_summary)
    add_row('Baseline SR', baseline_summary)
    add_row('ASR+JSAR', improved_summary)

    lines.extend([
        '',
        '## Interpretation Hints',
        '',
        '- If `ASR+JSAR` moves more mass out of the `tiny` bin and into `small` / `medium`, that supports the claim that tiny faces receive stronger supervision.',
        '- `tiny->>=16` is a direct proxy for how often originally tiny faces become at least small after augmentation.',
        '- `<32->>=32` is a stronger promotion proxy: small hard faces becoming medium-sized for training.',
        '- The transition heatmaps show whether the improved policy concentrates more probability on the `tiny -> small` and `tiny -> medium` routes.',
    ])

    with open(path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    baseline_cfg = load_config(args.baseline_config)
    improved_cfg = load_config(args.improved_config)

    baseline_policy = get_pipeline_crop_policy(baseline_cfg)
    improved_policy = get_pipeline_crop_policy(improved_cfg)
    improved_probs, improved_prob_meta = load_improved_probs(
        args.improved_state,
        fallback_candidates=improved_policy['crop_choice'],
        mode=args.prob_mode,
        warmup_epochs=improved_policy['warmup_epochs'],
    )
    baseline_probs = np.asarray(baseline_policy['crop_weights'], dtype=np.float64)
    baseline_probs = baseline_probs / baseline_probs.sum()

    images = parse_labelv2(args.ann_file)
    raw_values, raw_weights, raw_summary = build_raw_distribution(images, args.bin_edges)
    baseline_values, baseline_weights, baseline_summary = build_distribution(
        images,
        baseline_policy['crop_choice'],
        baseline_probs,
        args.resize_size,
        args.bin_edges,
    )
    improved_values, improved_weights, improved_summary = build_distribution(
        images,
        improved_policy['crop_choice'],
        improved_probs,
        args.resize_size,
        args.bin_edges,
    )

    metadata = {
        'ann_file': osp.abspath(args.ann_file),
        'baseline_config': osp.abspath(args.baseline_config),
        'improved_config': osp.abspath(args.improved_config),
        'improved_scale_source': improved_prob_meta,
        'baseline_scale_candidates': baseline_policy['crop_choice'],
        'improved_scale_candidates': improved_policy['crop_choice'],
        'baseline_scale_probs': baseline_probs.tolist(),
        'improved_scale_probs': improved_probs.tolist(),
        'resize_size': float(args.resize_size),
        'bin_edges': [
            float(x) if np.isfinite(x) else 'inf'
            for x in args.bin_edges
        ],
        'bin_names': DEFAULT_BIN_NAMES[:len(args.bin_edges) - 1],
    }

    save_json(
        osp.join(args.out_dir, 'face_size_distribution_analysis.json'),
        {
            'metadata': metadata,
            'raw_summary': raw_summary,
            'baseline_summary': baseline_summary,
            'improved_summary': improved_summary,
        })

    write_summary_csv(
        osp.join(args.out_dir, 'face_size_distribution_summary.csv'),
        {
            'original': raw_summary,
            'baseline_sr': baseline_summary,
            'asr_jsar': improved_summary,
        },
        metadata['bin_names'],
    )
    write_transition_csv(
        osp.join(args.out_dir, 'baseline_transition_matrix.csv'),
        baseline_summary['transition_matrix'],
        metadata['bin_names'],
    )
    write_transition_csv(
        osp.join(args.out_dir, 'asr_jsar_transition_matrix.csv'),
        improved_summary['transition_matrix'],
        metadata['bin_names'],
    )

    plot_histogram(
        osp.join(args.out_dir, 'face_size_histogram.png'),
        [
            ('Original', '#4C6A92', raw_values, raw_weights),
            ('Baseline SR', '#C87B2A', baseline_values, baseline_weights),
            ('ASR+JSAR', '#287D5F', improved_values, improved_weights),
        ],
        args.min_plot_size,
        args.max_plot_size,
        args.log_bins,
        args.bin_edges,
    )
    plot_cdf(
        osp.join(args.out_dir, 'face_size_cdf.png'),
        [
            ('Original', '#4C6A92', raw_values, raw_weights),
            ('Baseline SR', '#C87B2A', baseline_values, baseline_weights),
            ('ASR+JSAR', '#287D5F', improved_values, improved_weights),
        ],
        args.min_plot_size,
        args.max_plot_size,
    )
    plot_bin_bars(
        osp.join(args.out_dir, 'face_size_bin_ratios.png'),
        metadata['bin_names'],
        raw_summary,
        baseline_summary,
        improved_summary,
    )
    plot_scale_probs(
        osp.join(args.out_dir, 'scale_probabilities.png'),
        metadata['baseline_scale_candidates'],
        baseline_probs,
        metadata['improved_scale_candidates'],
        improved_probs,
    )
    plot_transition_heatmap(
        osp.join(args.out_dir, 'baseline_transition_heatmap.png'),
        baseline_summary['transition_matrix'],
        metadata['bin_names'],
        'Baseline SR transition: original bin -> augmented bin',
    )
    plot_transition_heatmap(
        osp.join(args.out_dir, 'asr_jsar_transition_heatmap.png'),
        improved_summary['transition_matrix'],
        metadata['bin_names'],
        'ASR+JSAR transition: original bin -> augmented bin',
    )

    write_markdown(
        osp.join(args.out_dir, 'analysis.md'),
        metadata,
        raw_summary,
        baseline_summary,
        improved_summary,
        metadata['bin_names'],
    )

    print('Wrote face-size distribution analysis to {}'.format(osp.abspath(args.out_dir)))


if __name__ == '__main__':
    main()
