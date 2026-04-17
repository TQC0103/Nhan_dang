import argparse
import csv
import glob
import json
import os
import os.path as osp

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_BIN_NAMES = ['tiny', 'small', 'medium', 'large']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot JSAR positive-assignment expansion over training epochs.')
    parser.add_argument(
        '--source',
        required=True,
        help='work_dir, adaptive_sr dir, or adaptive_sr/epoch_logs dir')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Output directory')
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_epoch_log_dir(source):
    source = osp.abspath(source)
    if osp.isdir(osp.join(source, 'adaptive_sr', 'epoch_logs')):
        return osp.join(source, 'adaptive_sr', 'epoch_logs')
    if osp.isdir(osp.join(source, 'epoch_logs')):
        return osp.join(source, 'epoch_logs')
    if osp.basename(source) == 'epoch_logs' and osp.isdir(source):
        return source
    raise FileNotFoundError('Could not find adaptive_sr/epoch_logs under {}'.format(source))


def load_epoch_records(epoch_log_dir):
    records = []
    for path in sorted(glob.glob(osp.join(epoch_log_dir, 'epoch_*.json'))):
        with open(path, 'r', encoding='utf-8') as infile:
            payload = json.load(infile)
        if 'jsar_before_hist' not in payload or 'jsar_after_hist' not in payload:
            continue
        records.append(payload)
    if not records:
        raise ValueError('No JSAR epoch logs found in {}'.format(epoch_log_dir))
    return records


def safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(numerator, np.maximum(denominator, 1e-12))


def plot_pos_counts(path, epochs, before_counts, after_counts, bin_idx, label):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, before_counts[:, bin_idx], linewidth=2.0, label='before JSAR')
    plt.plot(epochs, after_counts[:, bin_idx], linewidth=2.0, label='after JSAR')
    plt.xlabel('Epoch')
    plt.ylabel('Positive anchors')
    plt.title('JSAR positive counts for {}'.format(label))
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_boost_ratios(path, epochs, boost_ratios, bin_names):
    plt.figure(figsize=(10, 6))
    for idx, bin_name in enumerate(bin_names):
        plt.plot(epochs, boost_ratios[:, idx], linewidth=2.0, label=bin_name)
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.0)
    plt.xlabel('Epoch')
    plt.ylabel('after / before positive ratio')
    plt.title('JSAR boost ratio by face-size bin')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pos_per_gt(path, epochs, before_per_gt, after_per_gt, bin_idx, label):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, before_per_gt[:, bin_idx], linewidth=2.0, label='before JSAR')
    plt.plot(epochs, after_per_gt[:, bin_idx], linewidth=2.0, label='after JSAR')
    plt.xlabel('Epoch')
    plt.ylabel('positives / GT')
    plt.title('Positive anchors per GT for {}'.format(label))
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_final_boost(path, final_boost, bin_names):
    plt.figure(figsize=(8, 5))
    x = np.arange(len(bin_names))
    plt.bar(x, final_boost, color=['#C75C5C', '#E6B566', '#75A3D1', '#88B884'])
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.0)
    plt.xticks(x, bin_names)
    plt.ylabel('after / before positive ratio')
    plt.title('Final-epoch JSAR boost ratio')
    for idx, value in enumerate(final_boost):
        plt.text(idx, value + 0.02, '{:.2f}x'.format(value), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_csv(path, records, bin_names):
    fieldnames = ['epoch']
    for prefix in ('gt', 'jsar_before', 'jsar_after', 'boost_ratio', 'before_per_gt', 'after_per_gt'):
        fieldnames.extend(['{}_{}'.format(prefix, name) for name in bin_names])
    with open(path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {'epoch': record['epoch']}
            for idx, bin_name in enumerate(bin_names):
                row['gt_{}'.format(bin_name)] = record['gt_hist'][idx]
                row['jsar_before_{}'.format(bin_name)] = record['jsar_before_hist'][idx]
                row['jsar_after_{}'.format(bin_name)] = record['jsar_after_hist'][idx]
                row['boost_ratio_{}'.format(bin_name)] = record['boost_ratio'][idx]
                row['before_per_gt_{}'.format(bin_name)] = record['before_per_gt'][idx]
                row['after_per_gt_{}'.format(bin_name)] = record['after_per_gt'][idx]
            writer.writerow(row)


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    epoch_log_dir = resolve_epoch_log_dir(args.source)
    records = load_epoch_records(epoch_log_dir)
    bin_names = records[0].get('bin_names', DEFAULT_BIN_NAMES)
    bin_names = list(bin_names[:len(records[0]['jsar_before_hist'])])

    epochs = np.asarray([int(record['epoch']) for record in records], dtype=np.int64)
    gt_hist = np.asarray([record['gt_hist'] for record in records], dtype=np.float64)
    before_counts = np.asarray([record['jsar_before_hist'] for record in records], dtype=np.float64)
    after_counts = np.asarray([record['jsar_after_hist'] for record in records], dtype=np.float64)

    boost_ratios = safe_ratio(after_counts, before_counts)
    before_per_gt = safe_ratio(before_counts, gt_hist)
    after_per_gt = safe_ratio(after_counts, gt_hist)

    enriched_records = []
    for idx, record in enumerate(records):
        copied = dict(record)
        copied['boost_ratio'] = boost_ratios[idx].tolist()
        copied['before_per_gt'] = before_per_gt[idx].tolist()
        copied['after_per_gt'] = after_per_gt[idx].tolist()
        enriched_records.append(copied)

    save_csv(osp.join(args.out_dir, 'jsar_assignment_history.csv'), enriched_records, bin_names)
    save_json(
        osp.join(args.out_dir, 'jsar_assignment_summary.json'),
        {
            'epoch_log_dir': epoch_log_dir,
            'bin_names': bin_names,
            'final_epoch': int(epochs[-1]),
            'final_gt_hist': gt_hist[-1].tolist(),
            'final_before_hist': before_counts[-1].tolist(),
            'final_after_hist': after_counts[-1].tolist(),
            'final_boost_ratio': boost_ratios[-1].tolist(),
            'final_before_per_gt': before_per_gt[-1].tolist(),
            'final_after_per_gt': after_per_gt[-1].tolist(),
        })

    if 'tiny' in bin_names:
        tiny_idx = bin_names.index('tiny')
        plot_pos_counts(
            osp.join(args.out_dir, 'jsar_tiny_pos_counts.png'),
            epochs,
            before_counts,
            after_counts,
            tiny_idx,
            'tiny faces',
        )
        plot_pos_per_gt(
            osp.join(args.out_dir, 'jsar_tiny_pos_per_gt.png'),
            epochs,
            before_per_gt,
            after_per_gt,
            tiny_idx,
            'tiny faces',
        )

    if 'small' in bin_names:
        small_idx = bin_names.index('small')
        plot_pos_counts(
            osp.join(args.out_dir, 'jsar_small_pos_counts.png'),
            epochs,
            before_counts,
            after_counts,
            small_idx,
            'small faces',
        )
        plot_pos_per_gt(
            osp.join(args.out_dir, 'jsar_small_pos_per_gt.png'),
            epochs,
            before_per_gt,
            after_per_gt,
            small_idx,
            'small faces',
        )

    plot_boost_ratios(
        osp.join(args.out_dir, 'jsar_boost_ratios.png'),
        epochs,
        boost_ratios,
        bin_names,
    )
    plot_final_boost(
        osp.join(args.out_dir, 'jsar_final_boost_ratio.png'),
        boost_ratios[-1],
        bin_names,
    )

    print('Wrote JSAR assignment plots to {}'.format(osp.abspath(args.out_dir)))


if __name__ == '__main__':
    main()
