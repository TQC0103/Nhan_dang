import argparse
import csv
import json
import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scale_prob_history_utils import dedupe_epoch_records, load_scale_prob_records


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot scale-probability trajectories from SCRFD train logs.')
    parser.add_argument(
        '--source',
        required=True,
        help='work_dir, adaptive_sr dir, scale_prob_history.jsonl, or text train log')
    parser.add_argument(
        '--config',
        default=None,
        help='Optional config file used to recover crop_choice when the source log lacks scale_candidates')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Output directory')
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=0,
        help='Hide epochs <= warmup from the main epoch-end plot')
    parser.add_argument(
        '--include-non-epoch-end',
        action='store_true',
        help='Also plot non-epoch-end points if the history contains them')
    args = parser.parse_args()
    return args


def load_fallback_candidates(config_path):
    if not config_path:
        return None
    from mmcv import Config
    cfg = Config.fromfile(config_path)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    pipeline = cfg.data.train.pipeline if 'data' in cfg and 'train' in cfg.data else cfg.train_pipeline
    for item in pipeline:
        item_type = item['type'] if isinstance(item, dict) else getattr(item, 'type', None)
        if item_type == 'RandomSquareCrop':
            return [float(v) for v in item['crop_choice']]
    return None


def save_json(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)


def write_csv(path, records, scale_candidates):
    fieldnames = ['epoch', 'iteration', 'record_type', 'scheduler']
    fieldnames.extend(['scale_{:.4f}'.format(scale) for scale in scale_candidates])
    with open(path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                'epoch': int(record.get('epoch', 0)),
                'iteration': int(record.get('iteration', 0)),
                'record_type': record.get('record_type', ''),
                'scheduler': record.get('scheduler', ''),
            }
            for idx, scale in enumerate(scale_candidates):
                row['scale_{:.4f}'.format(scale)] = float(record['scale_probs'][idx])
            writer.writerow(row)


def plot_epoch_end(path, epoch_records, scale_candidates, warmup_epochs):
    plot_records = [
        record for record in epoch_records
        if int(record.get('epoch', 0)) > int(warmup_epochs)
    ]
    if not plot_records:
        plot_records = epoch_records

    epochs = [int(record.get('epoch', 0)) for record in plot_records]
    plt.figure(figsize=(10, 6))
    for idx, scale in enumerate(scale_candidates):
        values = [float(record['scale_probs'][idx]) for record in plot_records]
        plt.plot(epochs, values, marker='o', linewidth=2.0, label='scale={:.2f}'.format(scale))
    if warmup_epochs > 0:
        plt.axvline(warmup_epochs, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.title('Scale probability history (epoch end)')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_all_updates(path, records, scale_candidates):
    steps = list(range(1, len(records) + 1))
    plt.figure(figsize=(10, 6))
    for idx, scale in enumerate(scale_candidates):
        values = [float(record['scale_probs'][idx]) for record in records]
        plt.plot(steps, values, marker='o', linewidth=1.8, label='scale={:.2f}'.format(scale))
    plt.xlabel('Logged update index')
    plt.ylabel('Probability')
    plt.title('Scale probability history (all logged updates)')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_summary(records, scale_candidates, resolved_source):
    epoch_records = dedupe_epoch_records(records)
    latest = epoch_records[-1]
    return {
        'resolved_source': resolved_source,
        'num_records': len(records),
        'num_epoch_records': len(epoch_records),
        'scale_candidates': [float(v) for v in scale_candidates],
        'latest_epoch': int(latest.get('epoch', 0)),
        'latest_iteration': int(latest.get('iteration', 0)),
        'latest_scale_probs': [float(v) for v in latest.get('scale_probs', [])],
        'scheduler': latest.get('scheduler', ''),
        'record_type': latest.get('record_type', ''),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fallback_candidates = load_fallback_candidates(args.config)
    records, resolved_source = load_scale_prob_records(
        args.source,
        fallback_candidates=fallback_candidates,
    )
    epoch_records = dedupe_epoch_records(records)
    if not epoch_records:
        raise ValueError('No epoch-level scale-probability records found.')

    scale_candidates = epoch_records[0].get('scale_candidates', fallback_candidates)
    if not scale_candidates:
        raise ValueError('scale_candidates are missing. Pass --config to recover crop_choice.')

    # Normalize and validate lengths.
    scale_candidates = [float(v) for v in scale_candidates]
    for record in records:
        record['scale_probs'] = [float(v) for v in record['scale_probs']]
        if len(record['scale_probs']) != len(scale_candidates):
            raise ValueError(
                'scale_probs length mismatch for epoch {} in {}'.format(
                    record.get('epoch'), resolved_source))

    write_csv(
        osp.join(args.out_dir, 'scale_prob_history.csv'),
        epoch_records,
        scale_candidates,
    )

    summary = build_summary(epoch_records, scale_candidates, resolved_source)
    save_json(osp.join(args.out_dir, 'scale_prob_history_summary.json'), summary)

    plot_epoch_end(
        osp.join(args.out_dir, 'scale_prob_history_epoch_end.png'),
        epoch_records,
        scale_candidates,
        args.warmup_epochs,
    )

    if args.include_non_epoch_end and len(records) != len(epoch_records):
        plot_all_updates(
            osp.join(args.out_dir, 'scale_prob_history_all_updates.png'),
            records,
            scale_candidates,
        )

    print('Wrote scale-probability plots to {}'.format(osp.abspath(args.out_dir)))


if __name__ == '__main__':
    main()
