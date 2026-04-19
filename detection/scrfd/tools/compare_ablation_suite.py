import argparse
import csv
import glob
import json
import os
import os.path as osp


SUMMARY_METRICS = ('easy_AP', 'medium_AP', 'hard_AP', 'mAP')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate multiple SCRFD experiment results into one ablation report.')
    parser.add_argument(
        '--experiment',
        action='append',
        required=True,
        help='Experiment specification in the form name=results_dir_or_summary')
    parser.add_argument(
        '--baseline',
        default='baseline',
        help='Experiment name used as delta reference')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument(
        '--title',
        default='SCRFD 2.5G Ablation Study',
        help='Report title')
    args = parser.parse_args()
    return args


def parse_experiment_spec(spec):
    if '=' not in spec:
        raise ValueError('Invalid --experiment value: {}. Expected name=path'.format(spec))
    name, path = spec.split('=', 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError('Invalid --experiment value: {}'.format(spec))
    return name, path


def load_aps_summary(aps_path):
    with open(aps_path, 'r', encoding='utf-8') as infile:
        raw = infile.readline().strip()
    values = [float(item) for item in raw.split(',') if item.strip()]
    if len(values) != 3:
        raise ValueError('Expected 3 AP values in {}'.format(aps_path))
    return {
        'easy_AP': values[0],
        'medium_AP': values[1],
        'hard_AP': values[2],
        'mAP': sum(values) / 3.0,
        'source': aps_path,
    }


def resolve_summary_path(path):
    if osp.isdir(path):
        json_candidate = osp.join(path, 'results_summary.json')
        if osp.exists(json_candidate):
            return json_candidate
        aps_candidate = osp.join(path, 'aps')
        if osp.exists(aps_candidate):
            return aps_candidate
    return path


def load_summary(path):
    summary_path = resolve_summary_path(path)
    if not osp.exists(summary_path):
        raise FileNotFoundError(
            'Missing results summary: {}. Expected results_summary.json or aps.'.format(summary_path))
    if osp.basename(summary_path) == 'aps':
        return load_aps_summary(summary_path), summary_path
    with open(summary_path, 'r', encoding='utf-8') as infile:
        return json.load(infile), summary_path


def load_optional_json(path):
    if path and osp.exists(path):
        with open(path, 'r', encoding='utf-8') as infile:
            return json.load(infile)
    return {}


def find_latency_summary(result_root):
    candidates = [
        osp.join(result_root, 'latency_summary.json'),
        osp.join(result_root, 'latency', 'latency_summary.json'),
    ]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return None


def maybe_float(value):
    try:
        if value is None or value == '':
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def find_training_log_json(work_dir):
    if not work_dir or not osp.isdir(work_dir):
        return None
    candidates = sorted(glob.glob(osp.join(work_dir, '*.log.json')))
    if not candidates:
        return None
    return candidates[-1]


def load_training_summary(work_dir):
    log_json = find_training_log_json(work_dir)
    if not log_json:
        return {}

    train_entries = []
    with open(log_json, 'r', encoding='utf-8') as infile:
        for raw_line in infile:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if 'loss' not in entry:
                continue
            train_entries.append(entry)

    if not train_entries:
        return {}

    last_entry = max(
        train_entries,
        key=lambda entry: (int(entry.get('epoch', 0)), int(entry.get('iter', 0))))

    iter_times = [float(entry['time']) for entry in train_entries if 'time' in entry]
    last_times = iter_times[-50:] if iter_times else []
    return {
        'log_json': log_json,
        'final_epoch': int(last_entry.get('epoch', 0)),
        'final_iter': int(last_entry.get('iter', 0)),
        'final_loss': maybe_float(last_entry.get('loss')),
        'final_loss_cls': maybe_float(last_entry.get('loss_cls')),
        'final_loss_bbox': maybe_float(last_entry.get('loss_bbox')),
        'final_lr': maybe_float(last_entry.get('lr')),
        'avg_iter_time_s': (sum(iter_times) / len(iter_times)) if iter_times else None,
        'avg_last50_iter_time_s': (sum(last_times) / len(last_times)) if last_times else None,
        'train_hours': (sum(iter_times) / 3600.0) if iter_times else None,
    }


def format_float(value, digits=4):
    if value is None:
        return '-'
    return ('{:.%df}' % digits).format(float(value))


def build_row(name, result_path):
    summary, summary_path = load_summary(result_path)
    result_root = osp.dirname(summary_path) if osp.isfile(summary_path) else result_path
    latency_path = find_latency_summary(result_root)
    latency = load_optional_json(latency_path)

    checkpoint_path = summary.get('checkpoint')
    work_dir = osp.dirname(checkpoint_path) if checkpoint_path else None
    training = load_training_summary(work_dir)

    row = {
        'experiment': name,
        'result_root': result_root,
        'summary_path': summary_path,
        'config': summary.get('config'),
        'checkpoint': checkpoint_path,
        'work_dir': work_dir,
        'checkpoint_size_mb': latency.get('checkpoint_size_mb'),
        'latency_mean_ms': latency.get('mean_ms'),
        'ms_per_image': latency.get('ms_per_image'),
        'latency_p50_ms': latency.get('p50_ms'),
        'latency_p95_ms': latency.get('p95_ms'),
        'fps': latency.get('fps'),
        'peak_memory_mb': latency.get('peak_memory_mb'),
        'params_m': latency.get('params_m'),
        'flops_g': latency.get('flops_g'),
        'final_loss': training.get('final_loss'),
        'final_loss_cls': training.get('final_loss_cls'),
        'final_loss_bbox': training.get('final_loss_bbox'),
        'final_lr': training.get('final_lr'),
        'avg_iter_time_s': training.get('avg_iter_time_s'),
        'avg_last50_iter_time_s': training.get('avg_last50_iter_time_s'),
        'train_hours': training.get('train_hours'),
        'log_json': training.get('log_json'),
    }
    for metric in SUMMARY_METRICS:
        row[metric] = maybe_float(summary.get(metric))
    return row


def add_deltas(rows, baseline_name):
    baseline_row = None
    for row in rows:
        if row['experiment'] == baseline_name:
            baseline_row = row
            break
    if baseline_row is None:
        raise ValueError('Baseline experiment "{}" not found.'.format(baseline_name))

    for row in rows:
        row['delta_hard_AP'] = (
            None if row['hard_AP'] is None or baseline_row['hard_AP'] is None
            else row['hard_AP'] - baseline_row['hard_AP'])
        row['delta_mAP'] = (
            None if row['mAP'] is None or baseline_row['mAP'] is None
            else row['mAP'] - baseline_row['mAP'])
        row['delta_latency_mean_ms'] = (
            None if row['latency_mean_ms'] is None or baseline_row['latency_mean_ms'] is None
            else row['latency_mean_ms'] - baseline_row['latency_mean_ms'])
        row['delta_fps'] = (
            None if row['fps'] is None or baseline_row['fps'] is None
            else row['fps'] - baseline_row['fps'])
    return baseline_row


def write_csv(rows, out_path):
    fieldnames = [
        'experiment',
        'result_root',
        'work_dir',
        'easy_AP',
        'medium_AP',
        'hard_AP',
        'mAP',
        'delta_hard_AP',
        'delta_mAP',
        'latency_mean_ms',
        'ms_per_image',
        'latency_p50_ms',
        'latency_p95_ms',
        'delta_latency_mean_ms',
        'fps',
        'delta_fps',
        'peak_memory_mb',
        'params_m',
        'flops_g',
        'checkpoint_size_mb',
        'final_loss',
        'final_loss_cls',
        'final_loss_bbox',
        'final_lr',
        'avg_iter_time_s',
        'avg_last50_iter_time_s',
        'train_hours',
        'config',
        'checkpoint',
        'summary_path',
        'log_json',
    ]
    with open(out_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows, baseline_name, title, out_path):
    lines = [
        '# {}'.format(title),
        '',
        'Baseline reference: `{}`'.format(baseline_name),
        '',
        '## Accuracy',
        '',
        '| Experiment | easy_AP | medium_AP | hard_AP | mAP | delta_hard_AP | delta_mAP |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            '| {experiment} | {easy} | {medium} | {hard} | {map_} | {dhard} | {dmap} |'.format(
                experiment=row['experiment'],
                easy=format_float(row['easy_AP']),
                medium=format_float(row['medium_AP']),
                hard=format_float(row['hard_AP']),
                map_=format_float(row['mAP']),
                dhard=format_float(row['delta_hard_AP']),
                dmap=format_float(row['delta_mAP']),
            ))

    lines.extend([
        '',
        '## Runtime',
        '',
        '| Experiment | mean ms | ms/img | p50 ms | p95 ms | FPS | delta_mean ms | delta_FPS | peak mem MB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in rows:
        lines.append(
            '| {experiment} | {mean_ms} | {ms_per_img} | {p50_ms} | {p95_ms} | {fps} | {dmean_ms} | {dfps} | {peak_mem} |'.format(
                experiment=row['experiment'],
                mean_ms=format_float(row['latency_mean_ms'], 3),
                ms_per_img=format_float(row['ms_per_image'], 3),
                p50_ms=format_float(row['latency_p50_ms'], 3),
                p95_ms=format_float(row['latency_p95_ms'], 3),
                fps=format_float(row['fps'], 2),
                dmean_ms=format_float(row['delta_latency_mean_ms'], 3),
                dfps=format_float(row['delta_fps'], 2),
                peak_mem=format_float(row['peak_memory_mb'], 1),
            ))

    lines.extend([
        '',
        '## Model And Training',
        '',
        '| Experiment | params M | FLOPs G | ckpt MB | final loss | cls loss | bbox loss | avg iter s | train h |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in rows:
        lines.append(
            '| {experiment} | {params_m} | {flops_g} | {ckpt_mb} | {loss} | {loss_cls} | {loss_bbox} | {iter_s} | {train_h} |'.format(
                experiment=row['experiment'],
                params_m=format_float(row['params_m'], 3),
                flops_g=format_float(row['flops_g'], 3),
                ckpt_mb=format_float(row['checkpoint_size_mb'], 2),
                loss=format_float(row['final_loss'], 4),
                loss_cls=format_float(row['final_loss_cls'], 4),
                loss_bbox=format_float(row['final_loss_bbox'], 4),
                iter_s=format_float(row['avg_iter_time_s'], 4),
                train_h=format_float(row['train_hours'], 2),
            ))

    with open(out_path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()

    rows = []
    for spec in args.experiment:
        name, path = parse_experiment_spec(spec)
        rows.append(build_row(name, path))

    baseline_row = add_deltas(rows, args.baseline)

    out_dir = osp.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    suite = {
        'title': args.title,
        'baseline': baseline_row['experiment'],
        'rows': rows,
    }

    json_path = osp.join(out_dir, 'ablation_suite.json')
    with open(json_path, 'w', encoding='utf-8') as outfile:
        json.dump(suite, outfile, indent=2, sort_keys=True)

    csv_path = osp.join(out_dir, 'ablation_suite.csv')
    write_csv(rows, csv_path)

    md_path = osp.join(out_dir, 'ablation_suite.md')
    write_markdown(rows, args.baseline, args.title, md_path)

    print('Wrote ablation report to {}'.format(out_dir))


if __name__ == '__main__':
    main()
