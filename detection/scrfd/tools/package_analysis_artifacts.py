import argparse
import json
import os
import os.path as osp
import shutil
import sys
import zipfile


def parse_args():
    parser = argparse.ArgumentParser(
        description='Package SCRFD training/evaluation artifacts for offline analysis.')
    parser.add_argument(
        '--experiment',
        action='append',
        nargs='+',
        required=True,
        metavar='ARG',
        help=('Experiment spec: NAME WORK_DIR RESULT_DIR [CONFIG]. '
              'Repeat --experiment for multiple runs.'))
    parser.add_argument('--out-dir', required=True, help='Output directory for packaged artifacts.')
    parser.add_argument(
        '--copy-checkpoints',
        choices=['latest', 'all', 'none'],
        default='latest',
        help='Which checkpoints to copy from each work_dir.')
    parser.add_argument(
        '--include-predictions',
        action='store_true',
        default=True,
        help='Copy results/*/predictions for hard-subset analysis.')
    parser.add_argument(
        '--skip-predictions',
        action='store_true',
        help='Do not copy results/*/predictions.')
    parser.add_argument(
        '--include-dataset-metadata',
        action='store_true',
        default=True,
        help='Copy train/labelv2.txt, val/labelv2.txt, and val/gt/*.mat into the bundle.')
    parser.add_argument(
        '--skip-dataset-metadata',
        action='store_true',
        help='Do not copy train/labelv2.txt, val/labelv2.txt, and val/gt/*.mat.')
    parser.add_argument(
        '--train-ann',
        default='data/retinaface/train/labelv2.txt',
        help='Path to train labelv2.txt relative to repo root or absolute.')
    parser.add_argument(
        '--val-ann',
        default='data/retinaface/val/labelv2.txt',
        help='Path to val labelv2.txt relative to repo root or absolute.')
    parser.add_argument(
        '--gt-dir',
        default='data/retinaface/val/gt',
        help='Path to WIDERFace val gt directory relative to repo root or absolute.')
    parser.add_argument(
        '--zip-name',
        default='analysis_artifacts_bundle.zip',
        help='Bundle filename written under --out-dir.')
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_path(path, repo_root):
    if not path:
        return None
    if osp.isabs(path):
        return path
    return osp.join(repo_root, path)


def copy_file(src, dst):
    ensure_dir(osp.dirname(dst))
    shutil.copy2(src, dst)


def copy_tree(src_root, dst_root):
    copied = []
    if not osp.isdir(src_root):
        return copied
    for root, _, files in os.walk(src_root):
        for file_name in files:
            src_path = osp.join(root, file_name)
            rel_path = osp.relpath(src_path, src_root)
            dst_path = osp.join(dst_root, rel_path)
            copy_file(src_path, dst_path)
            copied.append((src_path, dst_path))
    return copied


def collect_matching_files(root_dir, patterns):
    matches = []
    if not osp.isdir(root_dir):
        return matches
    names = set(patterns.get('names', []))
    suffixes = tuple(patterns.get('suffixes', []))
    prefixes = tuple(patterns.get('prefixes', []))
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if names and file_name in names:
                matches.append(osp.join(root, file_name))
                continue
            if suffixes and file_name.endswith(suffixes):
                matches.append(osp.join(root, file_name))
                continue
            if prefixes and file_name.startswith(prefixes):
                matches.append(osp.join(root, file_name))
    return sorted(set(matches))


def parse_experiments(raw_experiments):
    experiments = []
    for raw in raw_experiments:
        if len(raw) not in (3, 4):
            raise ValueError(
                'Each --experiment must have 3 or 4 values: NAME WORK_DIR RESULT_DIR [CONFIG]. '
                'Got: {}'.format(raw))
        name = raw[0]
        work_dir = raw[1]
        result_dir = raw[2]
        config = raw[3] if len(raw) == 4 else None
        experiments.append({
            'name': name,
            'work_dir': work_dir,
            'result_dir': result_dir,
            'config': config,
        })
    return experiments


def copy_selected_files(file_paths, src_root, dst_root):
    copied = []
    for src_path in file_paths:
        rel_path = osp.relpath(src_path, src_root)
        dst_path = osp.join(dst_root, rel_path)
        copy_file(src_path, dst_path)
        copied.append((src_path, dst_path))
    return copied


def package_experiment(experiment, args, repo_root, out_dir):
    name = experiment['name']
    work_dir = resolve_path(experiment['work_dir'], repo_root)
    result_dir = resolve_path(experiment['result_dir'], repo_root)
    config_path = resolve_path(experiment['config'], repo_root) if experiment.get('config') else None
    dst_root = osp.join(out_dir, 'experiments', name)
    work_dst = osp.join(dst_root, 'work_dir')
    result_dst = osp.join(dst_root, 'results')

    ensure_dir(work_dst)
    ensure_dir(result_dst)

    manifest = {
        'name': name,
        'work_dir': work_dir,
        'result_dir': result_dir,
        'config': config_path,
        'copied': [],
        'available': {},
        'analysis_ready': {},
    }

    work_patterns = {
        'names': ['latest.pth'],
        'suffixes': ['.log', '.log.json', '.json', '.jsonl'],
        'prefixes': ['epoch_'],
    }
    work_files = collect_matching_files(work_dir, work_patterns)
    selected_work = []
    for src_path in work_files:
        file_name = osp.basename(src_path)
        if src_path.endswith('.pth'):
            if args.copy_checkpoints == 'none':
                continue
            if args.copy_checkpoints == 'latest' and file_name != 'latest.pth':
                continue
            selected_work.append(src_path)
            continue
        selected_work.append(src_path)
    for src, dst in copy_selected_files(selected_work, work_dir, work_dst):
        manifest['copied'].append({'source': src, 'target': dst})

    result_patterns = {
        'names': ['aps', 'results_summary.json', 'results_summary.csv', 'latency_summary.json'],
        'suffixes': ['.json', '.csv', '.md', '.png', '.log'],
        'prefixes': [],
    }
    result_files = collect_matching_files(result_dir, result_patterns)
    selected_result = []
    for src_path in result_files:
        if (not args.include_predictions) and ('{}predictions{}'.format(os.sep, os.sep) in '{}{}'.format(src_path, os.sep)):
            continue
        selected_result.append(src_path)
    for src, dst in copy_selected_files(selected_result, result_dir, result_dst):
        manifest['copied'].append({'source': src, 'target': dst})

    pred_dir = osp.join(result_dir, 'predictions')
    if args.include_predictions and osp.isdir(pred_dir):
        for src, dst in copy_tree(pred_dir, osp.join(result_dst, 'predictions')):
            manifest['copied'].append({'source': src, 'target': dst})

    if config_path and osp.isfile(config_path):
        config_dst = osp.join(dst_root, 'config', osp.basename(config_path))
        copy_file(config_path, config_dst)
        manifest['copied'].append({'source': config_path, 'target': config_dst})

    copied_targets = [item['target'] for item in manifest['copied']]
    manifest['available'] = {
        'latest_checkpoint': any(path.endswith('{}latest.pth'.format(os.sep)) for path in copied_targets),
        'train_log': any(path.endswith('.log') for path in copied_targets),
        'train_log_json': any(path.endswith('.log.json') for path in copied_targets),
        'adaptive_sr_history': any(path.endswith('scale_prob_history.jsonl') for path in copied_targets),
        'results_summary': any(path.endswith('results_summary.json') for path in copied_targets),
        'aps': any(osp.basename(path) == 'aps' for path in copied_targets),
        'latency_summary': any(path.endswith('latency_summary.json') for path in copied_targets),
        'predictions': any('{}predictions{}'.format(os.sep, os.sep) in '{}{}'.format(path, os.sep) for path in copied_targets),
        'config': any('{}config{}'.format(os.sep, os.sep) in '{}{}'.format(path, os.sep) for path in copied_targets),
    }
    manifest['analysis_ready'] = {
        'plot_scale_prob_history': (
            manifest['available']['adaptive_sr_history']
            or manifest['available']['train_log']
            or manifest['available']['train_log_json']
        ),
        'metric_comparison': manifest['available']['results_summary'] or manifest['available']['aps'],
        'latency_comparison': manifest['available']['latency_summary'],
        'hard_subset_analysis': manifest['available']['predictions'],
    }
    return manifest


def package_dataset_metadata(args, repo_root, out_dir):
    dataset_manifest = {
        'included': False,
        'copied': [],
        'available': {},
    }
    if args.skip_dataset_metadata or not args.include_dataset_metadata:
        return dataset_manifest

    dataset_root = osp.join(out_dir, 'dataset_metadata')
    ensure_dir(dataset_root)

    specs = [
        ('train_labelv2', resolve_path(args.train_ann, repo_root), osp.join(dataset_root, 'train', 'labelv2.txt')),
        ('val_labelv2', resolve_path(args.val_ann, repo_root), osp.join(dataset_root, 'val', 'labelv2.txt')),
    ]
    gt_dir = resolve_path(args.gt_dir, repo_root)

    for label, src, dst in specs:
        if src and osp.isfile(src):
            copy_file(src, dst)
            dataset_manifest['copied'].append({'label': label, 'source': src, 'target': dst})

    if gt_dir and osp.isdir(gt_dir):
        for src, dst in copy_tree(gt_dir, osp.join(dataset_root, 'val', 'gt')):
            dataset_manifest['copied'].append({'label': 'val_gt', 'source': src, 'target': dst})

    dataset_manifest['included'] = True
    copied_targets = [item['target'] for item in dataset_manifest['copied']]
    dataset_manifest['available'] = {
        'train_labelv2': any(path.endswith('{}train{}labelv2.txt'.format(os.sep, os.sep)) for path in copied_targets),
        'val_labelv2': any(path.endswith('{}val{}labelv2.txt'.format(os.sep, os.sep)) for path in copied_targets),
        'val_gt': any('{}val{}gt{}'.format(os.sep, os.sep, os.sep) in '{}{}'.format(path, os.sep) for path in copied_targets),
    }
    return dataset_manifest


def write_readme(out_dir, manifest):
    lines = [
        '# Analysis Artifacts Bundle',
        '',
        'This bundle is intended to let you re-run the analysis scripts later without keeping the full training workspace online.',
        '',
        '## Included',
        '',
    ]
    for exp in manifest['experiments']:
        lines.extend([
            '- `{}`'.format(exp['name']),
            '  - `plot_scale_prob_history`: {}'.format('yes' if exp['analysis_ready']['plot_scale_prob_history'] else 'no'),
            '  - `metric_comparison`: {}'.format('yes' if exp['analysis_ready']['metric_comparison'] else 'no'),
            '  - `latency_comparison`: {}'.format('yes' if exp['analysis_ready']['latency_comparison'] else 'no'),
            '  - `hard_subset_analysis`: {}'.format('yes' if exp['analysis_ready']['hard_subset_analysis'] else 'no'),
        ])

    lines.extend([
        '',
        '## Typical commands',
        '',
        'Adjust paths to where you unpacked this bundle.',
        '',
        '```bash',
        'python tools/plot_scale_prob_history.py \\',
        '  --source <bundle>/experiments/<run>/work_dir \\',
        '  --config <bundle>/experiments/<run>/config/<config.py> \\',
        '  --out-dir <bundle>/replots/<run>_scale_probs',
        '```',
        '',
        '```bash',
        'python tools/analyze_sr_face_size_distribution.py \\',
        '  --ann-file <bundle>/dataset_metadata/train/labelv2.txt \\',
        '  --baseline-config <bundle>/experiments/baseline/config/<baseline.py> \\',
        '  --improved-config <bundle>/experiments/improved/config/<improved.py> \\',
        '  --improved-state <bundle>/experiments/improved/work_dir \\',
        '  --out-dir <bundle>/replots/face_size_analysis',
        '```',
        '',
        '```bash',
        'python tools/compare_widerface_results.py \\',
        '  --baseline <bundle>/experiments/baseline/results \\',
        '  --improved <bundle>/experiments/improved/results \\',
        '  --out-dir <bundle>/replots/comparison',
        '```',
        '',
        '```bash',
        'python tools/analyze_hard_subset_comparison.py \\',
        '  --baseline <bundle>/experiments/baseline/results \\',
        '  --improved <bundle>/experiments/improved/results \\',
        '  --gt-dir <bundle>/dataset_metadata/val/gt \\',
        '  --out-dir <bundle>/replots/hard_subset',
        '```',
        '',
    ])

    with open(osp.join(out_dir, 'README.md'), 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(lines) + '\n')


def write_manifest(path, payload):
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)


def build_zip(src_root, zip_path):
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(src_root):
            for file_name in files:
                src_path = osp.join(root, file_name)
                if osp.abspath(src_path) == osp.abspath(zip_path):
                    continue
                arcname = osp.relpath(src_path, src_root)
                archive.write(src_path, arcname)


def console_safe(text):
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    return str(text).encode(encoding, errors='backslashreplace').decode(encoding)


def main():
    args = parse_args()
    if args.skip_predictions:
        args.include_predictions = False
    if args.skip_dataset_metadata:
        args.include_dataset_metadata = False
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    experiments = parse_experiments(args.experiment)

    out_dir = resolve_path(args.out_dir, repo_root)
    ensure_dir(out_dir)

    manifest = {
        'repo_root': repo_root,
        'out_dir': out_dir,
        'copy_checkpoints': args.copy_checkpoints,
        'include_predictions': bool(args.include_predictions),
        'include_dataset_metadata': bool(args.include_dataset_metadata),
        'experiments': [],
    }

    for experiment in experiments:
        manifest['experiments'].append(package_experiment(experiment, args, repo_root, out_dir))

    manifest['dataset_metadata'] = package_dataset_metadata(args, repo_root, out_dir)
    write_readme(out_dir, manifest)
    write_manifest(osp.join(out_dir, 'manifest.json'), manifest)

    zip_path = osp.join(out_dir, args.zip_name)
    build_zip(out_dir, zip_path)
    print(console_safe('Packaged analysis artifacts into {}'.format(out_dir)))
    print(console_safe('Bundle zip: {}'.format(zip_path)))


if __name__ == '__main__':
    main()
