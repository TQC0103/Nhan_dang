import argparse
import json
import os
import os.path as osp
import re
import shutil
import zipfile


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_path(path, repo_root):
    if not path:
        return None
    if osp.isabs(path):
        return path
    return osp.join(repo_root, path)


def resolve_group(group, repo_root):
    normalized = group.replace('\\', '/').rstrip('/')
    if normalized.startswith('configs/'):
        config_dir = resolve_path(normalized, repo_root)
        group_name = osp.basename(normalized)
    else:
        config_dir = resolve_path(osp.join('configs', normalized), repo_root)
        group_name = osp.basename(normalized)
    return config_dir, group_name


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


def list_candidate_configs(config_dir, prefix):
    pattern = re.compile(r'^{}_(\d+)\.py$'.format(re.escape(prefix)))
    candidates = []
    if not osp.isdir(config_dir):
        return candidates
    for file_name in os.listdir(config_dir):
        match = pattern.match(file_name)
        if not match:
            continue
        idx = int(match.group(1))
        candidates.append((idx, osp.join(config_dir, file_name)))
    return sorted(candidates, key=lambda item: item[0])


def collect_work_files(work_dir, checkpoint_mode):
    selected = []
    if not osp.isdir(work_dir):
        return selected
    for root, _, files in os.walk(work_dir):
        for file_name in files:
            src_path = osp.join(root, file_name)
            rel_path = osp.relpath(src_path, work_dir)
            if file_name.endswith('.pth'):
                if checkpoint_mode == 'none':
                    continue
                if checkpoint_mode == 'latest' and file_name != 'latest.pth':
                    continue
                selected.append((src_path, rel_path))
                continue
            if file_name.endswith(('.log', '.log.json', '.json', '.jsonl', '.md', '.txt', '.csv', '.png')):
                selected.append((src_path, rel_path))
    return selected


def load_best_candidate(viz_dir):
    stats_path = osp.join(viz_dir, 'search_stats.jsonl')
    if not osp.isfile(stats_path):
        return None
    with open(stats_path, 'r', encoding='utf-8') as file_obj:
        first_line = file_obj.readline().strip()
    if not first_line:
        return None
    try:
        return json.loads(first_line)
    except json.JSONDecodeError:
        return None


def build_readme(bundle_dir, manifest):
    lines = [
        '# SCRFD Search Artifacts Bundle',
        '',
        '## Overview',
        '',
        '- Search group: `{}`'.format(manifest['group_name']),
        '- Number of candidates: `{}`'.format(manifest['num_candidates']),
        '- Result root: `{}`'.format(manifest['result_root']),
        '',
    ]
    if manifest.get('best_candidate'):
        best = manifest['best_candidate']
        lines.extend([
            '## Best Candidate',
            '',
            '- Name: `{}`'.format(best.get('name')),
            '- Easy AP: `{:.4f}`'.format(best.get('aps', [0, 0, 0])[0]),
            '- Medium AP: `{:.4f}`'.format(best.get('aps', [0, 0, 0])[1]),
            '- Hard AP: `{:.4f}`'.format(best.get('aps', [0, 0, 0])[2]),
            '- GFLOPs: `{:.4f}`'.format(best.get('all_flops', 0.0)),
            '',
        ])

    lines.extend([
        '## Bundle Structure',
        '',
        '- `configs/{group}`: generated candidate configs'.format(group=manifest['group_name']),
        '- `candidates/<candidate>/work_dir`: training logs, metadata, checkpoints',
        '- `candidates/<candidate>/results`: WIDERFace eval outputs including `aps`',
        '- `visualization`: plots from `search_tools/visualize_search.py`',
        '- `logs`: launcher logs for generate/train/test/visualize',
        '- `run_metadata`: run settings and summary metadata',
        '',
        '## Main Plots',
        '',
        '- `visualization/ap_vs_candidate.png`: AP curves across candidate index',
        '- `visualization/ap_vs_flops.png`: ranking metric vs FLOPs',
        '- `visualization/topk_compute_distribution.png`: compute breakdown of top-k models',
        '- `visualization/topk_kernel_heatmap.png`: kernel choices of top-k models',
        '- `visualization/topk_summary.md`: top-k ranking table',
        '',
        '## Candidate Summary',
        '',
        '| Candidate | Config | Checkpoint | APS |',
        '| --- | --- | --- | --- |',
    ])
    for item in manifest['candidates']:
        lines.append(
            '| `{}` | {} | {} | {} |'.format(
                item['name'],
                'yes' if item['has_config'] else 'no',
                'yes' if item['has_checkpoint'] else 'no',
                'yes' if item['has_aps'] else 'no'))

    readme_path = osp.join(bundle_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write('\n'.join(lines) + '\n')
    return readme_path


def write_manifest(bundle_dir, manifest):
    manifest_path = osp.join(bundle_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as file_obj:
        json.dump(manifest, file_obj, indent=2, ensure_ascii=False)
    return manifest_path


def write_zip(bundle_dir, zip_name):
    zip_path = osp.join(bundle_dir, zip_name)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(bundle_dir):
            for file_name in files:
                src_path = osp.join(root, file_name)
                if osp.abspath(src_path) == osp.abspath(zip_path):
                    continue
                archive.write(src_path, osp.relpath(src_path, bundle_dir))
    return zip_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Package SCRFD search configs/logs/results/plots into a downloadable bundle.')
    parser.add_argument('--group', required=True, help='Search group name or configs/<group> path.')
    parser.add_argument('--out-dir', required=True, help='Output directory for the packaged bundle.')
    parser.add_argument('--result-root', default='wouts', help='Root directory used by search_test_parallel.sh.')
    parser.add_argument('--work-root', default='work_dirs', help='Root work_dirs path.')
    parser.add_argument('--viz-dir', required=True, help='Visualization directory produced by visualize_search.py.')
    parser.add_argument('--log-dir', required=True, help='Directory that contains launcher logs.')
    parser.add_argument('--run-metadata', default=None, help='Optional JSON metadata file to copy into the bundle.')
    parser.add_argument(
        '--copy-checkpoints',
        choices=['latest', 'all', 'none'],
        default='latest',
        help='Which candidate checkpoints to copy.')
    parser.add_argument('--zip-name', default='analysis_artifacts_bundle.zip', help='Zip filename written under --out-dir.')
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    config_dir, group_name = resolve_group(args.group, repo_root)
    prefix = group_name
    result_root = resolve_path(args.result_root, repo_root)
    work_root = resolve_path(args.work_root, repo_root)
    viz_dir = resolve_path(args.viz_dir, repo_root)
    log_dir = resolve_path(args.log_dir, repo_root)
    out_dir = resolve_path(args.out_dir, repo_root)
    metadata_path = resolve_path(args.run_metadata, repo_root) if args.run_metadata else None

    ensure_dir(out_dir)

    candidates = list_candidate_configs(config_dir, prefix)
    copied = []
    candidate_manifest = []

    config_dst_root = osp.join(out_dir, 'configs', group_name)
    for _, config_path in candidates:
        dst = osp.join(config_dst_root, osp.basename(config_path))
        copy_file(config_path, dst)
        copied.append({'source': config_path, 'target': dst})

    for idx, config_path in candidates:
        name = '{}_{}'.format(prefix, idx)
        work_dir = osp.join(work_root, name)
        result_dir = osp.join(result_root, group_name, name)
        candidate_root = osp.join(out_dir, 'candidates', name)

        has_checkpoint = False
        for src_path, rel_path in collect_work_files(work_dir, args.copy_checkpoints):
            dst_path = osp.join(candidate_root, 'work_dir', rel_path)
            copy_file(src_path, dst_path)
            copied.append({'source': src_path, 'target': dst_path})
            if src_path.endswith('.pth'):
                has_checkpoint = True

        has_aps = False
        if osp.isdir(result_dir):
            for src_path, dst_path in copy_tree(result_dir, osp.join(candidate_root, 'results')):
                copied.append({'source': src_path, 'target': dst_path})
                if osp.basename(src_path) == 'aps':
                    has_aps = True

        candidate_manifest.append({
            'idx': idx,
            'name': name,
            'config': config_path,
            'work_dir': work_dir,
            'result_dir': result_dir,
            'has_config': True,
            'has_checkpoint': has_checkpoint,
            'has_aps': has_aps,
        })

    if osp.isdir(viz_dir):
        for src_path, dst_path in copy_tree(viz_dir, osp.join(out_dir, 'visualization')):
            copied.append({'source': src_path, 'target': dst_path})

    if osp.isdir(log_dir):
        for src_path, dst_path in copy_tree(log_dir, osp.join(out_dir, 'logs')):
            copied.append({'source': src_path, 'target': dst_path})

    if metadata_path and osp.isfile(metadata_path):
        dst = osp.join(out_dir, 'run_metadata', osp.basename(metadata_path))
        copy_file(metadata_path, dst)
        copied.append({'source': metadata_path, 'target': dst})

    best_candidate = load_best_candidate(viz_dir)
    manifest = {
        'group_name': group_name,
        'config_dir': config_dir,
        'result_root': result_root,
        'work_root': work_root,
        'viz_dir': viz_dir,
        'log_dir': log_dir,
        'num_candidates': len(candidate_manifest),
        'best_candidate': best_candidate,
        'candidates': candidate_manifest,
        'copied': copied,
    }
    manifest_path = write_manifest(out_dir, manifest)
    readme_path = build_readme(out_dir, manifest)
    zip_path = write_zip(out_dir, args.zip_name)

    print('Packaged search artifacts into', out_dir)
    print('Manifest:', manifest_path)
    print('README:', readme_path)
    print('Bundle zip:', zip_path)


if __name__ == '__main__':
    main()
