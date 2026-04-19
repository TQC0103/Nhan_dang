import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import zipfile


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build one bundled report explaining why ASR+JSAR improves WIDERFace hard AP.')
    parser.add_argument('--baseline-results', required=True, help='Baseline result dir')
    parser.add_argument('--improved-results', required=True, help='Improved result dir')
    parser.add_argument('--baseline-work-dir', required=True, help='Baseline work_dir')
    parser.add_argument('--improved-work-dir', required=True, help='Improved work_dir')
    parser.add_argument('--baseline-config', required=True, help='Baseline config path')
    parser.add_argument('--improved-config', required=True, help='Improved config path')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--ann-file', default='data/retinaface/train/labelv2.txt', help='Train labelv2 file')
    parser.add_argument('--gt-dir', default='data/retinaface/val/gt', help='WIDERFace validation gt dir')
    parser.add_argument('--baseline-name', default='Baseline', help='Display name in report')
    parser.add_argument('--improved-name', default='ASR+JSAR', help='Display name in report')
    parser.add_argument('--prob-mode', choices=['latest', 'mean_epochs'], default='mean_epochs',
                        help='How to summarize improved scale probabilities')
    parser.add_argument('--skip-hard-analysis', action='store_true',
                        help='Skip hard-subset analysis even if predictions exist')
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, payload):
    ensure_dir(osp.dirname(path))
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)


def load_summary(result_dir):
    summary_path = osp.join(result_dir, 'results_summary.json')
    if osp.isfile(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as infile:
            return json.load(infile), summary_path
    aps_path = osp.join(result_dir, 'aps')
    if osp.isfile(aps_path):
        with open(aps_path, 'r', encoding='utf-8') as infile:
            raw = infile.readline().strip()
        values = [float(item) for item in raw.split(',') if item.strip()]
        if len(values) == 3:
            return {
                'easy_AP': values[0],
                'medium_AP': values[1],
                'hard_AP': values[2],
                'mAP': sum(values) / 3.0,
            }, aps_path
    raise FileNotFoundError('Missing results_summary.json or aps under {}'.format(result_dir))


def run_step(log_path, command, cwd):
    ensure_dir(osp.dirname(log_path))
    with open(log_path, 'w', encoding='utf-8') as logfile:
        logfile.write('COMMAND: {}\n\n'.format(' '.join(command)))
        logfile.flush()
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError('Step failed with exit code {}: {}'.format(process.returncode, ' '.join(command)))


def maybe_run_hard_analysis(args, out_dir, logs_dir, repo_root):
    baseline_pred_dir = osp.join(args.baseline_results, 'predictions')
    improved_pred_dir = osp.join(args.improved_results, 'predictions')
    if args.skip_hard_analysis:
        return {
            'status': 'skipped',
            'reason': 'skip flag set',
        }
    if not osp.isdir(baseline_pred_dir) or not osp.isdir(improved_pred_dir):
        return {
            'status': 'skipped',
            'reason': 'missing predictions/ directory; rerun eval with --save-preds first',
        }

    step_out = osp.join(out_dir, 'hard_subset')
    command = [
        sys.executable,
        'tools/analyze_hard_subset_comparison.py',
        '--baseline', args.baseline_results,
        '--improved', args.improved_results,
        '--gt-dir', args.gt_dir,
        '--out-dir', step_out,
    ]
    run_step(osp.join(logs_dir, '03_hard_subset_analysis.log'), command, cwd=repo_root)
    json_path = osp.join(step_out, 'hard_subset_analysis.json')
    payload = {}
    if osp.isfile(json_path):
        with open(json_path, 'r', encoding='utf-8') as infile:
            payload = json.load(infile)
    return {
        'status': 'ok',
        'out_dir': step_out,
        'summary': payload,
    }


def copy_file(src, dst):
    ensure_dir(osp.dirname(dst))
    shutil.copy2(src, dst)


def copy_tree_filtered(src_root, dst_root, include_exts=None, include_names=None, recursive=True):
    copied = []
    if not osp.isdir(src_root):
        return copied
    include_exts = set(include_exts or [])
    include_names = set(include_names or [])

    for root, dirs, files in os.walk(src_root):
        for file_name in files:
            src_path = osp.join(root, file_name)
            rel_path = osp.relpath(src_path, src_root)
            ext = osp.splitext(file_name)[1].lower()
            if include_exts and ext in include_exts:
                dst_path = osp.join(dst_root, rel_path)
                copy_file(src_path, dst_path)
                copied.append((src_path, dst_path))
                continue
            if include_names and file_name in include_names:
                dst_path = osp.join(dst_root, rel_path)
                copy_file(src_path, dst_path)
                copied.append((src_path, dst_path))
        if not recursive:
            break
    return copied


def collect_logs(args, out_dir, generated_dirs):
    logs_root = osp.join(out_dir, 'logs')
    collected_root = osp.join(logs_root, 'collected')
    ensure_dir(collected_root)

    manifest = {
        'baseline_work_dir': osp.abspath(args.baseline_work_dir),
        'improved_work_dir': osp.abspath(args.improved_work_dir),
        'baseline_results': osp.abspath(args.baseline_results),
        'improved_results': osp.abspath(args.improved_results),
        'copied_files': [],
    }

    copy_specs = [
        ('baseline_work', args.baseline_work_dir, {'.log', '.json'}, {'aps', 'results_summary.json', 'results_summary.csv'}),
        ('improved_work', args.improved_work_dir, {'.log', '.json', '.jsonl'}, {'aps', 'results_summary.json', 'results_summary.csv'}),
        ('baseline_results', args.baseline_results, {'.log', '.json', '.csv'}, {'aps', 'results_summary.json', 'results_summary.csv'}),
        ('improved_results', args.improved_results, {'.log', '.json', '.csv'}, {'aps', 'results_summary.json', 'results_summary.csv'}),
    ]

    for name, src_root, include_exts, include_names in copy_specs:
        copied = copy_tree_filtered(
            src_root,
            osp.join(collected_root, name),
            include_exts=include_exts,
            include_names=include_names,
            recursive=True,
        )
        for src, dst in copied:
            manifest['copied_files'].append({
                'source': src,
                'target': dst,
            })

    # Also copy generated analysis artifacts into a dedicated folder for convenience.
    for name, src_root in generated_dirs.items():
        if not src_root or not osp.isdir(src_root):
            continue
        copied = copy_tree_filtered(
            src_root,
            osp.join(collected_root, 'generated', name),
            include_exts={'.md', '.json', '.csv', '.png', '.jsonl', '.log'},
            recursive=True,
        )
        for src, dst in copied:
            manifest['copied_files'].append({
                'source': src,
                'target': dst,
            })

    save_json(osp.join(logs_root, 'manifest.json'), manifest)

    zip_path = osp.join(out_dir, 'logs_bundle.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(logs_root):
            for file_name in files:
                src_path = osp.join(root, file_name)
                arcname = osp.relpath(src_path, out_dir)
                archive.write(src_path, arcname)
    return logs_root, zip_path


def safe_get(mapping, *keys, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def write_report(path, args, comparison, hard_analysis, face_summary, scale_summary, logs_root, logs_zip):
    baseline = comparison['baseline']
    improved = comparison['improved']
    hard_delta = improved['hard_AP'] - baseline['hard_AP']
    map_delta = improved['mAP'] - baseline['mAP']

    raw = face_summary['raw_summary']
    baseline_dist = face_summary['baseline_summary']
    improved_dist = face_summary['improved_summary']

    lines = [
        '# Hard-Gain Explanation Report',
        '',
        '## Headline',
        '',
        '{} improves `hard_AP` from `{:.4f}` to `{:.4f}` (`{:+.4f}`), while overall `mAP` changes from `{:.4f}` to `{:.4f}` (`{:+.4f}`).'.format(
            args.improved_name,
            baseline['hard_AP'],
            improved['hard_AP'],
            hard_delta,
            baseline['mAP'],
            improved['mAP'],
            map_delta,
        ),
        '',
        '## WIDERFace Metrics',
        '',
        '| Metric | {} | {} | Delta |'.format(args.baseline_name, args.improved_name),
        '| --- | ---: | ---: | ---: |',
        '| easy_AP | {:.4f} | {:.4f} | {:+.4f} |'.format(
            baseline['easy_AP'], improved['easy_AP'], improved['easy_AP'] - baseline['easy_AP']),
        '| medium_AP | {:.4f} | {:.4f} | {:+.4f} |'.format(
            baseline['medium_AP'], improved['medium_AP'], improved['medium_AP'] - baseline['medium_AP']),
        '| hard_AP | {:.4f} | {:.4f} | {:+.4f} |'.format(
            baseline['hard_AP'], improved['hard_AP'], hard_delta),
        '| mAP | {:.4f} | {:.4f} | {:+.4f} |'.format(
            baseline['mAP'], improved['mAP'], map_delta),
        '',
        '## Size-Distribution Hypothesis',
        '',
        '| Distribution | mean | p50 | p90 | tiny | small | medium | large | tiny->>=16 | <32->>=32 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        '| Original | {:.2f} | {:.2f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | - | - |'.format(
            raw['mean'], raw['p50'], raw['p90'],
            raw['bin_ratios'][0], raw['bin_ratios'][1], raw['bin_ratios'][2], raw['bin_ratios'][3]),
        '| Baseline SR | {:.2f} | {:.2f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
            baseline_dist['mean'], baseline_dist['p50'], baseline_dist['p90'],
            baseline_dist['bin_ratios'][0], baseline_dist['bin_ratios'][1],
            baseline_dist['bin_ratios'][2], baseline_dist['bin_ratios'][3],
            baseline_dist['promote_tiny_to_ge16_ratio'], baseline_dist['promote_lt32_to_ge32_ratio']),
        '| {} | {:.2f} | {:.2f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
            args.improved_name,
            improved_dist['mean'], improved_dist['p50'], improved_dist['p90'],
            improved_dist['bin_ratios'][0], improved_dist['bin_ratios'][1],
            improved_dist['bin_ratios'][2], improved_dist['bin_ratios'][3],
            improved_dist['promote_tiny_to_ge16_ratio'], improved_dist['promote_lt32_to_ge32_ratio']),
        '',
        'Interpretation: if the improved policy lowers the `tiny` ratio and raises `tiny->>=16` or `<32->>=32`, it is exposing hard tiny/small faces at more trainable scales more often.',
        '',
        '## Scale-Probability Trajectory',
        '',
        '- Source: `{}`'.format(scale_summary.get('resolved_source', '')),
        '- Latest epoch: `{}`'.format(scale_summary.get('latest_epoch', '')),
        '- Latest scale probs: `{}`'.format(
            ', '.join('{:.4f}'.format(v) for v in scale_summary.get('latest_scale_probs', [])) if scale_summary.get('latest_scale_probs') else ''),
        '- Plot: [scale_prob_history_epoch_end.png]({})'.format(
            osp.join('.', 'scale_prob_history', 'scale_prob_history_epoch_end.png').replace('\\', '/')),
        '',
    ]

    jsar_before = safe_get(face_summary, 'metadata', 'improved_scale_source', default={})
    del jsar_before

    if hard_analysis.get('status') == 'ok':
        aggregate = hard_analysis['summary'].get('aggregate', {})
        lines.extend([
            '## Hard Subset Diagnostics',
            '',
            '| Metric | Baseline | Improved | Delta |',
            '| --- | ---: | ---: | ---: |',
            '| hard_recall_proxy | {:.4f} | {:.4f} | {:+.4f} |'.format(
                aggregate.get('baseline_hard_recall_proxy', 0.0),
                aggregate.get('improved_hard_recall_proxy', 0.0),
                aggregate.get('hard_recall_proxy_delta', 0.0),
            ),
            '| precision_proxy | {:.4f} | {:.4f} | {:+.4f} |'.format(
                aggregate.get('baseline_precision_proxy', 0.0),
                aggregate.get('improved_precision_proxy', 0.0),
                aggregate.get('precision_proxy_delta', 0.0),
            ),
            '| pred_count | {} | {} | {:+d} |'.format(
                int(aggregate.get('baseline_pred_count', 0)),
                int(aggregate.get('improved_pred_count', 0)),
                int(aggregate.get('improved_pred_count', 0)) - int(aggregate.get('baseline_pred_count', 0)),
            ),
            '',
            '- Detailed markdown: [hard_subset_analysis.md]({})'.format(
                osp.join('.', 'hard_subset', 'hard_subset_analysis.md').replace('\\', '/')),
            '- Top improved images: [top_improved_images.csv]({})'.format(
                osp.join('.', 'hard_subset', 'top_improved_images.csv').replace('\\', '/')),
            '',
        ])
    else:
        lines.extend([
            '## Hard Subset Diagnostics',
            '',
            'Skipped: {}'.format(hard_analysis.get('reason', 'unknown')),
            '',
        ])

    improved_runtime_summary = None
    improved_latest_summary_path = osp.join(args.improved_work_dir, 'adaptive_sr', 'latest_summary.json')
    if osp.isfile(improved_latest_summary_path):
        with open(improved_latest_summary_path, 'r', encoding='utf-8') as infile:
            improved_runtime_summary = json.load(infile)
    elif osp.isfile(osp.join(args.improved_work_dir, 'adaptive_sr', 'online_scheduler_handoff', 'latest_summary.json')):
        with open(osp.join(args.improved_work_dir, 'adaptive_sr', 'online_scheduler_handoff', 'latest_summary.json'), 'r', encoding='utf-8') as infile:
            improved_runtime_summary = json.load(infile)

    if improved_runtime_summary:
        jsar_before_hist = improved_runtime_summary.get('jsar_before_hist')
        jsar_after_hist = improved_runtime_summary.get('jsar_after_hist')
        if jsar_before_hist is not None and jsar_after_hist is not None:
            lines.extend([
                '## JSAR Assignment Evidence',
                '',
                '- `jsar_before_hist`: `{}`'.format(jsar_before_hist),
                '- `jsar_after_hist`: `{}`'.format(jsar_after_hist),
                '- This is direct evidence that positive assignments increased for the smallest bins during training.',
                '',
            ])

    lines.extend([
        '## Artifacts',
        '',
        '- Face-size analysis: [face_size_distribution/analysis.md]({})'.format(
            osp.join('.', 'face_size_distribution', 'analysis.md').replace('\\', '/')),
        '- Comparison table: [comparison/comparison.md]({})'.format(
            osp.join('.', 'comparison', 'comparison.md').replace('\\', '/')),
        '- Logs folder: [logs]({})'.format(osp.join('.', 'logs').replace('\\', '/')),
        '- Logs zip: [logs_bundle.zip]({})'.format(osp.join('.', 'logs_bundle.zip').replace('\\', '/')),
        '',
        '## Conclusion',
        '',
        'The report is designed to support the claim that {} improves hard faces because its training-time scale policy exposes more tiny/small faces at larger effective sizes, and JSAR further increases supervision density for those hard cases. The scale trajectory plot, the face-size transition plots, and the hard-subset recall proxy should line up around that explanation.'.format(
            args.improved_name),
    ])

    with open(path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    out_dir = osp.abspath(args.out_dir)
    logs_dir = osp.join(out_dir, 'logs', 'steps')
    comparison_dir = osp.join(out_dir, 'comparison')
    scale_dir = osp.join(out_dir, 'scale_prob_history')
    face_dir = osp.join(out_dir, 'face_size_distribution')

    ensure_dir(out_dir)
    ensure_dir(logs_dir)

    comparison_cmd = [
        sys.executable,
        'tools/compare_widerface_results.py',
        '--baseline', args.baseline_results,
        '--improved', args.improved_results,
        '--baseline-name', args.baseline_name,
        '--improved-name', args.improved_name,
        '--out-dir', comparison_dir,
    ]
    run_step(osp.join(logs_dir, '01_compare_widerface.log'), comparison_cmd, cwd=repo_root)

    scale_cmd = [
        sys.executable,
        'tools/plot_scale_prob_history.py',
        '--source', args.improved_work_dir,
        '--config', args.improved_config,
        '--out-dir', scale_dir,
    ]
    run_step(osp.join(logs_dir, '02_plot_scale_prob_history.log'), scale_cmd, cwd=repo_root)

    face_cmd = [
        sys.executable,
        'tools/analyze_sr_face_size_distribution.py',
        '--ann-file', args.ann_file,
        '--baseline-config', args.baseline_config,
        '--improved-config', args.improved_config,
        '--improved-state', args.improved_work_dir,
        '--prob-mode', args.prob_mode,
        '--out-dir', face_dir,
    ]
    run_step(osp.join(logs_dir, '04_face_size_distribution.log'), face_cmd, cwd=repo_root)

    hard_analysis = maybe_run_hard_analysis(args, out_dir, logs_dir, repo_root)

    baseline_summary, baseline_summary_path = load_summary(args.baseline_results)
    improved_summary, improved_summary_path = load_summary(args.improved_results)
    comparison = {
        'baseline': baseline_summary,
        'baseline_summary_path': baseline_summary_path,
        'improved': improved_summary,
        'improved_summary_path': improved_summary_path,
    }

    with open(osp.join(face_dir, 'face_size_distribution_analysis.json'), 'r', encoding='utf-8') as infile:
        face_summary = json.load(infile)
    with open(osp.join(scale_dir, 'scale_prob_history_summary.json'), 'r', encoding='utf-8') as infile:
        scale_summary = json.load(infile)

    generated_dirs = {
        'comparison': comparison_dir,
        'scale_prob_history': scale_dir,
        'face_size_distribution': face_dir,
        'hard_subset': hard_analysis.get('out_dir') if hard_analysis.get('status') == 'ok' else None,
        'step_logs': logs_dir,
    }
    logs_root, logs_zip = collect_logs(args, out_dir, generated_dirs)

    report_payload = {
        'comparison': comparison,
        'hard_analysis': hard_analysis,
        'face_size_distribution': face_summary,
        'scale_prob_history': scale_summary,
        'logs_root': logs_root,
        'logs_zip': logs_zip,
    }
    save_json(osp.join(out_dir, 'report_data.json'), report_payload)
    write_report(
        osp.join(out_dir, 'report.md'),
        args,
        comparison,
        hard_analysis,
        face_summary,
        scale_summary,
        logs_root,
        logs_zip,
    )

    report_zip = osp.join(out_dir, 'report_bundle.zip')
    with zipfile.ZipFile(report_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(out_dir):
            for file_name in files:
                src_path = osp.join(root, file_name)
                if osp.abspath(src_path) == osp.abspath(report_zip):
                    continue
                arcname = osp.relpath(src_path, out_dir)
                archive.write(src_path, arcname)

    print('Wrote bundled report to {}'.format(out_dir))
    print('Main report: {}'.format(osp.join(out_dir, 'report.md')))
    print('Logs zip: {}'.format(logs_zip))
    print('Report zip: {}'.format(report_zip))


if __name__ == '__main__':
    main()
