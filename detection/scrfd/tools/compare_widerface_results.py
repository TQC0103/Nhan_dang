import argparse
import csv
import json
import os


METRICS = ('easy_AP', 'medium_AP', 'hard_AP', 'mAP')


def resolve_summary_path(path):
    if os.path.isdir(path):
        candidate = os.path.join(path, 'results_summary.json')
        if os.path.exists(candidate):
            return candidate
    return path


def load_summary(path):
    summary_path = resolve_summary_path(path)
    if not os.path.exists(summary_path):
        raise FileNotFoundError('Missing results summary: {}'.format(summary_path))
    with open(summary_path, 'r', encoding='utf-8') as infile:
        return json.load(infile), summary_path


def format_delta(delta):
    return '{:+.4f}'.format(delta)


def main():
    parser = argparse.ArgumentParser(
        description='Compare two WIDERFace evaluation summaries and write a compact report.')
    parser.add_argument('--baseline', required=True, help='Baseline results_summary.json or directory containing it')
    parser.add_argument('--improved', required=True, help='Improved results_summary.json or directory containing it')
    parser.add_argument('--baseline-name', default='Baseline', help='Display name for baseline model')
    parser.add_argument('--improved-name', default='Improved', help='Display name for improved model')
    parser.add_argument('--out-dir', required=True, help='Output directory for comparison artifacts')
    args = parser.parse_args()

    baseline, baseline_path = load_summary(args.baseline)
    improved, improved_path = load_summary(args.improved)

    os.makedirs(args.out_dir, exist_ok=True)

    comparison = {
        'baseline_name': args.baseline_name,
        'baseline_summary': baseline_path,
        'improved_name': args.improved_name,
        'improved_summary': improved_path,
        'metrics': {},
    }

    for metric in METRICS:
        base_value = float(baseline.get(metric, 0.0))
        improved_value = float(improved.get(metric, 0.0))
        comparison['metrics'][metric] = {
            'baseline': base_value,
            'improved': improved_value,
            'delta': improved_value - base_value,
        }

    json_path = os.path.join(args.out_dir, 'comparison.json')
    with open(json_path, 'w', encoding='utf-8') as outfile:
        json.dump(comparison, outfile, indent=2, sort_keys=True)

    csv_path = os.path.join(args.out_dir, 'comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['metric', args.baseline_name, args.improved_name, 'delta'])
        for metric in METRICS:
            values = comparison['metrics'][metric]
            writer.writerow([
                metric,
                '{:.4f}'.format(values['baseline']),
                '{:.4f}'.format(values['improved']),
                format_delta(values['delta']),
            ])

    markdown_lines = [
        '# WIDERFace Comparison',
        '',
        '| Metric | {} | {} | Delta |'.format(args.baseline_name, args.improved_name),
        '| --- | ---: | ---: | ---: |',
    ]
    for metric in METRICS:
        values = comparison['metrics'][metric]
        markdown_lines.append(
            '| {} | {:.4f} | {:.4f} | {} |'.format(
                metric,
                values['baseline'],
                values['improved'],
                format_delta(values['delta']),
            ))

    md_path = os.path.join(args.out_dir, 'comparison.md')
    with open(md_path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(markdown_lines) + '\n')

    print('Wrote comparison artifacts to {}'.format(args.out_dir))
    print('Baseline summary: {}'.format(baseline_path))
    print('Improved summary: {}'.format(improved_path))


if __name__ == '__main__':
    main()
