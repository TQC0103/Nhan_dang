#!/usr/bin/env python
"""Run SCRFD config generation in parallel worker processes.

Workers generate configs into isolated directories, then this script merges the
unique results back into the target group using the target group's naming
scheme. This keeps the downstream train/test flow unchanged.
"""

import argparse
import glob
import hashlib
import math
import os
import os.path as osp
import shutil
import subprocess
import sys
import time


def get_args():
    parser = argparse.ArgumentParser(
        description='Parallel wrapper for SCRFD config generation')
    parser.add_argument(
        '--group',
        type=str,
        required=True,
        help='final output config directory')
    parser.add_argument(
        '--template-config',
        type=str,
        required=True,
        help='template config path passed to each worker')
    parser.add_argument(
        '--num-configs',
        type=int,
        required=True,
        help='target total number of configs in the final group')
    parser.add_argument(
        '--workers',
        type=int,
        default=max(2, min(8, os.cpu_count() or 8)),
        help='number of parallel worker processes')
    parser.add_argument(
        '--oversample-factor',
        type=float,
        default=2.0,
        help='per-run oversampling factor before dedup/merge')
    parser.add_argument(
        '--mode',
        type=int,
        default=1,
        help='1: search backbone, 2: search full detector')
    parser.add_argument(
        '--gflops',
        type=float,
        default=None,
        help='target FLOPs budget')
    parser.add_argument(
        '--kernel-search',
        action='store_true',
        default=False,
        help='enable MobileNet kernel search')
    parser.add_argument(
        '--kernel-only',
        action='store_true',
        default=False,
        help='only search kernels around the template design')
    parser.add_argument(
        '--eps',
        type=float,
        default=2e-2,
        help='exact FLOPs tolerance')
    parser.add_argument(
        '--prefilter-eps',
        type=float,
        default=None,
        help='fast prefilter tolerance')
    parser.add_argument(
        '--disable-fast-prefilter',
        action='store_true',
        default=False,
        help='disable the lightweight FLOPs prefilter')
    parser.add_argument(
        '--report-every',
        type=int,
        default=50,
        help='progress log frequency forwarded to each worker')
    parser.add_argument(
        '--base-seed',
        type=int,
        default=3407,
        help='base random seed; worker i uses base_seed + i')
    parser.add_argument(
        '--keep-workdir',
        action='store_true',
        default=False,
        help='keep temporary worker directories after merging')
    return parser.parse_args()


def _collect_group_files(group):
    group = osp.normpath(group)
    group_name = osp.basename(group)
    pattern = osp.join(group, f'{group_name}_*.py')
    return sorted(glob.glob(pattern))


def _next_group_index(group):
    group = osp.normpath(group)
    group_name = osp.basename(group)
    index = 0
    while osp.exists(osp.join(group, f'{group_name}_{index}.py')):
        index += 1
    return index


def _read_text(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()


def _fingerprint(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def _build_worker_command(args, worker_group, worker_num_configs, worker_seed):
    generator_script = osp.join(
        osp.dirname(osp.abspath(__file__)),
        'generate_configs_2.5g_kernel_search.py')
    command = [
        sys.executable,
        generator_script,
        '--group',
        worker_group,
        '--template-config',
        args.template_config,
        '--num-configs',
        str(worker_num_configs),
        '--mode',
        str(args.mode),
        '--eps',
        str(args.eps),
        '--report-every',
        str(args.report_every),
        '--seed',
        str(worker_seed),
    ]
    if args.gflops is not None:
        command.extend(['--gflops', str(args.gflops)])
    if args.kernel_search:
        command.append('--kernel-search')
    if args.kernel_only:
        command.append('--kernel-only')
    if args.prefilter_eps is not None:
        command.extend(['--prefilter-eps', str(args.prefilter_eps)])
    if args.disable_fast_prefilter:
        command.append('--disable-fast-prefilter')
    return command


def main():
    args = get_args()
    script_dir = osp.dirname(osp.abspath(__file__))
    scrfd_root = osp.dirname(script_dir)

    final_group = osp.abspath(osp.normpath(args.group))
    final_group_name = osp.basename(final_group)
    if not final_group_name:
        raise ValueError('Could not infer final group name from --group')
    args.template_config = osp.abspath(osp.normpath(args.template_config))

    os.makedirs(final_group, exist_ok=True)
    existing_files = _collect_group_files(final_group)
    existing_texts = [_read_text(path) for path in existing_files]
    seen_fingerprints = {_fingerprint(text) for text in existing_texts}

    if len(existing_files) >= args.num_configs:
        print(
            f'Final group already has {len(existing_files)} configs, '
            f'which meets/exceeds the requested {args.num_configs}.')
        return

    remaining = args.num_configs - len(existing_files)
    worker_num_configs = int(math.ceil(
        remaining * max(args.oversample_factor, 1.0) / max(args.workers, 1)))
    if worker_num_configs <= 0:
        worker_num_configs = 1

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    work_root = osp.join(
        osp.dirname(final_group),
        f'.parallel_{final_group_name}_{timestamp}')
    workers_root = osp.join(work_root, 'workers')
    logs_root = osp.join(work_root, 'logs')
    os.makedirs(workers_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    print(f'Final group: {final_group}')
    print(f'Existing configs: {len(existing_files)}')
    print(f'Remaining target: {remaining}')
    print(f'Workers: {args.workers}')
    print(f'Worker target configs: {worker_num_configs}')
    print(f'Temporary work root: {work_root}')

    processes = []
    log_files = []
    for worker_idx in range(args.workers):
        worker_group = osp.join(workers_root, f'worker_{worker_idx:02d}')
        os.makedirs(worker_group, exist_ok=True)
        worker_seed = args.base_seed + worker_idx
        command = _build_worker_command(
            args, worker_group, worker_num_configs, worker_seed)
        log_path = osp.join(logs_root, f'worker_{worker_idx:02d}.log')
        log_handle = open(log_path, 'w', encoding='utf-8')
        process = subprocess.Popen(
            command,
            cwd=scrfd_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT)
        processes.append((worker_idx, process, log_path))
        log_files.append(log_handle)
        print(
            f'Launched worker {worker_idx}: seed={worker_seed}, '
            f'log={log_path}')

    failed_workers = []
    for worker_idx, process, log_path in processes:
        return_code = process.wait()
        if return_code != 0:
            failed_workers.append((worker_idx, return_code, log_path))

    for log_handle in log_files:
        log_handle.close()

    if failed_workers:
        for worker_idx, return_code, log_path in failed_workers:
            print(
                f'Worker {worker_idx} failed with code {return_code}. '
                f'Inspect log: {log_path}',
                file=sys.stderr)
        raise SystemExit(1)

    next_index = _next_group_index(final_group)
    added = 0
    worker_files = []
    for worker_idx in range(args.workers):
        worker_group = osp.join(workers_root, f'worker_{worker_idx:02d}')
        worker_files.extend(sorted(glob.glob(osp.join(worker_group, '*.py'))))

    for path in worker_files:
        text = _read_text(path)
        fingerprint = _fingerprint(text)
        if fingerprint in seen_fingerprints:
            continue
        output_path = osp.join(final_group, f'{final_group_name}_{next_index}.py')
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        seen_fingerprints.add(fingerprint)
        next_index += 1
        added += 1
        if len(existing_files) + added >= args.num_configs:
            break

    final_count = len(_collect_group_files(final_group))
    print(f'Added configs: {added}')
    print(f'Final config count: {final_count}')

    if final_count < args.num_configs:
        print(
            f'Only reached {final_count}/{args.num_configs} configs after merge. '
            'Rerun with a larger --oversample-factor or more workers.',
            file=sys.stderr)
        raise SystemExit(2)

    if not args.keep_workdir:
        shutil.rmtree(work_root, ignore_errors=True)
        print('Removed temporary worker directories.')
    else:
        print(f'Kept worker directories: {work_root}')


if __name__ == '__main__':
    main()
