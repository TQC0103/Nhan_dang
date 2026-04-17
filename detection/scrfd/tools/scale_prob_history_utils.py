import glob
import json
import os
import os.path as osp
import re

import numpy as np


SCALE_PROBS_PATTERN = re.compile(
    r'epoch\s+(\d+).*?scale_probs=\[(.*?)\]',
    flags=re.IGNORECASE)


def normalize_probs(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total <= 0:
        return None
    values = values / total
    return values.tolist()


def resolve_history_source(path):
    path = osp.abspath(path)
    if osp.isfile(path):
        return path
    if not osp.isdir(path):
        raise FileNotFoundError('Could not find scale-probability source: {}'.format(path))

    candidates = [
        osp.join(path, 'scale_prob_history.jsonl'),
        osp.join(path, 'adaptive_sr', 'scale_prob_history.jsonl'),
    ]
    for candidate in candidates:
        if osp.isfile(candidate):
            return candidate

    epoch_dir_candidates = [
        osp.join(path, 'epoch_logs'),
        osp.join(path, 'adaptive_sr', 'epoch_logs'),
    ]
    for candidate in epoch_dir_candidates:
        if osp.isdir(candidate):
            return candidate

    log_candidates = sorted(glob.glob(osp.join(path, '*.log')))
    if log_candidates:
        return log_candidates[-1]

    logs_dir = osp.join(path, 'logs')
    log_candidates = sorted(glob.glob(osp.join(logs_dir, '*.log')))
    if log_candidates:
        return log_candidates[-1]

    raise FileNotFoundError(
        'Could not resolve scale-probability history under {}. Expected '
        'scale_prob_history.jsonl, epoch_logs/, or a text train log.'.format(path))


def _load_jsonl_records(path):
    records = []
    with open(path, 'r', encoding='utf-8') as infile:
        for raw_line in infile:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if 'scale_probs' not in payload:
                continue
            record = dict(payload)
            record['epoch'] = int(record.get('epoch', 0))
            record['iteration'] = int(record.get('iteration', 0))
            record['scale_probs'] = [float(v) for v in record.get('scale_probs', [])]
            if 'scale_candidates' in record:
                record['scale_candidates'] = [float(v) for v in record['scale_candidates']]
            records.append(record)
    return records


def _parse_scale_prob_list(raw_text):
    if not raw_text.strip():
        return []
    values = []
    for item in raw_text.split(','):
        item = item.strip().strip("'").strip('"')
        if not item:
            continue
        values.append(float(item))
    return values


def _load_text_log_records(path, fallback_candidates=None):
    records = []
    with open(path, 'r', encoding='utf-8', errors='replace') as infile:
        for line in infile:
            match = SCALE_PROBS_PATTERN.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            scale_probs = _parse_scale_prob_list(match.group(2))
            if not scale_probs:
                continue
            record = {
                'epoch': epoch,
                'iteration': 0,
                'scale_probs': scale_probs,
                'record_type': 'epoch_end',
                'source': path,
            }
            if fallback_candidates is not None:
                record['scale_candidates'] = [float(v) for v in fallback_candidates]
            if 'Adaptive SR epoch' in line:
                record['scheduler'] = 'adaptive_sr'
            elif 'Online Scheduler Handoff epoch' in line:
                record['scheduler'] = 'online_scheduler_handoff'
            else:
                record['scheduler'] = 'unknown'
            records.append(record)
    return records


def _load_epoch_dir_records(path):
    records = []
    for json_path in sorted(glob.glob(osp.join(path, 'epoch_*.json'))):
        with open(json_path, 'r', encoding='utf-8') as infile:
            payload = json.load(infile)
        if 'scale_probs' not in payload:
            continue
        record = dict(payload)
        record['epoch'] = int(record.get('epoch', 0))
        record['iteration'] = int(record.get('iteration', 0))
        record['scale_probs'] = [float(v) for v in record.get('scale_probs', [])]
        if 'scale_candidates' in record:
            record['scale_candidates'] = [float(v) for v in record['scale_candidates']]
        records.append(record)
    return records


def load_scale_prob_records(source_path, fallback_candidates=None):
    source = resolve_history_source(source_path)
    if osp.isdir(source):
        records = _load_epoch_dir_records(source)
    elif source.endswith('.jsonl'):
        records = _load_jsonl_records(source)
    elif source.endswith('.log') or source.endswith('.txt'):
        records = _load_text_log_records(source, fallback_candidates=fallback_candidates)
    elif source.endswith('.json'):
        with open(source, 'r', encoding='utf-8') as infile:
            payload = json.load(infile)
        payload = dict(payload)
        payload['epoch'] = int(payload.get('epoch', 0))
        payload['iteration'] = int(payload.get('iteration', 0))
        payload['scale_probs'] = [float(v) for v in payload.get('scale_probs', [])]
        if 'scale_candidates' in payload:
            payload['scale_candidates'] = [float(v) for v in payload['scale_candidates']]
        records = [payload]
    else:
        raise ValueError('Unsupported scale-probability source: {}'.format(source))

    if not records:
        raise ValueError('No scale-probability records found in {}'.format(source))

    if fallback_candidates is not None:
        for record in records:
            if 'scale_candidates' not in record or not record['scale_candidates']:
                record['scale_candidates'] = [float(v) for v in fallback_candidates]

    return records, source


def dedupe_epoch_records(records):
    selected = {}
    for record in records:
        epoch = int(record.get('epoch', 0))
        current = selected.get(epoch)
        if current is None:
            selected[epoch] = record
            continue
        current_type = current.get('record_type')
        record_type = record.get('record_type')
        if current_type != 'epoch_end' and record_type == 'epoch_end':
            selected[epoch] = record
            continue
        if record_type == current_type:
            current_iter = int(current.get('iteration', 0))
            record_iter = int(record.get('iteration', 0))
            if record_iter >= current_iter:
                selected[epoch] = record
    return [selected[epoch] for epoch in sorted(selected)]


def average_scale_probs(records, warmup_epochs=0):
    epoch_records = dedupe_epoch_records(records)
    usable = [record for record in epoch_records if int(record.get('epoch', 0)) > int(warmup_epochs)]
    if not usable:
        usable = epoch_records
    if not usable:
        raise ValueError('No usable scale-probability records after warmup filtering.')

    base_candidates = usable[0].get('scale_candidates')
    if not base_candidates:
        raise ValueError('scale_candidates are missing from the scale-probability records.')

    prob_matrix = []
    for record in usable:
        candidates = record.get('scale_candidates')
        if [float(v) for v in candidates] != [float(v) for v in base_candidates]:
            raise ValueError('scale_candidates changed across records; averaging is ambiguous.')
        probs = np.asarray(record.get('scale_probs', []), dtype=np.float64)
        if probs.size != len(base_candidates):
            raise ValueError('scale_probs length mismatch for epoch {}'.format(record.get('epoch')))
        probs = probs / probs.sum()
        prob_matrix.append(probs)

    mean_probs = np.mean(np.stack(prob_matrix, axis=0), axis=0)
    mean_probs = mean_probs / mean_probs.sum()
    return base_candidates, mean_probs.tolist(), usable
