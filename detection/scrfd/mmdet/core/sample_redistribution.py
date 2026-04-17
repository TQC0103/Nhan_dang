import json
import math
import os
import threading

import numpy as np
import torch
from mmcv.runner import HOOKS, Hook


DEFAULT_STATE_KEY = 'default'
DEFAULT_BIN_NAMES = ('tiny', 'small', 'medium', 'large')
_STATES = {}
_STATES_LOCK = threading.Lock()


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if key in cfg:
        return cfg[key]
    lower = key.lower()
    if lower in cfg:
        return cfg[lower]
    return default


def normalize_redistribution_cfg(cfg=None):
    cfg = dict(cfg or {})
    scale_candidates = _cfg_get(
        cfg,
        'ADAPTIVE_SR_SCALE_CANDIDATES',
        [0.35, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    )
    bin_edges = _cfg_get(
        cfg,
        'ADAPTIVE_SR_BIN_EDGES',
        [0.0, 16.0, 32.0, 96.0, float('inf')],
    )
    return {
        'STATE_KEY': str(_cfg_get(cfg, 'STATE_KEY', DEFAULT_STATE_KEY)),
        'STATE_DIR': _cfg_get(cfg, 'STATE_DIR', None),
        'ENABLE_ADAPTIVE_SR': bool(_cfg_get(cfg, 'ENABLE_ADAPTIVE_SR', False)),
        'ADAPTIVE_SR_WARMUP_EPOCHS': int(_cfg_get(cfg, 'ADAPTIVE_SR_WARMUP_EPOCHS', 1)),
        'ADAPTIVE_SR_UPDATE_INTERVAL': int(_cfg_get(cfg, 'ADAPTIVE_SR_UPDATE_INTERVAL', 1000)),
        'ADAPTIVE_SR_EMA': float(_cfg_get(cfg, 'ADAPTIVE_SR_EMA', 0.8)),
        'ADAPTIVE_SR_MIN_PROB': float(_cfg_get(cfg, 'ADAPTIVE_SR_MIN_PROB', 0.03)),
        'ADAPTIVE_SR_SCALE_CANDIDATES': [float(v) for v in scale_candidates],
        'ADAPTIVE_SR_BIN_EDGES': [float(v) for v in bin_edges],
        'ADAPTIVE_SR_DIFFICULTY_MODE': str(_cfg_get(cfg, 'ADAPTIVE_SR_DIFFICULTY_MODE', 'loss_recall')),
        'ADAPTIVE_SR_LOGGING': bool(_cfg_get(cfg, 'ADAPTIVE_SR_LOGGING', True)),
        'ENABLE_JSAR': bool(_cfg_get(cfg, 'ENABLE_JSAR', False)),
        'JSAR_MODE': str(_cfg_get(cfg, 'JSAR_MODE', 'hybrid_fallback')),
        'JSAR_TINY_MAX_SIZE': float(_cfg_get(cfg, 'JSAR_TINY_MAX_SIZE', 16.0)),
        'JSAR_SMALL_MAX_SIZE': float(_cfg_get(cfg, 'JSAR_SMALL_MAX_SIZE', 32.0)),
        'JSAR_TINY_IOU_DELTA': float(_cfg_get(cfg, 'JSAR_TINY_IOU_DELTA', 0.05)),
        'JSAR_SMALL_IOU_DELTA': float(_cfg_get(cfg, 'JSAR_SMALL_IOU_DELTA', 0.02)),
        'JSAR_TOPK': int(_cfg_get(cfg, 'JSAR_TOPK', 4)),
        'JSAR_CENTER_RADIUS_SCALE': float(_cfg_get(cfg, 'JSAR_CENTER_RADIUS_SCALE', 1.3)),
        'JSAR_SOFT_WEIGHT_TEMPERATURE': float(_cfg_get(cfg, 'JSAR_SOFT_WEIGHT_TEMPERATURE', 0.75)),
        'JSAR_MIN_POS_PER_TINY_GT': int(_cfg_get(cfg, 'JSAR_MIN_POS_PER_TINY_GT', 3)),
        'JSAR_LOGGING': bool(_cfg_get(cfg, 'JSAR_LOGGING', True)),
    }


def get_bin_names(bin_edges):
    num_bins = len(bin_edges) - 1
    if num_bins <= len(DEFAULT_BIN_NAMES):
        return list(DEFAULT_BIN_NAMES[:num_bins])
    return ['bin_{}'.format(i) for i in range(num_bins)]


def get_state_dir_env_name(state_key):
    safe_key = ''.join(ch.upper() if ch.isalnum() else '_' for ch in str(state_key))
    return 'SCRFD_REDIS_STATE_DIR_{}'.format(safe_key or DEFAULT_STATE_KEY.upper())


def compute_face_sizes_from_boxes(boxes):
    if boxes is None:
        return None
    if torch.is_tensor(boxes):
        if boxes.numel() == 0:
            return boxes.new_zeros((0, ), dtype=torch.float)
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
        return torch.sqrt(widths * heights)
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return np.zeros((0, ), dtype=np.float32)
    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=1e-6, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=1e-6, a_max=None)
    return np.sqrt(widths * heights)


def assign_size_bins_from_sizes(sizes, bin_edges):
    if torch.is_tensor(sizes):
        output = torch.full((sizes.shape[0], ), len(bin_edges) - 2, dtype=torch.long, device=sizes.device)
        for idx in range(len(bin_edges) - 1):
            left = bin_edges[idx]
            right = bin_edges[idx + 1]
            if math.isinf(right):
                mask = sizes >= left
            else:
                mask = (sizes >= left) & (sizes < right)
            output[mask] = idx
        return output
    sizes = np.asarray(sizes)
    output = np.full((sizes.shape[0], ), len(bin_edges) - 2, dtype=np.int64)
    for idx in range(len(bin_edges) - 1):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if math.isinf(right):
            mask = sizes >= left
        else:
            mask = (sizes >= left) & (sizes < right)
        output[mask] = idx
    return output


def compute_face_size_bins(boxes, bin_edges):
    sizes = compute_face_sizes_from_boxes(boxes)
    if sizes is None:
        return None
    return assign_size_bins_from_sizes(sizes, bin_edges)


def bins_to_hist(bin_indices, num_bins):
    hist = [0 for _ in range(num_bins)]
    if bin_indices is None:
        return hist
    if torch.is_tensor(bin_indices):
        if bin_indices.numel() == 0:
            return hist
        values, counts = torch.unique(bin_indices.detach().cpu(), return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            if 0 <= value < num_bins:
                hist[int(value)] += int(count)
        return hist
    values, counts = np.unique(np.asarray(bin_indices), return_counts=True)
    for value, count in zip(values.tolist(), counts.tolist()):
        if 0 <= value < num_bins:
            hist[int(value)] += int(count)
    return hist


def _to_float_list(values):
    if torch.is_tensor(values):
        return values.detach().cpu().float().tolist()
    return [float(v) for v in values]


class AdaptiveScalePolicyReader(object):
    """File-backed scale distribution reader for data loader workers."""

    def __init__(self, redistribution_cfg=None, fallback_candidates=None):
        self.cfg = normalize_redistribution_cfg(redistribution_cfg)
        self.fallback_candidates = [
            float(v) for v in (fallback_candidates or self.cfg['ADAPTIVE_SR_SCALE_CANDIDATES'])
        ]
        self._cached_candidates = list(self.fallback_candidates)
        self._cached_probs = None
        self._cached_mtime = None

    def _get_scale_state_path(self):
        state_dir = self.cfg['STATE_DIR']
        if not state_dir:
            state_dir = os.environ.get(get_state_dir_env_name(self.cfg['STATE_KEY']))
        if not state_dir:
            return None
        return os.path.join(state_dir, 'current_scale_probs.json')

    def get_distribution(self):
        if not self.cfg['ENABLE_ADAPTIVE_SR']:
            return list(self.fallback_candidates), None
        scale_state_path = self._get_scale_state_path()
        if not scale_state_path or not os.path.exists(scale_state_path):
            return list(self.fallback_candidates), None
        try:
            mtime = os.path.getmtime(scale_state_path)
            if self._cached_mtime is not None and self._cached_mtime == mtime:
                return list(self._cached_candidates), self._cached_probs
            with open(scale_state_path, 'r', encoding='utf-8') as infile:
                payload = json.load(infile)
            candidates = [float(v) for v in payload.get('scale_candidates', self.fallback_candidates)]
            probs = payload.get('scale_probs', None)
            if probs is not None:
                probs = np.asarray(probs, dtype=np.float64)
                if probs.shape[0] != len(candidates) or probs.sum() <= 0:
                    probs = None
                else:
                    probs = probs / probs.sum()
            self._cached_candidates = candidates
            self._cached_probs = probs
            self._cached_mtime = mtime
            return list(self._cached_candidates), self._cached_probs
        except Exception:
            return list(self.fallback_candidates), None

    def sample(self, rng=None):
        candidates, probs = self.get_distribution()
        rng = np.random if rng is None else rng
        if probs is None:
            return float(rng.choice(candidates))
        return float(rng.choice(candidates, p=probs))


class AdaptiveRedistributionRuntime(object):
    def __init__(self, redistribution_cfg=None):
        self.cfg = normalize_redistribution_cfg(redistribution_cfg)
        self.bin_edges = self.cfg['ADAPTIVE_SR_BIN_EDGES']
        self.bin_names = get_bin_names(self.bin_edges)
        self.num_bins = len(self.bin_names)
        self.scale_candidates = list(self.cfg['ADAPTIVE_SR_SCALE_CANDIDATES'])
        self.default_scale_probs = np.full(
            (len(self.scale_candidates), ),
            1.0 / len(self.scale_candidates),
            dtype=np.float64,
        )
        self.current_scale_probs = self.default_scale_probs.copy()
        self.current_epoch = 0
        self.current_iter = 0
        self._last_update_iter = 0
        self._lock = threading.RLock()
        self._last_difficulty = np.zeros((self.num_bins, ), dtype=np.float64)
        self.state_dir = None
        self.scale_state_path = None
        self.scale_history_path = None
        self.epoch_log_dir = None
        self._epoch_stats = self._empty_epoch_stats()

    def _empty_epoch_stats(self):
        return {
            'gt_hist': [0 for _ in range(self.num_bins)],
            'pos_hist': [0 for _ in range(self.num_bins)],
            'jsar_before_hist': [0 for _ in range(self.num_bins)],
            'jsar_after_hist': [0 for _ in range(self.num_bins)],
            'cls_loss': [0.0 for _ in range(self.num_bins)],
            'box_loss': [0.0 for _ in range(self.num_bins)],
            'level_pos': {},
            'num_images': 0,
            'num_batches': 0,
        }

    def _accumulate(self, dst, src):
        for idx, value in enumerate(src):
            dst[idx] += value

    def attach_work_dir(self, work_dir):
        state_dir = self.cfg['STATE_DIR'] or os.path.join(work_dir, 'adaptive_sr')
        with self._lock:
            self.state_dir = state_dir
            self.scale_state_path = os.path.join(state_dir, 'current_scale_probs.json')
            self.scale_history_path = os.path.join(state_dir, 'scale_prob_history.jsonl')
            self.epoch_log_dir = os.path.join(state_dir, 'epoch_logs')
            os.makedirs(self.epoch_log_dir, exist_ok=True)
            os.environ[get_state_dir_env_name(self.cfg['STATE_KEY'])] = state_dir
            self._write_scale_state_locked()

    def start_epoch(self, epoch):
        with self._lock:
            self.current_epoch = int(epoch)
            self._epoch_stats = self._empty_epoch_stats()

    def note_batch(self):
        with self._lock:
            self._epoch_stats['num_batches'] += 1

    def note_gt_boxes(self, gt_bboxes_list):
        with self._lock:
            for gt_bboxes in gt_bboxes_list:
                gt_hist = bins_to_hist(compute_face_size_bins(gt_bboxes, self.bin_edges), self.num_bins)
                self._accumulate(self._epoch_stats['gt_hist'], gt_hist)
                self._epoch_stats['num_images'] += 1

    def note_pos_bins(self, sample_bin_labels, stride):
        hist = bins_to_hist(sample_bin_labels, self.num_bins)
        stride_key = str(int(stride))
        with self._lock:
            self._accumulate(self._epoch_stats['pos_hist'], hist)
            self._epoch_stats['level_pos'][stride_key] = (
                self._epoch_stats['level_pos'].get(stride_key, 0) + int(sum(hist))
            )

    def note_jsar_stats(self, before_hist, after_hist):
        with self._lock:
            self._accumulate(self._epoch_stats['jsar_before_hist'], before_hist)
            self._accumulate(self._epoch_stats['jsar_after_hist'], after_hist)

    def note_loss_bins(self, cls_loss_by_bin, box_loss_by_bin):
        with self._lock:
            self._accumulate(self._epoch_stats['cls_loss'], _to_float_list(cls_loss_by_bin))
            self._accumulate(self._epoch_stats['box_loss'], _to_float_list(box_loss_by_bin))

    def _normalize_component(self, values):
        values = np.asarray(values, dtype=np.float64)
        positive = values[values > 0]
        if positive.size == 0:
            return np.zeros_like(values)
        return values / (positive.mean() + 1e-6)

    def _build_scale_support(self):
        desired = np.asarray([1.8, 1.35, 1.0, 0.72], dtype=np.float64)
        if desired.shape[0] != self.num_bins:
            desired = np.linspace(1.8, 0.75, self.num_bins)
        rows = []
        for scale in self.scale_candidates:
            row = np.exp(-np.abs(np.log(scale + 1e-6) - np.log(desired)) / 0.45)
            if scale >= 1.0 and self.num_bins >= 2:
                row[0] *= 1.15
                row[1] *= 1.05
            if scale <= 1.0 and self.num_bins >= 2:
                row[-1] *= 1.15
                row[-2] *= 1.05
            rows.append(row)
        return np.asarray(rows, dtype=np.float64)

    def compute_difficulty(self):
        gt_hist = np.asarray(self._epoch_stats['gt_hist'], dtype=np.float64)
        pos_hist = np.asarray(self._epoch_stats['pos_hist'], dtype=np.float64)
        cls_loss = np.asarray(self._epoch_stats['cls_loss'], dtype=np.float64)
        box_loss = np.asarray(self._epoch_stats['box_loss'], dtype=np.float64)
        recall_gap = np.maximum(0.0, 1.0 - np.minimum(pos_hist / np.maximum(gt_hist, 1.0), 1.0))
        cls_norm = cls_loss / np.maximum(pos_hist, 1.0)
        box_norm = box_loss / np.maximum(pos_hist, 1.0)
        mode = self.cfg['ADAPTIVE_SR_DIFFICULTY_MODE']
        if mode == 'recall':
            difficulty = recall_gap
        elif mode == 'loss_only':
            difficulty = self._normalize_component(cls_norm) + self._normalize_component(box_norm)
        else:
            difficulty = (
                self._normalize_component(recall_gap)
                + self._normalize_component(cls_norm)
                + self._normalize_component(box_norm)
            )
        if difficulty.sum() <= 0:
            difficulty = np.full((self.num_bins, ), 1.0 / self.num_bins, dtype=np.float64)
        else:
            difficulty = difficulty / difficulty.sum()
        self._last_difficulty = difficulty
        return difficulty

    def update_scale_distribution(self, epoch, iteration, force=False):
        if not self.cfg['ENABLE_ADAPTIVE_SR']:
            return False
        if epoch <= self.cfg['ADAPTIVE_SR_WARMUP_EPOCHS']:
            with self._lock:
                self.current_epoch = int(epoch)
                self.current_iter = int(iteration)
                self.current_scale_probs = self.default_scale_probs.copy()
                self._write_scale_state_locked()
            return False
        if (not force
                and self.cfg['ADAPTIVE_SR_UPDATE_INTERVAL'] > 0
                and iteration - self._last_update_iter < self.cfg['ADAPTIVE_SR_UPDATE_INTERVAL']):
            return False
        with self._lock:
            difficulty = self.compute_difficulty()
            raw_scores = self._build_scale_support().dot(difficulty)
            raw_scores = np.maximum(raw_scores, 1e-8)
            raw_probs = raw_scores / raw_scores.sum()
            min_prob = max(0.0, self.cfg['ADAPTIVE_SR_MIN_PROB'])
            if min_prob * raw_probs.shape[0] >= 1.0:
                raise ValueError('ADAPTIVE_SR_MIN_PROB is too large for the number of scale candidates.')
            raw_probs = raw_probs * (1.0 - min_prob * raw_probs.shape[0]) + min_prob
            raw_probs = raw_probs / raw_probs.sum()
            ema = float(np.clip(self.cfg['ADAPTIVE_SR_EMA'], 0.0, 0.9999))
            self.current_scale_probs = ema * self.current_scale_probs + (1.0 - ema) * raw_probs
            self.current_scale_probs = self.current_scale_probs / self.current_scale_probs.sum()
            self.current_epoch = int(epoch)
            self.current_iter = int(iteration)
            self._last_update_iter = int(iteration)
            self._write_scale_state_locked()
        return True

    def snapshot(self):
        with self._lock:
            return {
                'epoch': self.current_epoch,
                'iteration': self.current_iter,
                'bin_edges': list(self.bin_edges),
                'bin_names': list(self.bin_names),
                'scale_candidates': list(self.scale_candidates),
                'scale_probs': self.current_scale_probs.tolist(),
                'difficulty': self._last_difficulty.tolist(),
                'gt_hist': list(self._epoch_stats['gt_hist']),
                'pos_hist': list(self._epoch_stats['pos_hist']),
                'jsar_before_hist': list(self._epoch_stats['jsar_before_hist']),
                'jsar_after_hist': list(self._epoch_stats['jsar_after_hist']),
                'cls_loss': list(self._epoch_stats['cls_loss']),
                'box_loss': list(self._epoch_stats['box_loss']),
                'level_pos': dict(self._epoch_stats['level_pos']),
                'num_images': int(self._epoch_stats['num_images']),
                'num_batches': int(self._epoch_stats['num_batches']),
            }

    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as outfile:
            json.dump(payload, outfile, indent=2, sort_keys=True)

    def _append_jsonl(self, path, payload):
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as outfile:
            outfile.write(json.dumps(payload, sort_keys=True) + '\n')

    def _write_scale_state_locked(self):
        if not self.scale_state_path:
            return
        self._write_json(
            self.scale_state_path,
            {
                'epoch': self.current_epoch,
                'iteration': self.current_iter,
                'scale_candidates': list(self.scale_candidates),
                'scale_probs': self.current_scale_probs.tolist(),
                'state_key': self.cfg['STATE_KEY'],
            },
        )

    def record_scale_history(self, payload):
        if not self.scale_history_path:
            return
        self._append_jsonl(self.scale_history_path, payload)

    def flush_epoch(self, epoch, iteration, logger=None, record_history=True):
        with self._lock:
            self.current_epoch = int(epoch)
            self.current_iter = int(iteration)
            self._write_scale_state_locked()
            snapshot = self.snapshot()
            snapshot['state_key'] = self.cfg['STATE_KEY']
            snapshot['scheduler'] = 'adaptive_sr'
            snapshot['record_type'] = 'epoch_end'
            if self.epoch_log_dir:
                self._write_json(
                    os.path.join(self.epoch_log_dir, 'epoch_{:03d}.json'.format(int(epoch))),
                    snapshot,
                )
                self._write_json(os.path.join(self.state_dir, 'latest_summary.json'), snapshot)
            if record_history:
                self.record_scale_history(snapshot)
        if logger and (self.cfg['ADAPTIVE_SR_LOGGING'] or self.cfg['JSAR_LOGGING']):
            logger.info(
                'Adaptive SR epoch %d | gt=%s pos=%s jsar_before=%s jsar_after=%s scale_probs=%s',
                int(epoch),
                snapshot['gt_hist'],
                snapshot['pos_hist'],
                snapshot['jsar_before_hist'],
                snapshot['jsar_after_hist'],
                ['{:.4f}'.format(v) for v in snapshot['scale_probs']],
            )


def get_redistribution_state(redistribution_cfg=None):
    cfg = normalize_redistribution_cfg(redistribution_cfg)
    state_key = cfg['STATE_KEY']
    with _STATES_LOCK:
        if state_key not in _STATES:
            _STATES[state_key] = AdaptiveRedistributionRuntime(cfg)
        return _STATES[state_key]


@HOOKS.register_module()
class AdaptiveRedistributionHook(Hook):
    def __init__(self, redistribution_cfg=None):
        self.redistribution_cfg = normalize_redistribution_cfg(redistribution_cfg)

    def before_run(self, runner):
        state = get_redistribution_state(self.redistribution_cfg)
        state.attach_work_dir(runner.work_dir)
        state.flush_epoch(
            runner.epoch + 1,
            runner.iter + 1,
            logger=runner.logger,
            record_history=False)

    def before_train_epoch(self, runner):
        get_redistribution_state(self.redistribution_cfg).start_epoch(runner.epoch + 1)

    def after_train_iter(self, runner):
        get_redistribution_state(self.redistribution_cfg).update_scale_distribution(
            runner.epoch + 1, runner.iter + 1)

    def after_train_epoch(self, runner):
        state = get_redistribution_state(self.redistribution_cfg)
        state.update_scale_distribution(runner.epoch + 1, runner.iter + 1, force=True)
        state.flush_epoch(runner.epoch + 1, runner.iter + 1, logger=runner.logger)
