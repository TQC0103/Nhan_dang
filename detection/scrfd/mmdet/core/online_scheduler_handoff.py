import json
import os

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook

from mmdet.core.sample_redistribution import (
    get_redistribution_state,
    normalize_redistribution_cfg,
)


@HOOKS.register_module()
class OnlineSchedulerHandoffHook(Hook):
    """Online crop scheduler driven by stride-wise SCRFD training statistics.

    This hook is adapted from the standalone ``online_scheduler_handoff``
    prototype, but writes scale probabilities to the same
    ``current_scale_probs.json`` file already consumed by SCRFD's
    ``AdaptiveScalePolicyReader``. That keeps the inference path unchanged and
    lets the handoff scheduler coexist with the existing JSAR instrumentation.
    """

    def __init__(self,
                 redistribution_cfg=None,
                 crop_choice=None,
                 state_file=None,
                 target_strides=(8, 16, 32),
                 target_positive_ratios=(0.5, 0.3, 0.2),
                 loss_weight=0.65,
                 deficit_weight=0.35,
                 update_momentum=0.6,
                 temperature=0.8,
                 min_crop_prob=0.03,
                 log_interval=1):
        self.redistribution_cfg = normalize_redistribution_cfg(redistribution_cfg)
        self.crop_choice = None if crop_choice is None else [float(x) for x in crop_choice]
        self.state_file = state_file
        self.target_strides = tuple(int(s) for s in target_strides)
        self.target_positive_ratios = self._normalize(target_positive_ratios)
        self.loss_weight = float(loss_weight)
        self.deficit_weight = float(deficit_weight)
        self.update_momentum = float(update_momentum)
        self.temperature = max(float(temperature), 1e-4)
        self.min_crop_prob = float(min_crop_prob)
        self.log_interval = max(1, int(log_interval))
        self._crop_probs = None
        self._legacy_state_path = None
        self._summary_dir = None

    def before_run(self, runner):
        state = get_redistribution_state(self.redistribution_cfg)
        state.attach_work_dir(runner.work_dir)
        self._summary_dir = os.path.join(state.state_dir, 'online_scheduler_handoff')
        os.makedirs(os.path.join(self._summary_dir, 'epoch_logs'), exist_ok=True)
        self._legacy_state_path = self._resolve_legacy_state_path(state.state_dir)

        if not self._ensure_initialized(runner, state):
            return
        self._sync_runtime_state(
            state,
            epoch=runner.epoch + 1,
            iteration=runner.iter + 1,
            stride_metrics={},
            urgency=[],
            raw_scores=[],
            record_type='init',
        )
        state.flush_epoch(runner.epoch + 1, runner.iter + 1, logger=None, record_history=False)

    def before_train_epoch(self, runner):
        state = get_redistribution_state(self.redistribution_cfg)
        state.start_epoch(runner.epoch + 1)
        head = self._get_bbox_head(runner)
        if head is not None and hasattr(head, 'reset_sr_epoch_stats'):
            head.reset_sr_epoch_stats()

    def after_train_epoch(self, runner):
        state = get_redistribution_state(self.redistribution_cfg)
        if self.crop_choice is None or self._crop_probs is None:
            if not self._ensure_initialized(runner, state):
                return

        head = self._get_bbox_head(runner)
        if head is None or not hasattr(head, 'get_sr_epoch_stats'):
            runner.logger.warning(
                'OnlineSchedulerHandoffHook requires SCRFDHead.get_sr_epoch_stats(); '
                'the hook will stay inactive.')
            return

        stats = head.get_sr_epoch_stats()
        if not stats:
            return

        pos_values, loss_values = self._reduce_stats(stats)
        pos_ratios, loss_ratios = self._extract_ratios(pos_values, loss_values)

        urgency = []
        for idx in range(len(self.target_strides)):
            deficit = max(0.0, self.target_positive_ratios[idx] - pos_ratios[idx])
            urgency.append(self.loss_weight * loss_ratios[idx] +
                           self.deficit_weight * deficit)

        raw_scores = []
        for scale in self.crop_choice:
            prefs = self._scale_preferences(scale)
            raw_scores.append(float(np.dot(prefs, urgency)))

        fresh_probs = self._softmax(raw_scores)
        mixed_probs = [
            self.update_momentum * old + (1.0 - self.update_momentum) * new
            for old, new in zip(self._crop_probs, fresh_probs)
        ]
        self._crop_probs = self._with_floor(mixed_probs)

        stride_metrics = {
            str(stride): {
                'pos_count': float(pos_values[idx]),
                'pos_ratio': float(pos_ratios[idx]),
                'loss_sum': float(loss_values[idx]),
                'loss_ratio': float(loss_ratios[idx]),
            }
            for idx, stride in enumerate(self.target_strides)
        }

        self._sync_runtime_state(
            state,
            epoch=runner.epoch + 1,
            iteration=runner.iter + 1,
            stride_metrics=stride_metrics,
            urgency=urgency,
            raw_scores=raw_scores,
            record_type='epoch_end',
        )
        state.flush_epoch(runner.epoch + 1, runner.iter + 1, logger=None)

        if (runner.epoch + 1) % self.log_interval == 0:
            runner.logger.info(
                'Online Scheduler Handoff epoch %d | pos=%s loss=%s scale_probs=%s',
                runner.epoch + 1,
                ['{:.4f}'.format(v) for v in pos_ratios],
                ['{:.4f}'.format(v) for v in loss_ratios],
                ['{:.4f}'.format(v) for v in self._crop_probs],
            )

    def _ensure_initialized(self, runner, state):
        if self.crop_choice is None:
            crop_choice = self._find_crop_choice(runner)
            if crop_choice is None:
                runner.logger.warning(
                    'OnlineSchedulerHandoffHook could not find RandomSquareCrop in '
                    'the train pipeline. The hook will stay inactive.')
                return False
            self.crop_choice = [float(x) for x in crop_choice]

        if self._crop_probs is None:
            self._crop_probs = self._uniform(len(self.crop_choice))

        state.scale_candidates = list(self.crop_choice)
        state.default_scale_probs = np.full(
            (len(self.crop_choice), ),
            1.0 / len(self.crop_choice),
            dtype=np.float64)
        if state.current_scale_probs.shape[0] != len(self.crop_choice):
            state.current_scale_probs = state.default_scale_probs.copy()
        else:
            state.current_scale_probs = state.current_scale_probs / state.current_scale_probs.sum()
        self._crop_probs = state.current_scale_probs.tolist()
        return True

    def _sync_runtime_state(self, state, epoch, iteration, stride_metrics, urgency, raw_scores, record_type):
        state.current_epoch = int(epoch)
        state.current_iter = int(iteration)
        state.current_scale_probs = np.asarray(self._crop_probs, dtype=np.float64)
        state.current_scale_probs = state.current_scale_probs / state.current_scale_probs.sum()
        payload = {
            'epoch': int(epoch),
            'iteration': int(iteration),
            'state_key': self.redistribution_cfg['STATE_KEY'],
            'scale_candidates': list(self.crop_choice),
            'scale_probs': list(self._crop_probs),
            'crop_choice': list(self.crop_choice),
            'crop_choice_weights': list(self._crop_probs),
            'target_strides': list(self.target_strides),
            'target_positive_ratios': list(self.target_positive_ratios),
            'stride_metrics': stride_metrics,
            'urgency': [float(v) for v in urgency],
            'raw_scores': [float(v) for v in raw_scores],
            'scheduler': 'online_scheduler_handoff',
            'record_type': record_type,
        }
        if state.scale_state_path:
            self._write_json(state.scale_state_path, payload)
        if getattr(state, 'scale_history_path', None):
            state.record_scale_history(payload)
        if self._legacy_state_path:
            self._write_json(self._legacy_state_path, payload)
        if self._summary_dir:
            self._write_json(
                os.path.join(self._summary_dir, 'latest_summary.json'),
                payload)
            self._write_json(
                os.path.join(self._summary_dir, 'epoch_logs', 'epoch_{:03d}.json'.format(int(epoch))),
                payload)

    def _resolve_legacy_state_path(self, state_dir):
        if not self.state_file:
            return None
        if os.path.isabs(self.state_file):
            return self.state_file
        return os.path.join(state_dir, self.state_file)

    def _find_crop_choice(self, runner):
        dataset = getattr(runner.data_loader, 'dataset', None)
        transform = self._find_crop_transform_from_dataset(dataset)
        if transform is None:
            return None
        return getattr(transform, 'crop_choice', None)

    def _find_crop_transform_from_dataset(self, dataset):
        if dataset is None:
            return None
        pipeline = getattr(dataset, 'pipeline', None)
        transforms = getattr(pipeline, 'transforms', None)
        if transforms:
            for transform in transforms:
                if type(transform).__name__ == 'RandomSquareCrop':
                    return transform
        child = getattr(dataset, 'dataset', None)
        if child is not None:
            found = self._find_crop_transform_from_dataset(child)
            if found is not None:
                return found
        children = getattr(dataset, 'datasets', None)
        if children:
            for item in children:
                found = self._find_crop_transform_from_dataset(item)
                if found is not None:
                    return found
        return None

    def _get_bbox_head(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        return getattr(model, 'bbox_head', None)

    def _reduce_stats(self, stats):
        pos_values = np.asarray(
            [float(stats.get('pos_counts', {}).get(stride, 0.0)) for stride in self.target_strides],
            dtype=np.float64)
        loss_values = np.asarray(
            [float(stats.get('loss_sums', {}).get(stride, 0.0)) for stride in self.target_strides],
            dtype=np.float64)

        if dist.is_available() and dist.is_initialized():
            device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
            packed = torch.tensor(
                np.concatenate([pos_values, loss_values]).tolist(),
                dtype=torch.float64,
                device=device)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            packed = packed.cpu().numpy()
            split = len(self.target_strides)
            pos_values = packed[:split]
            loss_values = packed[split:]
        return pos_values, loss_values

    def _extract_ratios(self, pos_values, loss_values):
        pos_total = float(pos_values.sum())
        loss_total = float(loss_values.sum())
        pos_ratios = self._uniform(len(pos_values)) if pos_total <= 0 else [
            float(value / pos_total) for value in pos_values
        ]
        loss_ratios = self._uniform(len(loss_values)) if loss_total <= 0 else [
            float(value / loss_total) for value in loss_values
        ]
        return pos_ratios, loss_ratios

    def _scale_preferences(self, scale):
        delta = float(scale) - 1.0
        prefer_stride8 = 0.2 + max(0.0, delta) * 1.5
        prefer_stride32 = 0.2 + max(0.0, -delta) * 1.5
        prefer_stride16 = 0.2 + max(0.0, 1.0 - abs(delta) / 0.6)
        return np.asarray(
            self._normalize([prefer_stride8, prefer_stride16, prefer_stride32]),
            dtype=np.float32)

    def _softmax(self, values):
        values = np.asarray(values, dtype=np.float32) / self.temperature
        values -= values.max()
        probs = np.exp(values)
        probs_sum = probs.sum()
        if not np.isfinite(probs_sum) or probs_sum <= 0:
            return self._uniform(len(values))
        probs /= probs_sum
        return probs.tolist()

    def _with_floor(self, probs):
        min_prob = max(0.0, min(self.min_crop_prob, 1.0 / max(len(probs), 1)))
        probs = np.asarray(self._normalize(probs), dtype=np.float32)
        probs = np.maximum(probs, min_prob)
        probs /= probs.sum()
        return probs.tolist()

    def _uniform(self, length):
        if length <= 0:
            return []
        return [1.0 / length for _ in range(length)]

    def _normalize(self, values):
        values = [max(float(v), 1e-8) for v in values]
        total = sum(values)
        if total <= 0:
            return self._uniform(len(values))
        return [value / total for value in values]

    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = '{}.tmp'.format(path)
        with open(tmp_path, 'w', encoding='utf-8') as outfile:
            json.dump(payload, outfile, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
