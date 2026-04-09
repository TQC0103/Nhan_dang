"""
Custom logging hooks for Knowledge Distillation experiments.

These hooks read values from ``runner.log_buffer.output`` so they track the
same scalars the default MMDetection logger sees after loss parsing.
"""

import os
import csv
from collections import OrderedDict

import torch
from mmcv.runner import HOOKS, LoggerHook


@HOOKS.register_module()
class KDTensorboardLoggerHook(LoggerHook):
    """Enhanced TensorBoard logger for KD experiments.

    Logs all loss components including distillation losses and their ratios.
    """

    def __init__(self,
                 log_dir=None,
                 csv_log_file=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        super(KDTensorboardLoggerHook, self).__init__(
            interval=interval,
            ignore_last=ignore_last,
            reset_flag=reset_flag,
            by_epoch=by_epoch)
        self.csv_log_file = csv_log_file
        self.resolved_csv_log_file = None
        self.csv_file_handle = None
        self.csv_writer = None
        self.csv_headers = None
        self.log_dir = log_dir

    def before_run(self, runner):
        super(KDTensorboardLoggerHook, self).before_run(runner)
        if self.csv_log_file:
            self.resolved_csv_log_file = self.csv_log_file.replace(
                '${work_dir}', runner.work_dir)
            # Ensure log directory exists
            log_dir = os.path.dirname(self.resolved_csv_log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Create CSV file with headers
            self.csv_file_handle = open(self.resolved_csv_log_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file_handle)

    def log(self, runner):
        """Log loss components to CSV and other hooks."""
        if not self.every_n_iters(runner, self.interval):
            return

        # Get current loss values
        log_items = self.get_loggable_items(runner)
        if not log_items:
            return

        # Add to CSV
        if self.resolved_csv_log_file and self.csv_file_handle:
            row_data = {
                'iter': runner.iter,
                'epoch': runner.epoch,
            }
            row_data.update(log_items)

            if self.csv_headers is None:
                self.csv_headers = ['iter', 'epoch'] + list(log_items.keys())
                self.csv_writer.writerow(self.csv_headers)

            self.csv_writer.writerow([row_data.get(k, '') for k in self.csv_headers])
            self.csv_file_handle.flush()

    def get_loggable_items(self, runner):
        """Extract loggable items including all loss components."""
        log_items = OrderedDict()

        outputs = getattr(getattr(runner, 'log_buffer', None), 'output', {})
        for key, value in outputs.items():
            if isinstance(value, (int, float)):
                log_items[key] = float(value)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                log_items[key] = value.item()

        # Calculate distillation ratios if both task and distill losses exist
        if 'loss_cls' in log_items and 'loss_cls_distill' in log_items:
            total_cls = log_items['loss_cls'] + log_items.get('loss_cls_distill', 0)
            if total_cls > 0:
                log_items['cls_distill_ratio'] = log_items['loss_cls_distill'] / total_cls

        if 'loss_bbox' in log_items and 'loss_bbox_distill' in log_items:
            total_bbox = log_items['loss_bbox'] + log_items.get('loss_bbox_distill', 0)
            if total_bbox > 0:
                log_items['bbox_distill_ratio'] = log_items['loss_bbox_distill'] / total_bbox

        # Calculate total loss components
        task_loss = sum(v for k, v in log_items.items()
                       if k.startswith('loss_') and 'distill' not in k and k != 'loss'
                       and isinstance(v, (int, float)))
        distill_loss = sum(v for k, v in log_items.items()
                          if 'distill' in k and isinstance(v, (int, float)))

        log_items['task_loss_sum'] = task_loss
        log_items['distill_loss_sum'] = distill_loss
        if task_loss + distill_loss > 0:
            log_items['distill_total_ratio'] = distill_loss / (task_loss + distill_loss)

        return log_items

    def after_run(self, runner):
        if self.csv_file_handle:
            self.csv_file_handle.close()
            self.csv_file_handle = None


@HOOKS.register_module()
class KDTextLoggerHook(LoggerHook):
    """Enhanced text logger for KD experiments.

    Logs detailed loss breakdown at each interval.
    """

    def __init__(self,
                 interval=100,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True,
                 log_loss_components=True):
        super(KDTextLoggerHook, self).__init__(
            interval=interval,
            ignore_last=ignore_last,
            reset_flag=reset_flag,
            by_epoch=by_epoch)
        self.log_loss_components = log_loss_components

    def log(self, runner):
        """Log with detailed loss breakdown."""
        if not self.every_n_iters(runner, self.interval):
            return

        # Get loggable items
        log_items = self._get_loggable_items(runner)

        # Format log string
        log_str = f'Epoch[{runner.epoch}][{runner.iter % len(runner.data_loader) if hasattr(runner, "data_loader") else runner.iter}] '
        loss_strs = []

        # Format all losses
        for key, value in sorted(log_items.items()):
            if isinstance(value, float):
                if 'distill' in key.lower():
                    loss_strs.append(f'{key}: {value:.6f}')
                else:
                    loss_strs.append(f'{key}: {value:.4f}')

        log_str += ', '.join(loss_strs)
        runner.logger.info(log_str)

    def _get_loggable_items(self, runner):
        """Get all loggable loss items."""
        log_items = OrderedDict()

        outputs = getattr(getattr(runner, 'log_buffer', None), 'output', {})
        for key, value in outputs.items():
            if isinstance(value, (int, float)):
                log_items[key] = float(value)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                log_items[key] = value.item()

        return log_items
