# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""
import os, json,sys
from transformers import Trainer, is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
from typing import Dict, Iterator, List, Optional, Union

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

from transformers import logging

logger = logging.get_logger(__name__)

class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        # We might have removed columns from the dataset so we put them back.
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        eval_preds = self.post_process_function(test_examples, test_dataset, output.predictions)
        metrics = self.compute_metrics(eval_preds)

        return PredictionOutput(predictions=eval_preds.predictions, label_ids=eval_preds.label_ids, metrics=metrics)

    def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Reformat Trainer metrics values to a human-readable format

        Args:
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predict

        Returns:
            metrics (:obj:`Dict[str, float]`): The reformatted metrics
        """

        metrics_copy = metrics.copy()
        for k, v in metrics_copy.items():
            if "_mem_" in k:
                metrics_copy[k] = f"{ v >> 20 }MB"
            elif k == "total_flos":
                metrics_copy[k] = f"{ int(v) >> 30 }GF"
            elif type(metrics_copy[k]) == float:
                metrics_copy[k] = round(v, 4)

        return metrics_copy


    def log_metrics(self, split, metrics):
        """
        Log metrics in a specially formatted way

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (:obj:`str`):
                Mode/split name: one of ``train``, ``eval``, ``test``
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predictmetrics: metrics dict
        """
        if not self.is_world_process_zero():
            return

        logger.info(f"***** {split} metrics *****")
        metrics_formatted = self.metrics_format(metrics)
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        for key in sorted(metrics_formatted.keys()):
            logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")


    def save_metrics(self, split, metrics, combined=True):
        """
        Save metrics into a json file for that split, e.g. ``train_results.json``.

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (:obj:`str`):
                Mode/split name: one of ``train``, ``eval``, ``test``, ``all``
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predict
            combined (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Creates combined metrics by updating ``all_results.json`` with metrics of this call
        """
        if not self.is_world_process_zero():
            return

        path = os.path.join(self.args.output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        if combined:
            path = os.path.join(self.args.output_dir, "all_results.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)
            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=4, sort_keys=True)

