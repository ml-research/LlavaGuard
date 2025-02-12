import json
import math
import os
import time
from typing import Dict, Optional, List, Union, Tuple, Any
from xmlrpc.client import Error
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images
import torch.nn.functional as F

import torch
from rtpt import rtpt
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm
from transformers import TrainerCallback
from transformers.trainer_pt_utils import nested_concat, nested_numpify, nested_detach
from transformers.trainer_utils import speed_metrics, EvalLoopOutput
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import is_datasets_available
import datasets
from llava.train.llava_trainer import LLaVATrainer
from llavaguard.data.build_dataset import get_gt_assessment, get_prompt_and_gt
from llavaguard.evaluation.metrics_calculator import parse_json, EvaluationMetricsCalculator
from transformers import EvalPrediction

class LlavaGuardCallback(TrainerCallback):
    "A callback that updates remaining time using rtpt"

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        model_name = 'LlavaGuard'
        # save state to output dir
        os.makedirs('output/tmp', exist_ok=True)
        state.save_to_json(json_path='output/tmp/state.json')
        # print(f'current step: {state.global_step}, max steps: {state.max_steps}')
        self.r = rtpt.RTPT(name_initials='LH', experiment_name=f'{model_name}-Trainer',
                           max_iterations=state.max_steps - state.global_step)
        self.r.start()

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        self.r.step()

    # def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     if rtpt not initialized, initialize it
    #     model_name = 'LlavaGuard'
    #     if not hasattr(self, 'r'):
    #         self.r = rtpt.RTPT(name_initials='LH', experiment_name=f'{model_name}-eval',
    #                            max_iterations=1)
    #         self.r.start()
    #     return control


class LlavaGuardTrainer(LLaVATrainer):
    def __init__(self, *args, **kwargs):
        # add callback to kwargs
        kwargs['callbacks'] = [LlavaGuardCallback]
        super().__init__(*args, **kwargs)


class LlavaGuardTrainerWithMetrics(LLaVATrainer):
    def __init__(self, data_collator_eval=None, *args, **kwargs):
        # add callback to kwargs
        kwargs['callbacks'] = [LlavaGuardCallback]
        kwargs['compute_metrics'] = self.compute_metrics_llavaguard
        super().__init__(*args, **kwargs)
        self.data_collator_eval = data_collator_eval if data_collator_eval is not None else self.data_collator


    def compute_metrics_llavaguard(self, EvalPrediction):
        label_ids = EvalPrediction.label_ids
        predictions = EvalPrediction.predictions
        # all_inputs = EvalPrediction.inputs
        emc = EvaluationMetricsCalculator(pred_dir='output/eval/tmp/pred', debug=False)
        preds = predictions.argmax(-1)
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        batch_txt = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        batch_gt = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        for i, (p_text, gt) in enumerate(zip(batch_txt, batch_gt)):
            emc.add_sample(f'sample_{i}', p_text, gt, save_output=False)
        metrics, _ = emc.compute_stats()
        return {
            'ACC': metrics['Balanced Accuracy'],
            "Recall": metrics["Recall"],
            "SPC": metrics["Specificity"],
            "Precision": metrics["Precision"],
            "F1": metrics["F1"],
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'TN': metrics['TN'],
            'FN': metrics['FN'],
            'Invalid': metrics['Invalid'],
        }

        
    # def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
    #     """
    #     Returns the evaluation [`~torch.utils.data.DataLoader`].

    #     Subclass and override this method if you want to inject some custom behavior.

    #     Args:
    #         eval_dataset (`torch.utils.data.Dataset`, *optional*):
    #             If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
    #             by the `model.forward()` method are automatically removed. It must implement `__len__`.
    #     """
    #     if eval_dataset is None and self.eval_dataset is None:
    #         raise ValueError("Trainer: evaluation requires an eval_dataset.")
    #     eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
    #     data_collator = self.data_collator_eval if self.data_collator_eval is not None else self.data_collator

    #     if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
    #         eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
    #     else:
    #         data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

    #     dataloader_params = {
    #         "batch_size": self.args.eval_batch_size,
    #         "collate_fn": data_collator,
    #         "num_workers": self.args.dataloader_num_workers,
    #         "pin_memory": self.args.dataloader_pin_memory,
    #         "persistent_workers": self.args.dataloader_persistent_workers,
    #     }

    #     if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
    #         dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
    #         dataloader_params["drop_last"] = self.args.dataloader_drop_last

    #     return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
        
    def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        # eval_sample_count = len(eval_dataset) if len(eval_dataset) < 20 else 20
        # create subset of eval dataset
        # subset = Subset(eval_dataset, range(eval_sample_count))
        data_collator_tmp = self.data_collator
        self.data_collator = self.data_collator_eval
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.data_collator = data_collator_tmp
        
        # torch.jit._state.clear_class_registry()
        start_time = time.time()
        output = self.evaluateion_loop(
             eval_dataloader,
             description="Evaluation",
             prediction_loss_only=False,
             ignore_keys=ignore_keys,
             metric_key_prefix=metric_key_prefix,
         )
        # output = self.eval_loop_llavaguard(eval_dataset)
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
        if len(labels) == 1:
            labels = labels[0]

        inputs_without_labels = []
        # split input_ids into input and labels
        for input, label in zip(inputs['input_ids'], labels):
            idx_label_start = (label != -100).nonzero(as_tuple=True)[0][0]
            inputs_without_labels.append(input[:idx_label_start])

        with torch.no_grad():
            input_ids = torch.nn.utils.rnn.pad_sequence(
                inputs_without_labels,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            # we apply padding to the left of the input by reversing the input twice
            outs = model.generate(
                input_ids.to(self.args.device),
                images=inputs['images'].to(self.args.device),
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                num_beams=1,
                max_new_tokens=200,
                use_cache=True,
                stopping_criteria=[KeywordsStoppingCriteria(['}'], self.tokenizer, input_ids)],
                output_scores=True,
                return_dict_in_generate=True,
                # output_logits=True,
            )
        scores = outs['scores']
        # convert tuple([batch_size, vocab_size]) to tensor([batch_size, seq_len, vocab_size])
        output_logits = torch.stack(scores, dim=1)
        input_ids = inputs['input_ids']
        loss = None

        # compute loss
        # print(f'labels ids shape: {labels.shape}')
        # print(f'output logits shape: {output_logits.shape}')

        # label_logits = output_logits[torch.arange(output_logits.size(0)), labels]



        # labels_flat = labels.view(-1)
        # logits_flat = output_logits.view(-1, output_logits.size(-1))
        # print(f'labels_flat shape: {labels_flat.shape}')
        # print(f'logits_flat logits shape: {logits_flat.shape}')
        # loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=self.tokenizer.pad_token_id)

        # loss = None
        # convert output to logits
        # logits = torch.zeros(list(outs.shape) + [self.model.vocab_size]).to(self.args.device)
        # for i, output in enumerate(outs):
        #     for j, token in enumerate(output):
        #         logits[i, j, token] = 1

        # logits = nested_detach(logits)
        # if len(logits) == 1:
        #     logits = logits[0]

        # print(f'logits shape: {output_logits.shape}')
        return (loss, output_logits, labels)


    def eval_loop_llavaguard(self, eval_dataset):
        model = self.model
        tokenizer = self.tokenizer
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        list_data_dict = eval_dataset.list_data_dict
        all_preds = None
        all_labels = None
        all_inputs = None
        data = []
        model.eval()
        # if len(self.accelerator._models) == 0 and model is self.model:
        #     model = (
        #         self.accelerator.prepare(model)
        #         if self.is_deepspeed_enabled
        #         else self.accelerator.prepare_model(model, evaluation_mode=True)
        #     )
        i = 0
        gt = json.loads(list_data_dict[i]['conversations'][1]["value"]).keys()
        for data_dict in tqdm(list_data_dict, desc="Evaluating"):

            input, label = get_prompt_and_gt(data_dict)

            attention_mask = ~data_dict['labels'].ne(-100)
            image_tensor = data_dict['image'].half().unsqueeze(0).to(self.args.device)
            image_tensor_bfloat16 = image_tensor.to(torch.bfloat16)
            # input_txt = self.tokenizer.decode(data_dict['input_ids'], skip_special_tokens=True)
            # print(f'input txt: {input_txt}')
            # input_ids_with_mask_text = self.tokenizer.decode(data_dict['input_ids'] * attention_mask, skip_special_tokens=True)
            # print(f'input ids with mask: {input_ids_with_mask_text}')

            with torch.no_grad():
                output_ids = model.generate(
                    data_dict['input_ids'].unsqueeze(0).to(self.args.device),
                    attention_mask=attention_mask.unsqueeze(0).to(self.args.device),
                    pad_token_id=tokenizer.eos_token_id,
                    # images=image_tensor,
                    images=image_tensor_bfloat16,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    streamer=None,
                    use_cache=False,
                    # stopping_criteria=[stopping_criteria]
                )
            logits = nested_numpify(output_ids)
            labels = nested_numpify(data_dict['labels'])
            input = nested_numpify(data_dict['input_ids'])
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
            all_inputs = input if all_inputs is None else nested_concat(all_inputs, input, padding_index=-100)

            p_text = tokenizer.decode(output_ids[0, :]).strip()
            # print(f'Generated output: {out}')
            # print(f'Ground truth: {gt}')
            i += 1
            pbar.update(1)
        for i, d in enumerate(data):
            with open(f'output/eval/pred/{i}.json', 'w') as f:
                json.dump(d, f, indent=4)
        metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs))
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics,
                              num_samples=eval_sample_count)

