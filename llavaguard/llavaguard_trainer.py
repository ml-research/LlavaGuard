import json
import math
import os
import time
from typing import Dict

from torch.utils.data import Subset
from tqdm import tqdm
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.trainer_utils import speed_metrics, EvalLoopOutput, EvalPrediction

from llava.train.llava_trainer import LLaVATrainer
from llava.train.train import LazySupervisedDataset
from llavaguard.callbacks import LlavaGuardCallback
import torch

from llavaguard.evaluation_metrics_calculator import parse_json, EvaluationMetricsCalculator


class LlavaGuardTrainer(LLaVATrainer):
    def __init__(self, *args, **kwargs):
        # add callback to kwargs
        kwargs['callbacks'] = [LlavaGuardCallback]
        super().__init__(*args, **kwargs)


class LlavaGuardTrainerWithMetrics(LLaVATrainer):
    def __init__(self, *args, **kwargs):
        # add callback to kwargs
        kwargs['callbacks'] = [LlavaGuardCallback]
        kwargs['compute_metrics'] = self.compute_metrics_llavaguard
        super().__init__(*args, **kwargs)

    def compute_metrics_llavaguard(self, EvalPrediction):
        label_ids = EvalPrediction.label_ids
        predictions = EvalPrediction.predictions
        # all_inputs = EvalPrediction.inputs
        emc = EvaluationMetricsCalculator(pred_dir='output/eval/tmp/pred', debug=False)

        for i, (prediction, label) in enumerate(zip(predictions, label_ids)):
            label[label == 0] = 2  # convert output id 0 to 2 (eos_token_id)
            label[label == -100] = 1  # convert improper tokens to ''
            gt = json.loads(self.tokenizer.decode(label, skip_special_tokens=True))
            # input[input == -100] = 1  # convert improper tokens to ''
            # input[input == 0] = 2  # convert output id 0 to 2 (eos_token_id)
            # print(f'input {input}')
            # aaa = self.tokenizer.decode(input, skip_special_tokens=True)
            # print(f'input {aaa}')
            p_text = self.tokenizer.decode(prediction.argmax(-1), skip_special_tokens=True)
            emc.add_sample(i, p_text, gt, save_output=True)
        metrics, _ = emc.compute_stats()
        return {
            'Overall Accuracy': metrics['Overall Accuracy'],
            'Balanced Accuracy': metrics['Balanced Accuracy'],
            "compliant_hit_rate": metrics["compliant_hit_rate"],
            "review_needed_hit_rate": metrics["review_needed_hit_rate"],
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'TN': metrics['TN'],
            'FN': metrics['FN'],
            'Invalid': metrics['Invalid'],
        }

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
        eval_sample_count = len(eval_dataset) if len(eval_dataset) < 50 else 50
        # create subset of eval dataset
        subset = Subset(eval_dataset, range(eval_sample_count))
        eval_dataloader = self.get_eval_dataloader(subset)
        # print(self.data_collator.__name__)
        # only evaluate 20 samples
        # get first element of eval dataloader
        # s = next(iter(eval_dataloader))
        # # s = subset[0]
        # print(f'test: {s}')
        # print(f'input ids shape: {s["input_ids"].shape}')
        # print(f'labels shape: {s["labels"].shape}')
        # print(f'attention mask shape: {s["attention_mask"].shape}')
        # print(f'image shape: {s["images"].shape}')
        # first_input_ids = s["input_ids"][0]
        # first_labels = s["labels"][0]
        # first_labels[first_labels == -100] = 1  # convert improper tokens to ''
        # first_input_ids[first_input_ids == -100] = 1  # convert improper tokens to ''
        # # count number 0f 200
        # a = first_input_ids[first_input_ids == -200]
        # print(f'200 count: {len(a)}')
        # first_attention_mask = s["attention_mask"][0]
        # print(f'attention mask: {first_attention_mask}')
        # first_image = s["images"][0]
        # # apply attention mask to input ids
        # # first_input_ids = first_input_ids[first_attention_mask]
        # first_input_ids[first_input_ids == -200] = 1  # convert improper tokens to ''
        # input_txt = self.tokenizer.decode(first_input_ids, skip_special_tokens=True)
        # print(f'input txt: {input_txt}')
        # label_txt = self.tokenizer.decode(first_labels, skip_special_tokens=True)
        # print(f'label txt: {label_txt}')

        start_time = time.time()
        # self.args.include_inputs_for_metrics = True

        # output = self.eval_loop_llavaguard_v2(eval_dataloader)

        output = self.evaluation_loop(
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

    def eval_loop_llavaguard_v2(self, eval_dataloader):
        num_sampels = len(eval_dataloader) if len(
            eval_dataloader) < 50 else 1  # limit to 50 samples for each evaluation run
        model = self.model
        TP, FP, TN, FN, P, N, invalid_assessments = [], [], [], [], [], [], []
        pbar = tqdm(eval_dataloader, desc="Evaluating")
        all_preds = None
        all_labels = None
        all_inputs = None
        model.eval()
        for inputs in pbar:
            pbar.set_description(
                f'Evaluation: (TP {len(TP)}, FP {len(FP)}, TN {len(TN)}, FN {len(FN)}, Invalid {len(invalid_assessments)})')
            loss, logits, labels = self.prediction_step(model, inputs, False, ignore_keys=None)
            # inputs = model.prepare_inputs_for_generation(inputs['input_ids'])

            # logits = model.forward(
            #     input_ids=inputs['input_ids'],
            #     attention_mask=inputs['attention_mask'],
            #     labels=inputs['labels'],
            #     images=inputs['images']
            # )
            # print(f'logits: {logits}')

            logits = nested_numpify(logits)
            labels = nested_numpify(inputs['labels'])
            input = nested_numpify(inputs['input_ids'])
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
            all_inputs = input if all_inputs is None else nested_concat(all_inputs, input, padding_index=-100)
        metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs))
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_sampels)

    def eval_loop_llavaguard(self, eval_dataset):
        eval_sample_count = len(eval_dataset) if len(
            eval_dataset) < 20 else 20  # limit to 50 samples for each evaluation run
        model = self.model
        tokenizer = self.tokenizer
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        list_data_dict = eval_dataset.list_data_dict
        TP, FP, TN, FN, P, N, invalid_assessments = [], [], [], [], [], [], []
        pbar = tqdm(range(eval_sample_count), desc="Evaluating")
        all_preds = None
        all_labels = None
        all_inputs = None
        data = []
        model.eval()
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
        i = 0
        gt = json.loads(list_data_dict[i]['conversations'][1]["value"]).keys()
        final_assessment_key = 'final-assessment' if 'final-assessment' in list(gt) else 'decision'
        while pbar.n < eval_sample_count and i < len(eval_dataset):
            pbar.set_description(
                f'Evaluation: (TP {len(TP)}, FP {len(FP)}, TN {len(TN)}, FN {len(FN)}, Invalid {len(invalid_assessments)})')

            data_dict = eval_dataset[i]
            gt = json.loads(list_data_dict[i]['conversations'][1]["value"])
            if gt[final_assessment_key] == 'Compliant' and len(P) >= eval_sample_count / 2 or gt[
                final_assessment_key] == 'Review Needed' and len(N) >= eval_sample_count / 2:
                continue

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

            if gt[final_assessment_key] == 'Compliant':
                P.append(i)
            else:
                N.append(i)

            try:
                out = json.loads(parse_json(p_text))
                jj = {
                    'prediction_json': out,
                    'ground_truth': gt,
                    'predict_raw': p_text,
                }
                data.append(jj)
                # print(f'Parsed output: {out}')
                if out[final_assessment_key] == 'Compliant' and gt[final_assessment_key] == 'Compliant':
                    TP.append(i)
                elif out[final_assessment_key] == 'Compliant' and gt[final_assessment_key] == 'Review Needed':
                    FP.append(i)
                elif out[final_assessment_key] == 'Review Needed' and gt[final_assessment_key] == 'Compliant':
                    FN.append(i)
                elif out[final_assessment_key] == 'Review Needed' and gt[final_assessment_key] == 'Review Needed':
                    TN.append(i)
                else:
                    invalid_assessments.append([i])
            except:
                invalid_assessments.append([i])
                jj = {
                    'ground_truth': gt,
                    'predict_raw': p_text,
                }
                data.append(jj)
                pass
            i += 1
            pbar.update(1)
        for i, d in enumerate(data):
            with open(f'output/eval/pred/{i}.json', 'w') as f:
                json.dump(d, f, indent=4)
        metrics = compute_metric(TP, FP, TN, FN, P, N, invalid_assessments, eval_sample_count)
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics,
                              num_samples=eval_sample_count)


def compute_metric(TP, FP, TN, FN, P, N, invalid_assessments, num_samples):
    acc = round((len(TP) + len(TN)) / num_samples, 4)
    bal_acc = round((len(TP) / (len(P)) + len(TN) / (len(N))) / 2, 4) if len(P) > 0 and len(N) > 0 else 0
    compliant_hit_rate = round(len(TP) / len(P), 4) if len(P) > 0 else 0
    review_needed_hit_rate = round(len(TN) / len(N), 4) if len(N) > 0 else 0
    compliant_samples_percent = round(len(P) / num_samples * 100, 2)
    review_needed_samples_percent = round(len(N) / num_samples * 100, 2)
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "compliant_hit_rate": compliant_hit_rate,
        "review_needed_hit_rate": review_needed_hit_rate,
        "TP": len(TP),
        "FP": len(FP),
        "TN": len(TN),
        "FN": len(FN),
        "invalid_assessments": len(invalid_assessments)
    }
    out_txt = f'################# Evaluation Results #################\n' \
              f'Eval distribution Total: {len(P) + len(N)}, Compliant Samples: {len(P)} ({compliant_samples_percent}%),' \
              f' Review Needed Samples: {len(N)} ({review_needed_samples_percent}%)\n' \
              f'Overall Accuracy: {round(acc * 100, 2)}%, Balanced Accuracy: {round(bal_acc * 100, 2)}%, ' \
              f'Compliant hit rate: {round(compliant_hit_rate * 100, 2)}%, Review Needed hit rate: {round(review_needed_hit_rate * 100, 2)}%\n' \
              f'Confusion Matrix (True=Compliant, False=Review Needed): ' \
              f'TP: {len(TP)}, FP: {len(FP)}, TN: {len(TN)}, FN: {len(FN)}, Invalid json: {len(invalid_assessments)}' \
              f'invalid assessments: {invalid_assessments}\n'

    # save to file
    os.makedirs('output/eval/pred', exist_ok=True)
    with open('output/eval/metrics.txt', 'w') as f2:
        f2.write(out_txt)
    return metrics
