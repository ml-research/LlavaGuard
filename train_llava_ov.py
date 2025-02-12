import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Sequence

import argparse
import torch
import transformers
from accelerate.utils import set_seed

import llava.train.train as train
from llava.constants import IGNORE_INDEX
from llavaguard.llavaguard_trainer import LlavaGuardTrainer, LlavaGuardTrainerWithMetrics
from train_utils import create_ds_and_check_for_existing_model, preprocess_mpt, preprocess_v1

@dataclass
class DataArguments(train.DataArguments):
    data_path_eval: str = field(default=None,
                                metadata={"help": "Path to the validation data."})



@dataclass
class LlavaGuardDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        if self.tokenizer.model_max_length < input_ids.shape[1]:
            print('Input of shape', input_ids.shape, 'is too long. Truncating to', self.tokenizer.model_max_length)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = train.LazySupervisedDataset(tokenizer=tokenizer,
                                                data_path=data_args.data_path,
                                                data_args=data_args)
        # check the context length of the first sample when tokenized
    # sample = train_dataset[0]
    # input_ids = sample['input_ids']
    # if input_ids.shape[1] > tokenizer.model_max_length:
    #     raise ValueError(f'Input of shape {input_ids.shape} is too long. Truncating to {tokenizer.model_max_length}')
    # else:
    #     print(f'Input of shape {input_ids.shape} is okay. Max tokenizer length {tokenizer.model_max_length}')
    if hasattr(data_args, 'data_path_eval') and data_args.data_path_eval is not None:
        val_dataset = train.LazySupervisedDataset(tokenizer=tokenizer,
                                                  data_path=data_args.data_path_eval,
                                                  data_args=data_args)
    else:
        val_dataset = None
    data_collator = train.LlavaGuardDataCollator(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)

def make_supervised_data_module_llava_next(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = train.LazySupervisedDataset(tokenizer=tokenizer,
                                                data_path=data_args.data_path,
                                                data_args=data_args)
        # check the context length of the first sample when tokenized
    # sample = train_dataset[0]
    # input_ids = sample['input_ids']
    # if input_ids.shape[1] > tokenizer.model_max_length:
    #     raise ValueError(f'Input of shape {input_ids.shape} is too long. Truncating to {tokenizer.model_max_length}')
    # else:
    #     print(f'Input of shape {input_ids.shape} is okay. Max tokenizer length {tokenizer.model_max_length}')
    # raise ValueError('Stop here')
    if hasattr(data_args, 'data_path_eval') and data_args.data_path_eval is not None:
        val_dataset = train.LazySupervisedDataset(tokenizer=tokenizer,
                                                  data_path=data_args.data_path_eval,
                                                  data_args=data_args)
    else:
        val_dataset = None
    data_collator = train.DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)
    

if __name__ == "__main__":
    torch.set_num_threads(40)
    set_seed(42)
    # if model already exists, skip training
    args = sys.argv[1:]
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--output_dir', type=str, default=None)
    parser1.add_argument('--data_path_eval', type=str, default=None)
    parser1.add_argument('--model_name_or_path', type=str, default=None)
    parser1.add_argument('--data_path', type=str, default=None)
    parser1.add_argument('--run_name', type=str, default=None)
    name_space, a2 = parser1.parse_known_args(args)
    output_dir = name_space.output_dir
    data_path_train = name_space.data_path
    model_name = name_space.model_name_or_path.split('/')[-1]
    train_run = name_space.run_name
    if create_ds_and_check_for_existing_model(output_dir, data_path_train, train_run):
        exit(0)
    if 'ov' not in model_name:
        train.make_supervised_data_module = make_supervised_data_module
    else:
        train.make_supervised_data_module = make_supervised_data_module_llava_next
    train.DataArguments = DataArguments
    train.LLaVATrainer = LlavaGuardTrainerWithMetrics
    train.preprocess_mpt = preprocess_mpt
    # train.preprocess_v1 = preprocess_v1
    train.train(attn_implementation="flash_attention_2")
    print('Model saved at:', output_dir)
