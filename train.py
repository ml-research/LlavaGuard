import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import argparse
import torch
import transformers
from accelerate.utils import set_seed

import llava.train.train as train
from llava.mm_utils import tokenizer_image_token
from prepare_data import prepare_instruct_tuning_with_policy_augmentation
from llava.constants import IGNORE_INDEX
from llavaguard.llavaguard_trainer import LlavaGuardTrainer
from trl import DataCollatorForCompletionOnlyLM
from llava import conversation as conversation_lib
import tokenizers

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_path_eval: str = field(default=None,
                                metadata={"help": "Path to the validation data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len + 1
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

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
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            # attention_mask=~labels.ne(IGNORE_INDEX),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


@dataclass
class LlavaGuardDataCollator2(DataCollatorForCompletionOnlyLM):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        conv = conversation_lib.default_conversation.copy()
        response_template = conv.roles[1]
        super().__init__(tokenizer=tokenizer, response_template=response_template)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # instance
        batch = super().__call__(instances)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch['input_ids'],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(batch['labels'],
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        attention_mask = torch.nn.utils.rnn.pad_sequence(batch['attention_mask'],
                                                         batch_first=True,
                                                         padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        # print input_ids.shape, labels.shape, attention_mask.shape
        print(input_ids.shape, labels.shape, attention_mask.shape)
        attention_mask = attention_mask[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
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
    if hasattr(data_args, 'data_path_eval') and data_args.data_path_eval is not None:
        val_dataset = train.LazySupervisedDataset(tokenizer=tokenizer,
                                                  data_path=data_args.data_path_eval,
                                                  data_args=data_args)
    else:
        val_dataset = None
    # check whether train and test dataset are the have image tokens or not
    sample = train_dataset[0]
    # if 'image' in sample:
    #     print('Image found in the dataset')
    # else:
    #     raise ValueError('Image not found in the dataset')
    # # check for image token (-200) in input_ids
    # if -200 in sample['input_ids']:
    #     print('Image token found in input_ids')
    # else:
    #     raise ValueError('Image token not found in input_ids')
    # data_collator = LlavaGuardDataCollator(tokenizer=tokenizer)
    data_collator = train.DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)


if __name__ == "__main__":
    set_seed(42)
    # if model already exists, skip training
    args = sys.argv[1:]
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--output_dir', type=str, default=None)
    parser1.add_argument('--data_path_eval', type=str, default=None)
    parser1.add_argument('--model_name_or_path', type=str, default=None)
    parser1.add_argument('--data_path', type=str, default=None)
    name_space, a2 = parser1.parse_known_args(args)
    output_dir = name_space.output_dir
    data_path_train = name_space.data_path
    model_name = name_space.model_name_or_path.split('/')[-1]
    with_augmented_policies = 'augmented' in output_dir
    llavaguard = output_dir.split('/')[-3]
    template_version = output_dir.split('/')[-1]
    ds_version = output_dir.split('/')[-2]

    prepare_instruct_tuning_with_policy_augmentation(template_version, False, with_augmented_policies)

    # if we find a trained model in the path then we skip training
    if os.path.exists(output_dir + '/trainer_state.json'):
        print(
            f'Model {output_dir} already exists. Skipping training. If you want to retrain, delete the model directory.')
        print('######################################################################################################')
        sys.exit(0)
    print(f'''
Start Training {llavaguard}
Base model: {model_name}, Dataset version: {ds_version}, Template version: {template_version}
Train data path: {data_path_train}
Output directory: {output_dir}
######################################################################################################
    ''')
    # add the template version to the command line arguments
    sys.argv.append('--run_name')
    sys.argv.append(f'{llavaguard}_{template_version}')
    # sys.argv.append('--tags')
    # sys.argv.append(f'{ds_version}_{template_version}')
    train.make_supervised_data_module = make_supervised_data_module
    train.DataArguments = DataArguments
    train.LLaVATrainer = LlavaGuardTrainer
    train.preprocess_mpt = preprocess_mpt
    os.makedirs(output_dir, exist_ok=True)
    # safe all parser arguments in the output directory for future reference
    with open(f'{output_dir}/params.txt', 'w+') as f:
        params = ' '.join(sys.argv)
        f.write(params.replace('--', '\n'))

    train.train(attn_implementation="flash_attention_2")
    #
    # lora_dir = None if 'lora' not in output_dir else output_dir
    # model_base = model_base if 'lora' not in output_dir else output_dir
    # # eval LlavaGuard after training
    # eval_llavaguard.evaluation(lora_dir=lora_dir, model_base=model_base,
    #                            data_path_eval=data_path_eval,
    #                            data_path_train=None,
    #                            copy_images=False, replace_existing_output=True)
