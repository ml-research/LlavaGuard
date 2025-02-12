import os
import sys
from datetime import datetime
import torch
from llavaguard.data.build_dataset import prepare_instruct_tuning_with_policy_augmentation
from llava.mm_utils import tokenizer_image_token
from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX
from packaging import version
import tokenizers
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
import transformers
from typing import Dict, Optional, Sequence

def get_rank():
    """Get process rank, checking both torch.distributed and env vars."""
    # Try torch.distributed first
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except:
        pass
    
    # Fallback to environment variables
    rank = int(os.environ.get('RANK', '-1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    return local_rank if rank == -1 else rank

def create_ds_and_check_for_existing_model(output_dir, data_path_train, train_run):
    version = output_dir.split('/')[-1]
    pretaining = 'Pretraining' if 'pretrain' in output_dir else 'Fine-tuning'

    # only if file not exists and local rank is 0
    if not os.path.isfile(data_path_train) and get_rank() == 0:
        print(f'Dataset not found at {data_path_train}. Preparing dataset...')
        ds_root = data_path_train.split('/' + version)[0]
        prepare_instruct_tuning_with_policy_augmentation(int(version.replace('v', '')), ds_root)
    else:
        if get_rank() == 0:
            print(f'Using existing data at {data_path_train}')

    # if we find a trained model in the path then we skip training
    if os.path.exists(output_dir + '/trainer_state.json') or os.path.exists(output_dir + '/llava/trainer_state.json'):
        print(
            f'Model {output_dir} already exists. Skipping training. If you want to retrain, delete the model directory.')
        print('######################################################################################################')
        return True
    if get_rank() == 0:
        print(f'''
Start {train_run}
Training: {pretaining}, Dataset version: {version}
Train data path: {data_path_train}
Output directory: {output_dir}
######################################################################################################
        ''')
    os.makedirs(output_dir, exist_ok=True)
    # safe all parser arguments in the output directory for future reference
    with open(f'{output_dir}/params.txt', 'w+') as f:
        params = ' '.join(sys.argv)
        params = params.replace('--', '\n')
        params += f'\nEval date: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
        f.write(params)
    return False

# the fixed version of preprocess_mpt and preprocess_v1
def preprocess_mpt(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    # use the first source as the policy prompt and the rest as the conversation
    # policy_prompt = sources[0][0]["value"]
    # system_prompt = policy_prompt.replace("<image>", "")
    # sources[0][0]["value"] = '<image>'
    # conv = Conversation(
    #     system=system_prompt,
    #     roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    #     version="mpt",
    #     messages=(),
    #     offset=0,
    #     sep_style=SeparatorStyle.MPT,
    #     sep="<|im_end|>",
    # )

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
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
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
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
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

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

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


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    # policy_prompt = sources[0][0]["value"]
    # system_prompt = policy_prompt.replace("<image>", "")
    # sources[0][0]["value"] = '<image>'
    # conv = Conversation(
    #     system=system_prompt,
    #     roles=("USER", "ASSISTANT"),
    #     version="v1",
    #     messages=(),
    #     offset=0,
    #     sep_style=SeparatorStyle.TWO,
    #     sep=" ",
    #     sep2="</s>",
    # )

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
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
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