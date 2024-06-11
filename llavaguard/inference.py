import json
import os
import time
from io import BytesIO

import requests
import rtpt
import torch
from PIL import Image
from torch.nn.functional import pad
from tqdm import tqdm
from transformers import TextStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    IGNORE_INDEX
from llava.conversation import SeparatorStyle
from llava.eval.run_llava import load_images
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images
from llavaguard.evaluation_metrics_calculator import EvaluationMetricsCalculator


def clear_conv(conv):
    conv.messages = []
    return conv


def load_image(image_processor, image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    return image, image_tensor


def run_v0(model, tokenizer, image, conv, image_tensor, text=None, verbose=True):
    if text is not None:
        inp = text
    else:
        try:
            inp = input(f"{conv.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")

    if image is not None:
        conv = clear_conv(conv)
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # remove <image> token
    prompt = prompt.replace('<image>', '')

    if verbose:
        print(f"{conv.roles[1]}: ", end="")
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    attention_mask = input_ids.ne(0)
    # print image tensor size
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            num_beams=2,
            top_p=0.95,
            top_k=50,
            max_new_tokens=1024,
            streamer=streamer if verbose else None,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    outputs = tokenizer.decode(output_ids[0, :]).strip()
    conv.messages[-1][-1] = outputs
    return outputs


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


def run_llava_not_batched(model, tokenizer, emc, image_processor, prompts, gts, ids, im_paths, conv):
    rt = rtpt.RTPT(name_initials='LH', experiment_name=f'LlavaGuard-Eval', max_iterations=len(prompts) + 1)
    rt.start()
    pbar = tqdm(zip(prompts, gts, ids, im_paths), total=len(prompts))
    print('Running forward without batching')
    for prompt, gt, sample_id, im_path in pbar:
        metrics = emc.get_metrics()
        pbar.set_description(
            f'Evaluating TP: {metrics["TP"]}, FP: {metrics["FP"]}, TN: {metrics["TN"]}, FN: {metrics["FN"]}, '
            f'Invalid: {metrics["invalid_assessments"]}')
        out = run_llava(model, tokenizer, image_processor, prompt, im_path, conv)
        emc.add_sample(sample_id, out, gt, prompt, save_output=True)
        rt.step()


def run_llava_batched(model, tokenizer, emc, image_processor, prompts, gts, ids, im_paths, conv, batch_size=1):
    def batched_forward(b_prompts, b_im_paths, conv_):
        images = load_images(b_im_paths)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        )
        if isinstance(images_tensor, list):
            images_tensor = torch.cat(images_tensor, dim=0)
        images_tensor = images_tensor.to(model.device, dtype=torch.float16)
        input_ids = []
        max_len = 0
        for p in b_prompts:
            conv_ = clear_conv(conv_)
            conv_.append_message(conv_.roles[0], p)
            conv_.append_message(conv_.roles[1], None)
            prompt = conv_.get_prompt()
            # prompt += '<unk>'
            ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .cuda()
            )
            input_ids.append(ids)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        # input_ids_test = None
        # for i_ids in input_ids:
        #     i_tensor = pad(i_ids.unsqueeze(0), (0, max_len - i_ids.shape[0]), value=tokenizer.pad_token_id)
        #     # pad tensor to max_len
        #     if input_ids_test is None:
        #         input_ids_test = i_tensor
        #     else:
        #         input_ids_test = torch.cat([input_ids_test, i_tensor], dim=0)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
                image_sizes=image_sizes,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                num_beams=2,
                max_new_tokens=200,
                use_cache=True,
                stopping_criteria=[KeywordsStoppingCriteria(['}'], tokenizer, input_ids)]

            )

        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    print('Running batched forward with batch size:', batch_size)
    rt = rtpt.RTPT(name_initials='LH', experiment_name=f'LlavaGuard-Eval',
                   max_iterations=len(prompts) // batch_size + 1)
    rt.start()
    pbar = tqdm(range(0, len(prompts), batch_size))
    for i in pbar:
        metrics = emc.get_metrics()
        pbar.set_description(
            f'Evaluating TP: {metrics["TP"]}, FP: {metrics["FP"]}, TN: {metrics["TN"]}, FN: {metrics["FN"]}, '
            f'Invalid: {metrics["invalid_assessments"]}')
        batch_prompts = prompts[i:i + batch_size]
        batch_im_paths = im_paths[i:i + batch_size]
        batch_gts = gts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        # run llava batched forward
        outs = batched_forward(batch_prompts, batch_im_paths, conv)
        batch = [{'prediction': out, 'gt': gt, 'id': sample_id, 'prompt': prompt} for out, gt, sample_id, prompt in zip(
            outs, batch_gts, batch_ids, prompts)]
        emc.add_batch(batch, save_output=True)
        rt.step()
    return emc


def run_llava(model, tokenizer, image_processor, prompt, im_path, conv):
    # stop time
    conv = clear_conv(conv)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image_, image_tensor_ = load_image(image_processor, im_path)
    image_sizes = [image_.size]
    images_tensor = image_tensor_.unsqueeze(0).cuda()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            num_beams=2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[KeywordsStoppingCriteria(['}'], tokenizer, input_ids)]

        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # it = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

    # print(f'input length: {it.shape[0]}')
    return outputs[0].strip()
