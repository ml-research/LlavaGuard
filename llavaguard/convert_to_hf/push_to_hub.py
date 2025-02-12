import os

import torch



import glob

import torch
from safetensors import safe_open
from transformers import (
    AddedToken,
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    SiglipVisionConfig,
)
import gc
import glob
import json
from pathlib import Path

import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
import requests
from PIL import Image
from llava.model.builder import load_pretrained_model
from llavaguard.hf_utils import set_up_env_and_token
from llavaguard_config import local_image_dirs, local_data_dir

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    ".vision_resampler": "",  # all lmms-lab models do avg pooling, so no vision_resampler
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
}

KEYS_TO_MODIFY_MAPPING_LLAVA_NEXT = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "language_model.model.image_newline": "image_newline",
}

def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    return new_state_dict

def convert_state_dict_to_hf_llava_next(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING_LLAVA_NEXT.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value.to(torch.float16)
    return new_state_dict


def load_original_state_dict_llava_next(directory_path):

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def load_image():

    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def convert_llava_next_to_hf(base_model_id, base_model_hf_id, model_dir, hub_id):
    push_to_hub = True


    from transformers import (
        AddedToken,
        AutoConfig,
        AutoTokenizer,
        LlavaNextConfig,
        LlavaNextForConditionalGeneration,
        LlavaNextImageProcessor,
        LlavaNextProcessor,
    )
    pytorch_dump_folder_path = Path(model_dir + '/tmp')

    # load original config
    filepath = hf_hub_download(repo_id=base_model_id, filename="config.json", repo_type="model")
    # read json
    with open(filepath) as f:
        data = json.load(f)
        print(data)

    if base_model_id == "liuhaotian/llava-v1.6-mistral-7b":
        text_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        image_token_index = 32000
    elif base_model_id == "liuhaotian/llava-v1.6-vicuna-7b":
        text_model_id = "lmsys/vicuna-7b-v1.5"
        image_token_index = 32000
    elif base_model_id == "liuhaotian/llava-v1.6-vicuna-13b":
        text_model_id = "lmsys/vicuna-13b-v1.5"
        image_token_index = 32000
    elif base_model_id == "liuhaotian/llava-v1.6-34b":
        text_model_id = "NousResearch/Nous-Hermes-2-Yi-34B"
        image_token_index = 64000
    elif base_model_id == "lmms-lab/llama3-llava-next-8b":
        text_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        image_token_index = 128256
    elif base_model_id == "lmms-lab/llava-next-72b":
        text_model_id = "Qwen/Qwen1.5-72B-Chat"
        image_token_index = 151646
    elif base_model_id == "lmms-lab/llava-next-110b":
        text_model_id = "Qwen/Qwen1.5-110B-Chat"
        image_token_index = 151646

    vision_model_id = data["mm_vision_tower"]

    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    use_fast = False if base_model_id == "liuhaotian/llava-v1.6-34b" else True
    tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=use_fast)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)

    if base_model_id in ("liuhaotian/llava-v1.6-mistral-7b", "lmms-lab/llama3-llava-next-8b"):
        # Mistral-7B doesn't have a padding token set yet
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # image_processor = LlavaNextImageProcessor.from_pretrained(vision_model_id)
    # processor = LlavaNextProcessor(tokenizer=tokenizer, image_processor=image_processor)
    processor = LlavaNextProcessor.from_pretrained(base_model_hf_id)


    config = LlavaNextConfig(
        text_config=text_config.to_dict(),
        image_grid_pinpoints=processor.image_processor.image_grid_pinpoints,
        use_image_newline_parameter=True,
        image_token_index=image_token_index,
    )

    with init_empty_weights():
        model = LlavaNextForConditionalGeneration(config)

    # load original state dict
    state_dict = load_original_state_dict_llava_next(model_dir)
    state_dict = convert_state_dict_to_hf_llava_next(state_dict)
    model.load_state_dict(state_dict, assign=True)
    model.eval()

    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model
    # Pad to 64 for performance reasons
    # Qwen-based models have extra unused space in the vocab size already, so no need to resize
    if base_model_id not in ["lmms-lab/llava-next-72b", "lmms-lab/llava-next-110b"]:
        pad_shape = 64
        vocab_size = config.text_config.vocab_size
        if base_model_id == "liuhaotian/llava-v1.6-34b":
            # this one has 3 additional tokens, namely <|startoftext|>, <|endoftext|> and <image>
            num_tokens = vocab_size + 3
        else:
            # this one has 2 additional tokens, namely <image> and <pad>
            num_tokens = vocab_size + 2
        model.resize_token_embeddings(num_tokens, pad_to_multiple_of=pad_shape)
        model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
            tuple(
                (
                    dist.sample()
                    for _ in range(model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0])
                )
            ),
            dim=0,
        )
        model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
            tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
            dim=0,
        )

    print(f"Saving model and processor for {base_model_id} to {pytorch_dump_folder_path}")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)

    # Make space so we can load the model properly now.
    del state_dict
    gc.collect()

    # Load everything back for inference tests in float32 because prev script was written as that
    # Though it's mostly loaded in fp16 as original weights are in fp16
    model = LlavaNextForConditionalGeneration.from_pretrained(pytorch_dump_folder_path, device_map="auto")
    processor = LlavaNextProcessor.from_pretrained(pytorch_dump_folder_path)
    device = model.device

    # prepare inputs
    image = load_image()
    if base_model_id == "liuhaotian/llava-v1.6-mistral-7b":
        prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    elif base_model_id in ["liuhaotian/llava-v1.6-vicuna-7b", "liuhaotian/llava-v1.6-vicuna-13b"]:
        prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"
    elif base_model_id == "liuhaotian/llava-v1.6-34b":
        prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"
    elif base_model_id == "lmms-lab/llama3-llava-next-8b":
        prompt = "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat is shown in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif base_model_id in ["lmms-lab/llava-next-72b", "lmms-lab/llava-next-110b"]:
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"

    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # verify inputs
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_pixel_values.pt", repo_type="dataset")
    original_pixel_values = torch.load(filepath, map_location="cpu")
    assert torch.allclose(original_pixel_values, inputs.pixel_values.half())

    if base_model_id == "liuhaotian/llava-v1.6-mistral-7b":
        filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_input_ids.pt", repo_type="dataset")
        original_input_ids = torch.load(filepath, map_location="cpu")
        # replace -200 by image_token_index (since we use token ID = 32000 for the image token)
        original_input_ids[original_input_ids == -200] = image_token_index
        assert original_input_ids[0].tolist() == inputs.input_ids[0].tolist()

    elif base_model_id == "liuhaotian/llava-v1.6-34b":
        filepath = hf_hub_download(
            repo_id="nielsr/test-image", filename="llava_1_6_34b_input_ids.pt", repo_type="dataset"
        )
        original_input_ids = torch.load(filepath, map_location="cpu")
        # replace -200 by image_token_index
        original_input_ids[original_input_ids == -200] = image_token_index

        assert original_input_ids[0].tolist() == inputs.input_ids[0].tolist()

    image_sizes = torch.tensor([[899, 1024]])
    assert image_sizes[0].tolist() == inputs.image_sizes[0].tolist()

    # verify single forward pass
    print("Single forward pass")
    with torch.inference_mode():
        inputs = inputs.to(device)
        outputs = model(**inputs)
        print("Shape of logits:", outputs.logits.shape)
        print("First values of logits:", outputs.logits[0, :3, :3])

        if base_model_id == "liuhaotian/llava-v1.6-mistral-7b":
            expected_slice = torch.tensor(
                [[-4.8555, -4.6992, -0.1996], [-10.5703, -10.7344, -2.7246], [-7.0391, -7.3672, -0.2634]],
                dtype=torch.float32,
                device=device,
            )
        elif base_model_id == "liuhaotian/llava-v1.6-vicuna-7b":
            expected_slice = torch.tensor(
                [[1.4883, 0.9976, -0.6992], [-9.7031, -5.7031, -1.5557], [-5.1328, -5.5586, 8.8281]],
                dtype=torch.float32,
                device=device,
            )
        elif base_model_id == "liuhaotian/llava-v1.6-vicuna-13b":
            expected_slice = torch.tensor(
                [[-0.9614, 7.3125, 0.2106], [-7.2695, -8.5469, 3.6211], [-6.3750, -8.1875, 5.4688]],
                dtype=torch.float32,
                device=device,
            )
        elif base_model_id == "liuhaotian/llava-v1.6-34b":
            expected_slice = torch.tensor(
                [[-9.0859, -9.1406, 5.9453], [-5.9570, -5.9766, 2.2754], [-5.7305, -5.7539, 4.0000]],
                dtype=torch.float32,
                device=device,
            )
        elif base_model_id == "lmms-lab/llama3-llava-next-8b":
            expected_slice = torch.tensor(
                [[-3.9648, 1.1396, 3.3145], [-5.3594, -1.5654, -1.9619], [-12.3750, -10.6797, -9.3125]],
                dtype=torch.float32,
                device=device,
            )
        elif base_model_id == "lmms-lab/llava-next-72b":
            # Not yet checked against reference
            expected_slice = torch.tensor(
                [[3.7148, 3.9277, 3.4395], [-0.4341, 1.1387, 6.5117], [3.2324, 3.4688, 4.1133]],
                dtype=torch.float32,
                device=device,
            )
        elif base_model_id == "lmms-lab/llava-next-110b":
            # Not yet checked against reference
            expected_slice = torch.tensor(
                [[-2.5449, -1.6738, -2.0371], [1.0811, 3.4961, 5.0312], [1.7803, 2.5137, 2.4277]],
                dtype=torch.float32,
                device=device,
            )
        else:
            raise ValueError(f"Model {base_model_id} not supported")

        assert torch.allclose(outputs.logits[0, :3, :3], expected_slice, atol=1e-4)
        print("Logits are ok!")

    # verify generation
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
    )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print("Generated text:", repr(generated_text))

    if base_model_id == "liuhaotian/llava-v1.6-mistral-7b":
        expected_text = '[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot that displays data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point.\n\nIn this particular radar chart, there are several axes labeled with different metrics or benchmarks, such as "MMM-Vet," "MMM-Bench," "LLaVA-Bench," "SLED-Bench," "'
    elif base_model_id == "liuhaotian/llava-v1.6-vicuna-7b":
        expected_text = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER:  \nWhat is shown in this image? ASSISTANT: The image appears to be a graphical representation of a benchmarking study comparing the performance of various models or systems. It\'s a scatter plot with a circular layout, where each point represents a different model or system, and the axes represent different metrics or dimensions of comparison.\n\nThe metrics are likely related to machine learning or artificial intelligence performance, as indicated by the terms like "BLIP-2," "Instruct BLIP," "POE," "QWA," "V"""
    elif base_model_id == "liuhaotian/llava-v1.6-vicuna-13b":
        expected_text = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:  \nWhat is shown in this image? ASSISTANT: The image appears to be a radar chart, also known as a spider chart or star chart, which is a graphical method of displaying multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point.\n\nIn this particular radar chart, there are several variables represented:\n\n- MM-Vet\n- LLa-Va-Bench\n- SEED-Bench\n- MM"
    elif base_model_id == "liuhaotian/llava-v1.6-34b":
        expected_text = "<|im_start|> system\nAnswer the questions. <|im_start|> user\n\nWhat is shown in this image? <|im_start|> assistant\nThe image appears to be a radar chart, also known as a spider chart, which is a graphical method of displaying multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point.\n\nIn this particular chart, there are several datasets represented by different colors and labeled with various acronyms such as MM-Vet, LLaVA-Bench, SEED-Bench, MM-Bench-CN, MM-"
    elif base_model_id == "lmms-lab/llama3-llava-next-8b":
        expected_text = 'system\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.user\n\n\nWhat is shown in this image?assistant\n\n\nThe image shows a radar chart, also known as a spider chart or a web chart, which is a type of graph used to display multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point. Each axis represents a different variable, and the values are plotted along each axis and connected to form a polygon.\n\nIn this particular radar chart, there are several axes labeled with different variables, such as "MM-Vet," "LL'
    elif base_model_id == "lmms-lab/llava-next-72b":
        expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image displays a radar chart, also known as a spider chart or a star chart, which is a graphical method of displaying multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point. Each axis represents a different variable, and the value of each variable is represented by the distance from the center of the chart to the point where the axis intersects with the line representing that variable's value.\n\nIn this particular chart, there are several axes"
    elif base_model_id == "lmms-lab/llava-next-110b":
        expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image shows a radar chart comparing the performance of different models on various visual question answering (VQA) benchmarks. Each colored line represents a different model, and the distance from the center of the chart indicates the score or performance level of the model on a particular benchmark. The benchmarks are labeled around the edges of the chart, and include VQA v2, GQA, VizWiz, TextVQA, MMBench-CN, MME, and others. The chart allows for a"
    else:
        raise ValueError(f"Model {base_model_id} not supported")

    assert generated_text == expected_text
    print("Generated text is ok!")

    # verify batched generation
    print("Batched generation...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    cats_image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(
        images=[image, cats_image],
        text=[prompt, prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    for k, v in inputs.items():
        print(k, v.shape)

    print("Image sizes:", inputs.image_sizes)

    # make sure image_sizes are the same
    # as otherwise batched generation doesn't work
    inputs.image_sizes[1] = inputs.image_sizes[0]

    print("Batched generation...")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        use_cache=True,
    )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(outputs)

    if push_to_hub:
        checkpoint_name = base_model_id.split("/")[-1]
        print(f"Pushing to repo {hub_id}")
        model.push_to_hub(hub_id)
        processor.push_to_hub(hub_id)


def convert_llava_llama_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    if "Qwen" not in text_model_id:  # qwen already has a pad token
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    if "Qwen" in text_model_id:
        vision_config = SiglipVisionConfig(
            hidden_size=1152,
            image_size=384,
            intermediate_size=4304,
            num_attention_heads=16,
            num_hidden_layers=26,
            patch_size=14,
            vision_use_head=False,
        ).to_dict()
    else:
        vision_config = None

    config = LlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
    )

    # llms-lab interleeave models do not use any selection startegy except for last hidden state
    if "Qwen" in text_model_id:
        config.image_token_index = 151646
        config.vision_feature_select_strategy = "full"
        config.vision_feature_layer = -1
    else:
        config.pad_token_id = 32001
        config.image_token_index = 32000


    with torch.device("meta"):
        # if '34b' in text_model_id.lower():
        #     from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, LlavaNextImageProcessor
        #     model = LlavaNextForConditionalGeneration(config)
        #     image_processor = LlavaNextImageProcessor.from_pretrained(vision_model_id)
        #     processor = LlavaNextProcessor(tokenizer=tokenizer, image_processor=image_processor)

        # else:
        model = LlavaForConditionalGeneration(config)
        image_processor = AutoImageProcessor.from_pretrained(vision_model_id)
        processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    if "Qwen" in text_model_id:
        state_dict = load_original_state_dict(old_state_dict_id)
    else:
        # state_dict_path = hf_hub_download(local_dir=old_state_dict_id)
        state_dict = torch.load(old_state_dict_id, map_location="cpu")
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, strict=True, assign=True)

    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model and pad to 64 for performance reasons
    pad_shape = 64
    vocab_size = config.text_config.vocab_size
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(
            (dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0]))
        ),
        dim=0,
    )
    model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
        dim=0,
    )
    model.push_to_hub(output_hub_path, private=True)
    processor.push_to_hub(output_hub_path, private=True)


models = [
    # {
    #     'local_path': f'{local_data_dir}/models/LlavaGuard-v1.1-7b/smid_and_crawled_v2_with_augmented_policies/json-v16-arxiv/llava',
    #     'hf_name': 'LukasHug/LlavaGuard-7B-hf',
    #     'lm': 'lmsys/vicuna-7b-v1.5',
    #     'vm': 'openai/clip-vit-large-patch14-336' 
    # },
    # {
    #     'local_path': f'{local_data_dir}/models/LlavaGuard-v1.1-13b/smid_and_crawled_v2_with_augmented_policies/json-v16-arxiv/llava',
    #     'hf_name': 'LukasHug/LlavaGuard-13B-hf',
    #     'lm': 'meta-llama/Llama-2-13b-hf',
    #     'vm': 'openai/clip-vit-large-patch14-336' 
    # },
    # {
    #     'local_path': f'{local_data_dir}/models/LlavaGuard-v1.2-34b/smid_and_crawled_v2_with_augmented_policies/json-v16-arxiv/llava',
    #     'hf_name': 'LukasHug/LlavaGuard-34B-hf',
    #     'lm': "NousResearch/Nous-Hermes-2-Yi-34B",
    #     'base_model_id': 'liuhaotian/llava-v1.6-34b',
    #     'base_model_hf_id': 'llava-hf/llava-v1.6-34b-hf'
    # },
    {
        'local_path': f'{local_data_dir}/models/LlavaGuard-v1.2-7b-ov/v24/llava',
        'hf_name': 'LukasHug/LlavaGuard-v1.2-7B-OV-HF',
        'lm': "NousResearch/Nous-Hermes-2-Yi-34B",
        'base_model_id': 'lmms-lab/llava-onevision-qwen2-7b-ov',
        'base_model_hf_id': 'llava-hf/llava-onevision-qwen2-7b-ov-hf'
    },
    # {
    #     'local_path': f'{local_data_dir}/models/LlavaGuard-v1.1-7b/LlavaGuard-DS/json-v22-oversampled/llava',
    #     'hf_name': 'LukasHug/LlavaGuard-v1.1-7B-hf',
    #     'lm': 'lmsys/vicuna-7b-v1.5',
    #     'vm': 'openai/clip-vit-large-patch14-336' 
    # },
    # {
    #     'local_path': f'{local_data_dir}/models/LlavaGuard-v1.1-13b/LlavaGuard-DS/json-v22-oversampled/llava',
    #     'hf_name': 'LukasHug/LlavaGuard-v1.1-13B-hf',
    #     'lm': 'meta-llama/Llama-2-13b-hf',
    #     'vm': 'openai/clip-vit-large-patch14-336' 
    # },
    # {
    #     'local_path': f'{local_data_dir}/models/LlavaGuard-v1.2-34b/LlavaGuard-DS/json-v22-oversampled/llava',
    #     'hf_name ': 'LukasHug/LlavaGuard-v1.1-34B-hf',
    #     'lm': "NousResearch/Nous-Hermes-2-Yi-34B",
    #     'vm': 'llava-hf/llava-v1.6-34b-hf'
    # }
]
set_up_env_and_token(read=False, write=True)
for settings in models:
    model_path = settings['local_path']    
    model_base = None
    model_name = settings['hf_name'].split('/')[1]
    state_dict_path = model_path.split('/llava')[0] + '/tmp/model_state_dict.bin'
    if '34b' in model_name.lower():
        convert_llava_next_to_hf(base_model_id=settings['base_model_id'],base_model_hf_id=settings['base_model_hf_id'],
                                  model_dir=settings['local_path'], hub_id=settings['hf_name'])
    else:
        if not os.path.exists(state_dict_path):
            tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                                device='cpu', device_map='cpu')
            model.get_vision_tower().load_model()
            os.makedirs(os.path.dirname(state_dict_path), exist_ok=True)
            print(f'saving model to {state_dict_path}')
            torch.save(model.state_dict(), state_dict_path)
        else:
            print(f'Using existing state dict at {state_dict_path}')
        convert_llava_llama_to_hf(text_model_id=settings['lm'], vision_model_id=settings['vm'], output_hub_path=settings['hf_name'], old_state_dict_id=state_dict_path)

