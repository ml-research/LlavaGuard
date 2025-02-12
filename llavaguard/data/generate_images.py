import argparse
import os

import numpy as np
from PIL import Image
from datasets import load_dataset
import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline, Transformer2DModel, PixArtSigmaPipeline
from rtpt import rtpt
from llavaguard_config import local_image_dirs, local_data_dir


def dummy(images, **kwargs):
    return images, False


def is_uniform(image_path):
    # Open the image
    im = Image.open(image_path)
    # Convert the image to grayscale
    im = im.convert('L')
    # Get all pixel values
    pixels = list(im.getdata())
    # Check if 50% or more of the pixels are the same value
    pixels = np.array(pixels)
    if pixels.std() < 1:
        return True


def generate_sd3(prompts, output_dir, rt):
    print('Generating sd3 images')
    os.makedirs(f"{output_dir}/im", exist_ok=True)
    os.makedirs(f"{output_dir}/prompts", exist_ok=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
    )
    pipe.safety_checker = dummy
    pipe = pipe.to(device)
    for i, prompt in enumerate(prompts):
        # if file exists and is not empty, skip
        if os.path.exists(f"{output_dir}/im/{i}.png") and not is_uniform(f"{output_dir}/im/{i}.png"):
            rt.step()
            continue
        image = pipe(prompt).images[0]
        image.save(f"{output_dir}/im/{i}.png")
        with open(f"{output_dir}/prompts/{i}.txt", 'w') as f:
            f.write(prompt)
        rt.step()


def generate_sd(prompts, output_dir, rt):
    print('Generating sd images')
    os.makedirs(f"{output_dir}/im", exist_ok=True)
    os.makedirs(f"{output_dir}/prompts", exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    pipe = pipe.to(device)
    for i, prompt in enumerate(prompts):
        if os.path.exists(f"{output_dir}/im/{i}.png") and not is_uniform(f"{output_dir}/im/{i}.png"):
            rt.step()

            continue
        image = pipe(prompt).images[0]
        image.save(f"{output_dir}/im/{i}.png")
        with open(f"{output_dir}/prompts/{i}.txt", 'w') as f:
            f.write(prompt)
        rt.step()


def generate_pixart(prompts, output_dir, rt):
    print('Generating pixart images')
    weight_dtype = torch.float16
    os.makedirs(f"{output_dir}/im", exist_ok=True)
    os.makedirs(f"{output_dir}/prompts", exist_ok=True)

    transformer = Transformer2DModel.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        subfolder='transformer',
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        transformer=transformer,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    pipe.safety_checker = dummy
    pipe.to(device)
    for i, prompt in enumerate(prompts):
        if os.path.exists(f"{output_dir}/im/{i}.png") and not is_uniform(f"{output_dir}/im/{i}.png"):
            rt.step()
            continue
        image = pipe(prompt).images[0]
        image.save(f"{output_dir}/im/{i}.png")
        with open(f"{output_dir}/prompts/{i}.txt", 'w') as f:
            f.write(prompt)
        rt.step()

    # Enable memory optimizations.
    # pipe.enable_model_cpu_offload()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    if os.path.exists(f'{local_data_dir}/data'):
        output_dir = f'{local_data_dir}/data/GenAI-DS'
    else:
        output_dir = '/workspace/output/generated_ds/'

    ds_name = 'LG-Anonym/i2p'
    ds_hold_out = 'LG-Anonym/i2p-adversarial-split'
    # load hugface data
    ds = load_dataset(ds_name, split='train')
    ds_hold_out = load_dataset(ds_hold_out, split='train')
    ds_hold_out_prompts = ds_hold_out['prompt']

    # remve hold out set from training data
    ds = ds.filter(lambda x: x['prompt'] not in ds_hold_out_prompts)
    ds = ds.filter(lambda x: x['hard'] == 1)
    print(f'A total of {len(ds)} prompts are available for generating images')
    ds_spli1 = ds['prompt']
    ds_spli2 = ds['prompt']
    ds_spli3 = ds['prompt']
    rt = rtpt.RTPT(name_initials='LH', experiment_name=f'Generate-LG_DS', max_iterations=len(ds_spli1) * 3)
    rt.start()
    generate_sd3(ds_spli1, output_dir + '/sd3', rt)
    generate_pixart(ds_spli2, output_dir + '/pixart', rt)
    generate_sd(ds_spli3, output_dir + '/sd', rt)
