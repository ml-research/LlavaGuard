import asyncio
import glob
import signal
import psutil
import os
import torch
from tqdm.asyncio import tqdm
import json
from llavaguard.evaluation.eval_utils import get_conv, get_model_name_from_path
import json
import os

import openai
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)
import base64
import glob
import os
import sys
import random

if '/workspace' not in sys.path:
    sys.path.append('/llava/LLaVA-NeXT')


def kill_child_process(pid=None, include_cls=False, skip_pid=None):
    """Kill the process and all its children process."""
    if pid is None:
        pid = os.getpid()

    try:
        itcls = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = itcls.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_cls:
        try:
            itcls.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itcls.send_signal(signal.SIGINT)
        except psutil.NoSuchProcess:
            pass

def encode_image(p):
    if 'www.' in p:
        return p
    with open(p, "rb") as image_file:
        return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"


def prepare_model_for_sglang(model_dir: str):
    def prepare_local_llava_model():
        # check if model exists and if a file with safetensors exists
        if not os.path.exists(model_dir) or len(glob.glob(f'{model_dir}/*.safetensors')) == 0:
            raise Exception('Failed to prepare model for SGLang. Model does not exist at: ', model_dir)
        # move all files to llava folder
        os.makedirs(sglang_dir)
        try:
            os.system(f'mv {model_dir}/* {sglang_dir}/')
        except Exception as e:
            pass
        # replace config file
        root = os.path.abspath("").split('LlavaGuard')[0]
        if 'LlavaGuard-v1.2-7b-ov' != model_name:
            os.system(f'cp {root}/llavaguard/llava-configs/{model_name}.json {sglang_dir}/config.json')
        else:
            os.system(f'cp {root}/llavaguard/llava-configs/{model_name}_preprocessor_config.json {sglang_dir}/preprocessor_config.json')
            vocab_size = 152064
            # load_config
            with open(f'{sglang_dir}/config.json', 'r') as f:
                config = json.load(f)
            config['vocab_size'] = vocab_size
            with open(f'{sglang_dir}/config.json', 'w') as f:
                json.dump(config, f)
    
    if os.path.exists(model_dir):
        model_name = model_dir.split('models/')[-1].split('/')[0]
        sglang_dir = model_dir
    elif 'AIML' in model_dir:
        model_name, sglang_dir = model_dir.split('/')[-1], model_dir
    else:
        raise Exception(f'Model not found! {model_dir}')
    
    if 'LlavaGuard-v' in model_dir and 'AIML' not in model_dir:
        if os.path.exists(model_dir):
            # prepare server command
            sglang_dir = f'{model_dir}/llava'
            if os.path.exists(sglang_dir) and len(glob.glob(f'{sglang_dir}/*.safetensors')) > 0:
                print(f'Model {model_name} already prepared for sglang.')
            else:
                prepare_local_llava_model()
        else:
            raise Exception(f'Model not found! {model_dir}')


    # elif os.path.exists(model_dir):
    #     model_name = model_dir.split('/')[-1]
    #     sglang_dir = model_dir
    #     tokenizer = model_dir

        
    # if you want to use different tokenizer than the one provided in the model directory/hub, you can specify it here
    sglang_tokenizers = {
        'LlavaGuard-v1.1-7b': 'llava-hf/llava-1.5-7b-hf',
        'LlavaGuard-v1.1-13b': 'llava-hf/llava-1.5-13b-hf',
        'LlavaGuard-v1.2-13b': 'llava-hf/llava-v1.6-vicuna-13b-hf',
        'LlavaGuard-v1.2-34b': 'liuhaotian/llava-v1.6-34b-tokenizer',
        'LlavaGuard-v1.2-7b-ov': None,
        'LlavaGuard-v1.2-8b': 'lmms-lab/llama3-llava-next-8b',
        'llava-v1.5-7b': 'llava-hf/llava-1.5-7b-hf',
        'llava-v1.5-13b': 'llava-hf/llava-1.5-13b-hf',
        'v1.6-vicuna-7b': 'llava-hf/llava-v1.6-vicuna-7b-hf',
        'v1.6-vicuna-13b': 'llava-hf/llava-v1.6-vicuna-13b-hf',
        'llava-v1.6-34b': 'liuhaotian/llava-v1.6-34b-tokenizer',
        'llama3-llava-next-8b': None,
        'llava-onevision-qwen2-7b-ov': None,
        'Llama-Guard-3-11B-Vision': None,
        }

    try:
        tokenizer = sglang_tokenizers[model_name]
    except KeyError:
        print(f"Tokenizer not found for model {model_name} using default tokenizer.")
        tokenizer = None
    print(f'Model {model_name} prepared for sglang! Ready to evaluate.')
    
    if '34b' in model_name or 'ov' in model_name.lower():
            chat_template = "chatml-llava"
    elif 'v1.5' in model_name:
        chat_template = 'vicuna_v1.1'
    elif '8b' in model_name:
        chat_template = 'llava_llama_3'
    elif 'qwen2-vl' in model_name.lower() or 'QwenGuard' in model_name:
        chat_template = 'qwen2-vl'
    # elif 'Llama-3.2-11B-Vision' in model_name or 'Llama-Guard-3' in model_name:
    elif 'Llama-3.2-11B-Vision' in model_name or 'Llama-3.2-90B-Vision-Instruct' in model_name:
        chat_template = 'llama_3_vision'
    else:
        raise Exception(f'Chat template not found for model {model_name} please specify chat template. Model dir: {model_dir}')
    return sglang_dir, tokenizer, chat_template



class SGLangServer:
    def __init__(cls, model_dir: str, device, HF_HOME=None, quantized=False):
        import pkg_resources
        package, version = 'sglang', '0.4.0'
        installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        cls.sglang_version = pkg_resources.parse_version(installed[package])        
        print(f' Sglang version: {cls.sglang_version}')
        print(f"Evaluating model: {model_dir}")        
        model, tokenizer, chat_template = prepare_model_for_sglang(model_dir)
        model_name = get_model_name_from_path(model_dir)
        # Set the environment variable
        env = os.environ.copy()
        device = str(device)
        if device == 'cuda':
            number_of_devices = torch.cuda.device_count()
            d_ts = torch.zeros(1).to(f'cuda')
        else:
            env["CUDA_VISIBLE_DEVICES"] = device
            number_of_devices = device.count(',') + 1
            d_ts = torch.zeros(1).to(f'cuda')
            # d_ts = [torch.zeros(1).to(f'cuda:{d}') for d in device.split(',')]
        if '0.5' in model_name:
            number_of_devices = 2 if number_of_devices >=2 else 1
    
        if HF_HOME is not None:
            env["HF_HOME"] = HF_HOME
            print('HF_HOME set to ', HF_HOME)
        random.seed(None)
        port = random.randint(10000, 20000)
        cls.default_hyperparameters = {
            'temperature': 0.2, 
            'top_p': 0.95,
            # 'top_k': 50,
            'max_tokens': 500,
            # 'stop': "}",
        }
        if 'qwenguard' in model_name.lower():
            cls.default_hyperparameters['temperature'] = 0.01
            cls.default_hyperparameters['top_p'] = 0.001
        
        cls.base_url = f'http://127.0.0.1:{port}'
        cls.api_key = "sk-123456"
        cls.model = model
        cls.model_name = model_name
        cls.chat_template = chat_template
        other_args = ['--tokenizer-path', tokenizer] if tokenizer is not None else []
        other_args += ['--chat-template', chat_template] if chat_template is not None else []
        other_args += ['--quantization', 'fp8'] if quantized else []
        # other_args += ['--grammar-backend xgrammar'] if cls.sglang_version >= pkg_resources.parse_version(version) else []
        # other_args += ['--mem-fraction-static', '.9'] if '72b' in model_name.lower() else []
        other_args += ['--tp', str(number_of_devices)]
        print(f"Launching server at GPU {device} with model: {cls.model} and base_url: {cls.base_url} and other_args: {other_args}, Server Engine: {cls.__class__}")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH*10,
            other_args=other_args,
            env=env,
            return_stdout_stderr=False,
            api_key = cls.api_key,
        )
        cls.base_url += "/v1"


    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)
        
    
    def request(cls, inputs: list[dict], tqdm_bar=None, args={}) -> list[str]:
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        o_text = []
        client = openai.Client(api_key=cls.api_key, base_url=cls.base_url)
        hyperparameters = cls.default_hyperparameters.copy()
        hyperparameters.update(args)
        tqdm_bar = tqdm(inputs) if tqdm_bar is None else tqdm_bar
        for input_data in inputs:
            base64_image = encode_image(input_data['image'])
            response = client.chat.completions.create(
                model="default",
                # response_format={ "type": "json_object" },
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image},
                            },
                            {
                                "type": "text",
                                "text": input_data['prompt'],
                            },
                        ],
                    },
                ],
                **hyperparameters,
            )
            o_text.append(response.choices[0].message.content.strip())
            tqdm_bar.update(1)
        client.close()
        return o_text



class SGLangServerAsync(SGLangServer):
    def request(cls, inputs: list[dict], tqdm_bar=None, args={}) -> list[str]:
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        async def request_async(inputs: list[dict], args={}) -> list[str]:
            '''
            inputs: list of dictionaries containing keys 'prompt' and 'image'
            args: dictionary containing hyperparameters
            returns: list of completions
            '''
            
            # async with openai.AsyncOpenAI(api_key=cls.api_key, base_url=cls.base_url) as client:
            client = openai.AsyncOpenAI(api_key=cls.api_key, base_url=cls.base_url)
            hyperparameters = cls.default_hyperparameters

            hyperparameters.update(args)
            response = []
            for input_data in inputs:
                if '<image>' not in input_data['prompt']:
                    raise ValueError('Prompt must contain <image> tag')
                base64_image = encode_image(input_data['image'])
                response.append(
                    client.chat.completions.create(
                        model="default",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                        "url": f"{base64_image}"
                                            },
                                    },
                                    {
                                        "type": "text",
                                        "text": input_data['prompt'],
                                    },
                                ],
                            },
                        ],
                        **hyperparameters,
                        timeout=None
                    )
                )
            print('Sending requests...')
            rets = await asyncio.gather(*response)
            await client.close()  # Ensure the client is properly closed
            out = []
            for r in rets:
                out.append(r.choices[0].message.content.strip())
                tqdm_bar.update(1)
            return out
        return asyncio.run(request_async(inputs, args))
