import glob
import os
import signal
import subprocess
import sys
import time
import traceback
from random import randint

if '/workspace' not in sys.path:
    sys.path.append('/workspace')


def prepare_model_as_sglang(model_dir: str, model_name: str):
    if not 'LlavaGuard' in model_dir:
        raise Exception('Model is not a LlavaGuard model!')
    dest_dir = f'{model_dir}/llava'

    if os.path.exists(dest_dir) and len(glob.glob(f'{dest_dir}/*.safetensors')) > 0:
        print(f'Model {model_name} already prepared for sglang.')
        return
    # check if model exists and if a file with safetensors exists
    elif not os.path.exists(model_dir) or len(glob.glob(f'{model_dir}/*.safetensors')) == 0:
        # print('Failed to prepare model for SGLang. Model does not exist!')
        raise Exception('Failed to prepare model for SGLang. Model does not exist at: ', model_dir)
    root = os.path.abspath("").split('LlavaGuard')[0]
    config_file = f'{root}/llavaguard/configs/{model_name}.json'
    # move all files to llava folder

    os.makedirs(dest_dir)
    try:
        os.system(f'mv {model_dir}/* {dest_dir}/')
    except Exception as e:
        pass
    # replace config file
    if model_name != 'LlavaGuard-v1.2-7b-ov':
        os.system(f'cp {config_file} {dest_dir}/config.json')
    # remove previous checkpoints from dest_dir
    # for f in glob.glob(f'{dest_dir}/checkpoint*'):
    #     # remove dir
    #     if os.path.isdir(f):
    #         os.system(f'rm -rf {f}')
    #         print(f"Removed intermediate checkpoint: {f}")

    print(f'Model {model_name} prepared for sglang! Ready to evaluate.')


def launch_server_and_run_funct(model_dir: str, device, function, function_kwargs, HF_HOME: str = None):
    print(f"Evaluating model: {model_dir}")        
    if 'LlavaGuard-v' in model_dir and 'AIML' not in model_dir:
        if os.path.exists(f"{model_dir}"):
            # prepare model as sglang
            model_name = model_dir.split('models/')[-1].split('/')[0]
            prepare_model_as_sglang(model_dir, model_name)
            # prepare server command
        else:
            print(f'Run aborted. No model avaiable at {model_dir}')
            return
    else:
        model_name = model_dir
    sglang_tokenizers = {
        'LlavaGuard-v1.1-7b': 'llava-hf/llava-1.5-7b-hf',
        'LlavaGuard-v1.1-13b': 'llava-hf/llava-1.5-13b-hf',
        'LlavaGuard-v1.2-13b': 'llava-hf/llava-v1.6-vicuna-13b-hf',
        'LlavaGuard-v1.2-34b': 'liuhaotian/llava-v1.6-34b-tokenizer',
        'LlavaGuard-v1.2-7b-ov': 'llava-hf/llava-onevision-qwen2-7b-ov-hf',
        'LlavaGuard-v1.2-8b': 'lmms-lab/llama3-llava-next-8b',
        'liuhaotian/llava-v1.5-7b': 'llava-hf/llava-1.5-7b-hf',
        'liuhaotian/llava-v1.5-13b': 'llava-hf/llava-1.5-13b-hf',
        'liuhaotian/v1.6-vicuna-7b': 'llava-hf/llava-v1.6-vicuna-7b-hf',
        'liuhaotian/v1.6-vicuna-13b': 'llava-hf/llava-v1.6-vicuna-13b-hf',
        'liuhaotian/llava-v1.6-34b': 'liuhaotian/llava-v1.6-34b-tokenizer',
        'lmms-lab/llama3-llava-next-8b': None,
        'lmms-lab/llava-onevision-qwen2-7b-ov': None,
        }
    if '34b' in model_name:
        chat_template = None
    elif 'ov' in model_name:
        chat_template = "chatml-llava"
    elif '8b' in model_name:
        chat_template = 'llava_llama_3'
    else:
        chat_template = None
    try:
        tokenizer = sglang_tokenizers[model_name]
    except KeyError:
        print(f"Tokenizer not found for model {model_name} using default tokenizer.")
        tokenizer = None
    # Set the environment variable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    number_of_devices = str(device).count(',') + 1
    if HF_HOME is not None:
        env["HF_HOME"] = HF_HOME
        print('HF_HOME set to ', HF_HOME)

    port = randint(10000, 20000)
    model_dir = f"{model_dir}/llava" if os.path.exists(f"{model_dir}/llava") else model_dir
    server = ["python", "-m", "sglang.launch_server", "--model-path", model_dir, "--port", str(port), '--tp', str(number_of_devices),
              # '--chunked-prefill-size', '-1',
              # '--mem-fraction-static', '0.5',
              # '--context-length', '8192'
              ]
    server += ['--tokenizer-path', tokenizer] if tokenizer is not None else []
    server += ['--chat-template', chat_template] if chat_template is not None else []
    print(f"Launching server at GPU {device} with command: {' '.join(server)}")
    server_process = subprocess.Popen(server, env=env, preexec_fn=os.setsid)

    time.sleep(100)
    # add port to function_kwargs
    function_kwargs['port'] = port
    print(function_kwargs)
    # start evaluation
    try:
        function(**function_kwargs)
    except Exception:
        print(f'Could not evaluate model. Exiting with error:')
        traceback.print_exc()

    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)  # Send the signal to all the process groups
    time.sleep(30)
