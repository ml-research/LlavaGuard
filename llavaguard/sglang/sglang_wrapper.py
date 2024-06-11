import ast
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from random import randint

if '/workspace' not in sys.path:
    sys.path.append('/workspace')

from llavaguard.sglang.evaluation_wrapper import prepare_model_as_sglang


def launch_server_and_run_funct(model_dir: str, device, function, function_kwargs, HF_HOME: str = '/HF_TMP'):
    print(f"Evaluating model: {model_dir}")
    if 'LlavaGuard' in model_dir:
        if os.path.exists(f"{model_dir}"):
            # prepare model as sglang
            prepare_model_as_sglang(model_dir)
            # prepare server command
            model_size = model_dir.split('LlavaGuard-')[-1].split('-')[1]
        else:
            print('Model not found!')
            return
    else:
        model_size = model_dir.split('-')[-1]

    tokenizers = {
        '7b': 'llava-hf/llava-1.5-7b-hf',
        '13b': 'llava-hf/llava-1.5-13b-hf',
        '34b': 'liuhaotian/llava-v1.6-34b-tokenizer'
    }
    tokenizer = tokenizers[model_size]
    # Set the environment variable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    number_of_devices = str(device).count(',') + 1
    env["HF_HOME"] = HF_HOME
    port = randint(10000, 20000)
    model_dir = f"{model_dir}/llava" if os.path.exists(f"{model_dir}/llava") else model_dir
    server = ["python3", "-m", "sglang.launch_server", "--model-path", model_dir, "--tokenizer-path",
              tokenizer, "--port", str(port), '--tp', str(number_of_devices)]
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
