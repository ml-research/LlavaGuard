import argparse
import glob
import signal
import subprocess
import sys
import time

import argparse
import os
from random import randint

if '/workspace' not in sys.path:
    sys.path.append('/workspace')
from llavaguard.sglang.evaluation import evaluate_sglang


def prepare_model_as_sglang(model_dir: str):
    if not 'LlavaGuard' in model_dir:
        print('Model is not a LlavaGuard model!')
        return
    dest_dir = f'{model_dir}/llava'

    if os.path.exists(dest_dir) and len(glob.glob(f'{dest_dir}/*.safetensors')) > 0:
        print('Model already prepared for sglang.')
        return
    elif not os.path.exists(model_dir):
        print('Model does not exist!')
        return
    root = os.path.abspath("").split('LlavaGuard')[0]
    llavaguard_name = model_dir.split('/')[-3].replace('-full', '')
    config_file = f'{root}/llavaguard/configs/{llavaguard_name}.json'
    # move all files to llava folder

    os.makedirs(dest_dir)
    try:
        os.system(f'mv {model_dir}/* {dest_dir}/')
    except Exception as e:
        pass
    # replace config file
    os.system(f'cp {config_file} {dest_dir}/config.json')
    # remove previous checkpoints from dest_dir
    # for f in glob.glob(f'{dest_dir}/checkpoint*'):
    #     # remove dir
    #     if os.path.isdir(f):
    #         os.system(f'rm -rf {f}')
    #         print(f"Removed intermediate checkpoint: {f}")

    print('Model prepared for sglang! Ready to evaluate.')


def launch_server_and_evaluate(model_dir: str, data_path: str, device: int, infer_train_data: bool = False):
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
    env["HF_HOME"] = '/HF_TMP' if 'vicuna' not in model_dir else '/HF_TMP2'
    port = randint(10000, 20000)
    model_dir = f"{model_dir}/llava" if os.path.exists(f"{model_dir}/llava") else model_dir
    server = ["python3", "-m", "sglang.launch_server", "--model-path", model_dir, "--tokenizer-path",
              tokenizer, "--port", str(port)]
    # launch the server
    print(f"Launching server with command: {' '.join(server)}")
    server_process = subprocess.Popen(server, env=env, preexec_fn=os.setsid,
                                      # stdout=subprocess.PIPE,
                                      # stderr=subprocess.STDOUT, universal_newlines=True
                                      )
    # Wait until "load weight end." is printed
    # for line in iter(server_process.stdout.readline, ''):
    #     print(line, end='')  # print the server's output in real-time
    #     if "POST /generate HTTP/1.1" in line:
    #         break
    # print("Server is ready!")
    # Wait for another 10 seconds
    # time.sleep(int(model_size.replace('b', '')) * 10)
    time.sleep(60)

    # start evaluation
    try:
        evaluate_sglang(data_path=data_path, port=port, infer_train_data=infer_train_data)
    except Exception as e:
        print(f'Could not evaluate model. Exiting with error: {e}')

    # Kill the server process
    # server_process.terminate()
    # server_process.kill() # process terminates but the server is still running, killpg does not work when called subsequently
    # server_process.wait()

    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)  # Send the signal to all the process groups
    # wait for the process to terminate
    time.sleep(30)
    # close the running docker containers



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--infer_train_data', action='store_true')
    parser.add_argument('--device', default=0)
    args = parser.parse_args()
    d = args.device if isinstance(args.device, int) else int(args.device[0])
    launch_server_and_evaluate(args.model_dir, args.data_path, d, args.infer_train_data)
