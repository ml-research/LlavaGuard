import asyncio

import torch
from tqdm.asyncio import tqdm


from llavaguard.server.sglang import encode_image, prepare_model_for_sglang
import sglang
import base64
import sys

if '/workspace' not in sys.path:
    sys.path.append('/llava/LLaVA-NeXT')


class SGLangServerOffline:
    def __init__(cls, model_dir: str, device, quantized:bool=False):
        if quantized:
            print('Quantization not supported for VLLM turning quantization OFF')
        sglang_dir, tokenizer, chat_template = prepare_model_for_sglang(model_dir)
        number_of_devices = torch.cuda.device_count() if device == 'cuda' else device.count(',') + 1
        d_ts = torch.zeros(1).to(f'cuda')
        if '0.5' in sglang_dir:
            number_of_devices = 2 if number_of_devices >=2 else 1
            
        cls.vlm = sglang.Engine(model_path=sglang_dir, tokenizer_path=tokenizer, tp_size=number_of_devices, chat_template=chat_template)
        cls.default_hyperparameters = {
            'temperature': 0.2, 
            'top_p': 0.95,
            # 'top_k': 50,
            'max_tokens': 500,
            # 'stop': "}",
        }
   
    def request(cls, inputs: list[dict], tqdm_bar=None, args={}):
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        
        # async with openai.AsyncOpenAI(api_key=cls.api_key, base_url=cls.base_url) as client:
        hyperparameters = cls.default_hyperparameters
        hyperparameters.update(args)
        prompts = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"{encode_image(i['image'])}"
                                },
                        },
                        {
                            "type": "text",
                            "text": i['prompt'],
                        },
                    ],
                } for i in inputs]
        
        outputs = cls.vlm.generate(prompts, hyperparameters)
        out = []
        for output in outputs:
            out.append(output['text'])
            tqdm_bar.update(1)
        return out
    
    def tearDownClass(cls):
        cls.vlm.shutdown()


class SGLangServerOfflineAsync(SGLangServerOffline):
    def request(cls, inputs: list[dict], tqdm_bar=None, args={}) -> list[str]:
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        async def request_async(inputs: list[dict], args={}):
            '''
            inputs: list of dictionaries containing keys 'prompt' and 'image'
            args: dictionary containing hyperparameters
            returns: list of completions
            '''
            
            # async with openai.AsyncOpenAI(api_key=cls.api_key, base_url=cls.base_url) as client:
            hyperparameters = cls.default_hyperparameters
            hyperparameters.update(args)
            prompts = [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": f"{encode_image(i['image'])}"
                                    },
                            },
                            {
                                "type": "text",
                                "text": i['prompt'],
                            },
                        ],
                    } for i in inputs]
            outputs = await asyncio.gather(cls.vlm.async_generate(prompts, hyperparameters))           
            out = []
            for output in outputs:
                out.append(output['text'])
                tqdm_bar.update(1)
            return out
        return asyncio.run(request_async(inputs, args))