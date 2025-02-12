import asyncio
import base64
import warnings
from llavaguard.server.sglang import encode_image
from openai import AsyncOpenAI
from lmdeploy import ChatTemplateConfig, GenerationConfig, VisionConfig, pipeline, TurbomindEngineConfig
import torch
from mmengine import Registry
from tqdm.asyncio import tqdm

MODELS = Registry('model', locations=['lmdeploy.model'])

class lmdeployServer:
    def __init__(cls, model_dir: str, device, quantized:bool=False, max_batch_size=20):
        if quantized:
            print('Quantization not supported for lmdeploy turning quantization OFF')
        if device == 'cuda':
            number_of_devices = torch.cuda.device_count()
            d_ts = torch.zeros(1).to(f'cuda')
        else:
            number_of_devices = device.count(',') + 1
            # d_ts = [torch.zeros(1).to(f'cuda:{d}') for d in device.split(',')]
            d_ts = torch.zeros(1).to(f'cuda')
        
        if '1B' in model_dir:
            number_of_devices = 2 if number_of_devices >= 2 else 1
        print(f'Number of devices: {number_of_devices}')
        engine = TurbomindEngineConfig(session_len=8192, 
                                       tp=number_of_devices, 
                                    #    use_tqdm=True,
                                    #    download_dir="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/HF_HOME"
                                       )
        vision_config=VisionConfig(max_batch_size=max_batch_size, thread_safe=True) 
        # chat_template_config = ChatTemplateConfig(model_name=model_dir.split('/')[-1])
        cls.pipe = pipeline(model_dir, 
                            backend_config=engine, 
                            use_tqdm=True,
                            vision_config=vision_config,
                            # chat_template_config=chat_template_config
                            )
        print(f'Loaded model: {model_dir} on device: {device}')
        cls.hyperparameters = {
            'temperature': 0.2, 
            'top_p': 0.95,
            'do_sample': True,
            # 'top_k': 50,
            # 'max_tokens': 500,
            # 'stop': "}",
        }
        
    def request(cls, inputs: list[dict], args={}) -> list[str]:
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        hyperparameters = cls.hyperparameters
        hyperparameters.update(args)
        gen_config = GenerationConfig(**hyperparameters)
        response = cls.pipe([(i['prompt'], i['image']) for i in inputs], gen_config=gen_config, use_tqdm=True)
        return [r.text for r in response]
    
    def request_async(cls, inputs: list[dict], args={}) -> list[str]:
        warnings.warn('Async request is not implemented for this model')
        return cls.request(inputs, args)
    
    def tearDownClass(cls):
        del cls.pipe

class lmdeployServerSequentiel(lmdeployServer):
    def request(cls, inputs: list[dict], args={}) -> list[str]:
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        hyperparameters = cls.hyperparameters
        hyperparameters.update(args)
        gen_config = GenerationConfig(**hyperparameters)
        
        response = []
        for input_data in tqdm(inputs):
            response.append(cls.pipe(input_data['prompt'], input_data['image'], gen_config=gen_config))
        return [r.text for r in response]



class lmdeployServerOpenAIAPI(lmdeployServer):
    def request(cls, inputs: list[dict], args={}) -> list[str]:
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        hyperparameters = cls.hyperparameters
        hyperparameters.update(args)
        gen_config = GenerationConfig(**hyperparameters)
        
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
        response = cls.pipe(prompts, gen_config=gen_config, use_tqdm=True)
        return [r.text for r in response]



class lmdeployServerAsync: # broken
    def __init__(cls, model_dir: str, device):
        raise NotImplementedError
        cls.api_key = "sk-123456"
        cls.base_url = "http://0.0.0.0:23333/v1"

        if device == 'cuda':
            number_of_devices = torch.cuda.device_count()
            d_ts = torch.zeros(1).to(f'cuda')
        else:
            number_of_devices = device.count(',') + 1
            d_ts = [torch.zeros(1).to(f'cuda:{d}') for d in device.split(',')]
            
    async def request(self, inputs: list[dict], args={}):
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        hyperparameters = {
            'temperature': 0.2,
            'top_p': 0.95,
            # 'top_k': 50,
            'max_tokens': 500,
            # 'stop': "}",
        }
        model_cards = await client.models.list()._get_page()
        hyperparameters.update(args)
        response = []
        for input_data in inputs:
            base64_image = encode_image(input_data['image'])
            response.append(
                client.chat.completions.create(
                    model=model_cards.data[0].id,
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
        rets = await tqdm.gather(*response)
        await client.close()  # Ensure the client is properly closed
        return [r.choices[0].message.content.strip() for r in rets]


    def async_request(self, inputs: list[dict], args={}) -> list[str]:
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        return asyncio.run(self.request(inputs, args))