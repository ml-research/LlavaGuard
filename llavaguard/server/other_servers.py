# LlamaGuard works better if we just ask is it sae without providing a taxonomy
import asyncio
import os

import openai
from tqdm import tqdm
from llavaguard.server.sglang import SGLangServer, encode_image


class LlamaGuardServer(SGLangServer):
    async def request(cls, inputs: list[dict], tqdm_bar=None, args={}) -> list[str]:
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
        prompt = "Is the image safe to view?"
        for input_data in inputs:
            org_prompt = input_data['prompt']
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
                                    "text": org_prompt,
                                },
                            ],
                        },
                    ],
                    **hyperparameters,
                    timeout=None
                )
            )
        rets = await asyncio.gather(*response)
        await client.close()  # Ensure the client is properly closed
        out = []
        for r in rets:
            o = r.choices[0].message.content.strip()
            
            if 'unsafe' in o:
                out.append('Unsafe')
            elif 'safe' in o:
                out.append('Safe')
            else:
                raise ValueError(f'Invalid output: "{o}"')
            tqdm_bar.update(1)
            # out.append(o)
        return out
        
    
    def request_async(cls, inputs: list[dict], args={}) -> list[str]:
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        return asyncio.run(cls.request(inputs, args))
    
    

class OpenAIServer():
    def __init__(cls, model_dir: str, device, quantized:bool=False):
        if quantized:
            print('Quantization not supported for VLLM turning quantization OFF')
        cls.client = openai.OpenAI(
            organization='org-Bq1wEbsBxgeiOXBsN1zx2xEf',
            # project='LlavaGuard',
            api_key=os.getenv('OpenAI_token')
        )
        cls.model = model_dir.split('/')[-1]

    def request(cls, inputs: list[dict], tqdm_bar=None, args={}) -> list[str]:
            '''
            inputs: list of dictionaries containing keys 'prompt' and 'image'
            args: dictionary containing hyperparameters
            returns: list of completions
            '''
            o_text = []
            for input_data in tqdm(inputs):
                base64_image = encode_image(input_data['image'])
                response = cls.client.chat.completions.create(
                    model=cls.model,
                    response_format={ "type": "json_object" },
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
                )
                o_text.append(response.choices[0].message.content.strip())
                tqdm_bar.update(1)
            return o_text


    def tearDownClass(cls):
        pass


class OpenAIModerationsServer(OpenAIServer):       
    def request(cls, inputs: list[dict], tqdm_bar=None, args={}) -> list[str]:
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        o_text = []
        for input_data in inputs:
            base64_image = encode_image(input_data['image'])
            try:
                response = cls.client.moderations.create(
                    model=cls.model,
                    # response_format={ "type": "json_object" },
                    input=[
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": f"{base64_image}"
                                    },
                            },
                            # {
                            #     "type": "text",
                            #     "text": input_data['prompt'],
                            # },
                    ],
                )
                # print(type(response))
                # print(response)
                flagged = response.results[0].flagged
                category, c_score = 'None Applying', 0
                category_scores = response.results[0].category_scores
                for c, s in category_scores.to_dict().items():
                    if s > c_score:
                        category = c
                        c_score = s
                gpt_assessment = {}
                gpt_assessment['category'] = category
                gpt_assessment['rating'] = 'Unsafe' if flagged else 'Safe'
                gpt_assessment['model_output'] = response.to_dict()
                o_text.append(gpt_assessment)
            except Exception as e:
                # convert the error to string
                e = str(e)
                print(f'Error at image {input_data["image"]}: {e}')
                o_text.append(e)
            tqdm_bar.update(1)
        return o_text
    
    def request_async(cls, inputs: list[dict], args={}) -> list[str]:
        '''
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        '''
        print('Async request not supported for OpenAI')
        return cls.request(inputs, args)