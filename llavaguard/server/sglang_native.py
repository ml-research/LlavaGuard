import asyncio
import copy
import glob
import signal
from typing import List
import psutil
import aiohttp
import os
import torch
from tqdm.asyncio import tqdm
import json
from llavaguard.evaluation.eval_utils import get_conv, get_model_name_from_path
import json
import os

import openai
from llavaguard.server.sglang import SGLangServer
import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)
import base64
import glob
import os
import sys
import random

async def send_request(url, data, delay=0):
    await asyncio.sleep(delay)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            output = await resp.json()
    return output


@sgl.function
def guard_gen(s, image_path, prompt, tmpl_version, rx=None):
    # if int(tmpl_version.split('v')[1]) >= 20:
    #     s += sgl.system(prompt)
    #     s += sgl.user(sgl.image(image_path))
    # else:
    s += sgl.user(sgl.image(image_path) + prompt)

    hyperparameters = {
        'temperature': 0.2,
        'top_p': 0.95,
        'top_k': 50,
        'max_tokens': 500,
        # 'stop': "}",
    }
    if rx is None:
        s += sgl.assistant(sgl.gen("json_output", **hyperparameters))
    else:
        s += sgl.assistant(sgl.gen("json_output", **hyperparameters, regex=rx))


class SGLangServerNativeAsync(SGLangServer):
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
            response = []
            hyperparameters = cls.default_hyperparameters
            hyperparameters.update(args)
            for input_data in inputs:
                image_path = input_data['image']
                policy = input_data['prompt']
                policy = policy.replace('<image>', '')
                conv_template = copy.deepcopy(get_conv(cls.model_name))
                conv_template.append_message(role=conv_template.roles[0], message=policy)
                conv_template.append_message(role=conv_template.roles[1], message=None)
                policy_with_template = conv_template.get_prompt()
                response.append(
                    send_request(
                        cls.base_url + "/generate",
                        {
                            "text": policy_with_template,
                            "image_data": image_path,
                            "sampling_params": hyperparameters,
                        },
                    )
                    )
            rets = await asyncio.gather(*response)
            out = []
            for r in rets:
                out.append(r.choices[0].message.content.strip())
                tqdm_bar.update(1)
            return out
        return asyncio.run(request_async(inputs, args))


class SGLangServerNative(SGLangServer):
    @classmethod
    def request(cls, inputs: list[dict], tqdm_bar=None, args={}) -> List[str]:
        """Make synchronous request to the server"""
        response = []
        for input_data in inputs:
            prompt = prompt.replace('<image>', '')
            out = guard_gen.run(
                image_path=input_data['image'],
                prompt=input_data['prompt'].replace('<image>', ''),
                rx=None,
                tmpl_version=None)
            response.append(out['json_output'])
            tqdm_bar.update(1)
        return rets