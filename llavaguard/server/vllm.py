import PIL
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import torch


class VLLMServer():
    def __init__(cls, model_dir: str, device, quantized:bool=False, model_args: dict=None, sampling_params: dict=None):
        if quantized:
            print('Quantization not supported for VLLM turning quantization OFF')
        
        if 'Qwen/Qwen2.5-VL' in model_dir or 'QwenGuard-v1.2-3b' in model_dir or 'QwenGuard2.5-v1.2':
            cls.model = LLM(
                model=model_dir,
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=32760,

                )
            cls.processor = AutoProcessor.from_pretrained(model_dir)
            cls.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.001,
                top_k=1,
                repetition_penalty=1.05,
                max_tokens=500,
                stop_token_ids=[],
            )
        elif 'LlavaGuard' in model_dir:
            cls.model = LLM(
                model=model_dir,
                tensor_parallel_size=torch.cuda.device_count(),
                # max_model_len=131072,
            )
            cls.processor = AutoProcessor.from_pretrained(model_dir)
            cls.sampling_params = SamplingParams(
                temperature=0.2,
                top_p=0.95,
                max_tokens=500,
            )
            

    def request(cls, inputs: list[dict], tqdm_bar=None, args={}):
        """
        Run model inference in batches for multiple prompts and images.
        
        Args:
            inputs: list of dictionaries containing keys 'prompt' and 'image'
        
        Returns:
            List of model outputs
        """
        
        # apply chat template from processor
        prompts_v3 = []
        for i in inputs:
            prompt_without_im_token =  i['prompt'].replace('<image>', '')
            conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt_without_im_token
                            },
                            {
                                "type": "image",
                            },
                        ],
                    },
                ]
            wrapped_conversation = cls.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            prompts_v3.append({
                "prompt": wrapped_conversation,
                "multi_modal_data": {"image": PIL.Image.open(i['image'])},
            })
        output = cls.model.generate(prompts_v3, sampling_params=cls.sampling_params, use_tqdm=True)
        return [o.outputs[0].text for o in output]
    
    def tearDownClass(cls):
        del cls.model, cls.processor
        


# def forward_llava_ov(self, inputs):
#         """
#         Run model inference in batches for multiple prompts and images.
        
#         Args:
#             inputs
        
#         Returns:
#             List of model outputs
#         """
#         prompts_v2 = [{
#                 "prompt": f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{i['prompt']}<|im_end|>\n<|im_start|>assistant\n",
#                 "multi_modal_data": {"image": PIL.Image.open(i['image'])},
#             } for i in inputs]
#         output = self.model.generate(prompts_v2, sampling_params=self.sampling_params, use_tqdm=True)
#         return [o.outputs[0].text for o in output]
