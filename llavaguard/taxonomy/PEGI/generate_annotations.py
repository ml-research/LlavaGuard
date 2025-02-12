
import argparse
import asyncio
import glob
from itertools import product
import os
import random

import pandas as pd
import rtpt
import torch
from tqdm import tqdm
from llavaguard_config import local_image_dirs, local_data_dir
from llavaguard.taxonomy.PEGI.PEGI_Graph import get_content_categories, get_content_categories_with_examples, get_content_categories_with_numbers, get_policy_intro, get_safety_categories, policy_graph, policy_graph_to_safety_policy


not_clear_statement = 'If the context of the image is not clear, select the most related categories.'
no_explanation_statement = 'You do not need to provide an explanation for your answer.'
describe_image_prompt = 'Briefly describe the image content.'

def prompt_for_every_content_category(im_paths, output_dir, replace_existing=False, describe_image=False):
    content_categories = get_content_categories(policy_graph)
    inputs = []
    prompting_name = 'prompt_for_every_content_category' + ('_describe_image2' if describe_image else '')
    existing_invalids = 0
    existing = 0
    for im_path in tqdm(im_paths):
        data_id = im_path.split('/')[-1].split('.')[0]
        folder_path = os.path.dirname(im_path).split('images/')[-1]
        content_category_id = 1
        for safety_category, graph_detail in policy_graph.items():
            for content_category, graph_info in graph_detail.items():
                out_path = os.path.join(output_dir, prompting_name, folder_path, data_id, f'{content_category_id}. {content_category}' + '.txt')
                if os.path.exists(out_path) and not replace_existing:
                    with open(out_path, 'r') as f:
                        c = f.read()
                        if 'Yes' in c or 'Not Applicable' in c or 'No' in c:
                            content_category_id += 1
                            existing += 1
                            continue
                        else:
                            # print(c)
                            # print(f"Existing file {out_path} is not valid. Replacing it.")
                            existing_invalids += 1
                
                s_cat = safety_category.split(": ")[1]
                prompt = ""
                # prompt += describe_image_prompt + ' ' if describe_image else ''
                if content_category == 'Educational Content':
                    # prompt += f"Does the image include content of type '{s_cat}' that is of educational nature {graph_info['edge']} {s_cat} content of type {', '.join(graph_info['examples'])}? If the content is of educational nature, answer with 'Yes'. Otherwise, answer with 'No'."
                    # prompt += f"Evaluate whether the image contains educational content related to the category '{s_cat}', {graph_info['edge']} {', '.join(graph_info['examples'])}. If the image is related to '{s_cat}' and is clearly e, respond with 'Yes', otherwise with 'No'."
                    prompt += f"Evaluate whether the image contains educational content related to {s_cat}. This includes {graph_info['edge']} {', '.join(graph_info['examples'])}. Specifically, determine if the image is associated with {s_cat} but does not depict explicit {s_cat} activities and is clearly educational in its intent and presentation." 
                elif 'disturbing' in content_category.lower() or 'scary' in content_category.lower():
                    prompt += f"Evaluate whether the image contains disturbing content. While it may not necessarly depict violence, it should evoke feelings of fear or horror. Specifically, it should be categorized as '{content_category}' {graph_info['edge']} {', '.join(graph_info['examples'])}."
                    if 'highly disturbing content' in content_category.lower():
                        prompt += f" Evaluate whether the image is evokes feelings of extreme fear depicting intense horror sequenses or strong gore. Mildly disturbing content should be answered with 'No'."
                else:
                    prompt += f"Evaluate whether the image contains content of type '{s_cat}' that is considered '{content_category}' {graph_info['edge']} {', '.join(graph_info['examples'])}."
                if describe_image:
                    prompt += f" Describe the image context then answer the question with 'Yes' or 'No'."
                else:
                    prompt += f" If this exact type of content is present respond with 'Yes', otherwise with 'No'."
                
                if content_category == 'Educational Content' or 'disturbing' in content_category.lower() or 'scary' in content_category.lower():
                    inputs.append(
                        {"prompt": prompt,
                        "content_category": content_category,
                        "image": im_path,
                        'data_id': data_id,
                        'out_path': out_path,
                        'prediction_type': prompting_name
                        }
                        )
                content_category_id += 1
    print(f"Inference on {len(inputs)}, All Images: {len(im_paths)}, Replace Existing: {replace_existing} (Existing: {existing}/{len(im_paths) * len(content_categories)}, Num. Invalids to be replaced: {existing_invalids})")
    return inputs


def prompt_for_image_description(im_paths, output_dir, replace_existing=False):
    if not os.path.exists(f"{local_data_dir}/data/annotations/auto_generated_annotations/v0_2/majority_vote.csv"):
        raise FileNotFoundError("Annotations file not found. Please run the annotation generation script first.")
    annotations = pd.read_csv(f"{local_data_dir}/data/annotations/auto_generated_annotations/v0_2/majority_vote.csv")
    annotations = annotations[annotations['voting_mechanism'] == 'majority_vote'].reset_index(drop=True)    
    intro = f"Describe the image and how it relates the following categories in 2-3 sentance. Do not use quotations. Safety Categories:\n"
    inputs = []
    prompting_name = 'image_description'
    for im_path in tqdm(im_paths):
        data_id = im_path.split('/')[-1].split('.')[0]
        sample_annotations = annotations[annotations['sample_id'] == data_id]
        folder_path = os.path.dirname(im_path).split('images/')[-1]
        out_path = os.path.join(output_dir, prompting_name, folder_path, f'{data_id}.txt')
        if os.path.exists(out_path) and not replace_existing:
            continue
        cats = []
        for category in sample_annotations.columns[4:]:
            if 1 in sample_annotations[category].values:
                cats.append(category)
        if len(cats) == 0:
            prompt = "Describe the user content in 2-3 sentance. Do not use quotations."
            inputs.append(
                {"prompt": prompt,
                "image": im_path,
                'data_id': data_id,
                'out_path': out_path,
                'prediction_type': prompting_name
                }
            )
        else:
            prompt = f"{intro} {', '.join(cats)}"

        # inputs.append(
        #     {"prompt": prompt,
        #     "image": im_path,
        #     'data_id': data_id,
        #     'out_path': out_path,
        #     'prediction_type': prompting_name
        #     }
        # )
    print(f"Inference (image descriptions) on {len(inputs)}, All Images: {len(im_paths)}, Replace Existing: {replace_existing} (Existing: {len(im_paths) - len(inputs)}/{len(im_paths)})")
    return inputs



def prompt_for_all_content_categories(im_paths, output_dir, replace_existing=False):
    inputs = []
    # content_categories_with_examples = get_content_categories_with_examples(policy_graph)
    # category_list = "\n".join(content_categories_with_examples)
    categories = get_content_categories(policy_graph)
    categories_with_numbers = '"' + '","'.join([f"{i+1}. {category}" for i, category in enumerate(categories)]) + '"'
    prompting_name = 'prompt_for_all_content_categories'
    
    policy = policy_graph_to_safety_policy(18).replace('Prohibited content categories:\n', '').replace('Permitted content categories:\n', '').replace(get_policy_intro(), '').replace('None\n', '').split('Assessment Steps:')[0]
    prompt = f"Above you find a list of safety categories (S1-S9) and content categories (1-47). You may answer with 'None Applicable' if none of the 47 categories are included. Otherwise, list all content categories (1-47) that are included in the image."
    
    for im_path in tqdm(im_paths):
        data_id = im_path.split('/')[-1].split('.')[0]
        folder_path = os.path.dirname(im_path).split('images/')[-1]
        out_path = os.path.join(output_dir, prompting_name, folder_path, f'{data_id}.txt')
        if os.path.exists(out_path) and not replace_existing:
            continue
        inputs.append(
            {"prompt": policy + prompt,
                "image": im_path,
                'data_id': data_id,
                'out_path': out_path,
                'prediction_type': prompting_name
                }
            )
    print(f"Inference on {len(inputs)}, All Images: {len(im_paths)}, Replace Existing: {replace_existing} (Existing: {len(im_paths) - len(inputs)}/{len(im_paths)})")
    return inputs


def prompt_for_all_safety_categories(im_paths, output_dir, replace_existing=False):
    inputs = []
    # content_categories_with_examples = get_content_categories_with_examples(policy_graph)
    # category_list = "\n".join(content_categories_with_examples)
    categories = get_safety_categories(policy_graph)
    categories_with_numbers = '"' + '","'.join(categories) + '"'
    prompting_name = 'prompt_for_all_safety_categories'
    
    policy = policy_graph_to_safety_policy(18).replace('Prohibited content categories:\n', '').replace('Permitted content categories:\n', '').replace(get_policy_intro(), '').replace('None\n', '').split('Assessment Steps:')[0]
    prompt = f"Above you find a list of safety categories (S1-S9) with corresponding sub-categories (1-47) providing additional details on the safety categories. If none of the safety categories are appliable for the image, answer with 'None Applicable'. Otherwise list all safety categories (S1-S9) that are included in the image."
    
    for im_path in tqdm(im_paths):
        data_id = im_path.split('/')[-1].split('.')[0]
        folder_path = os.path.dirname(im_path).split('images/')[-1]
        out_path = os.path.join(output_dir, prompting_name, folder_path, f'{data_id}.txt')
        if os.path.exists(out_path) and not replace_existing:
            continue
        inputs.append(
            {"prompt": policy + prompt,
                "image": im_path,
                'data_id': data_id,
                'out_path': out_path,
                'prediction_type': prompting_name
                }
            )
    print(f"Inference on {len(inputs)} samples, All Images: {len(im_paths)}, Replace Existing: {replace_existing} (Existing: {len(im_paths) - len(inputs)}/{len(im_paths)})")
    return inputs

def prompt_for_every_safety_categories(im_paths, output_dir, replace_existing=False):
    prompting_name = 'prompt_for_every_safety_categories'
    content_categories = get_content_categories(policy_graph)
    safety_categories = get_safety_categories(policy_graph)
    content_categories_with_examples = get_content_categories_with_examples(policy_graph)
    policy_graph
    inputs = []
    prompts = {}
    for safety_category in safety_categories:
        safety_category_details = policy_graph[safety_category]
        prompts[safety_category] = ''
        for c_id, (content_category, graph_info) in enumerate(safety_category_details.items()):
            prompts[safety_category] += f'\n{c_id}. "{content_category}" {graph_info["edge"]} {", ".join(graph_info["examples"])}'
            
    for im_path in tqdm(im_paths):
        for safety_category in safety_categories:
            data_id = im_path.split('/')[-1].split('.')[0]
            folder_path = os.path.dirname(im_path).split('images/')[-1]
            out_path = os.path.join(output_dir, prompting_name, folder_path, data_id, safety_category.replace(' ', '_') + '.txt')
            if os.path.exists(out_path) and not replace_existing:
                continue
            s_cat = safety_category.split(": ")[1]
            prompt = f"Does the image include content of type '{s_cat}' that is within the following categories: {prompts[safety_category]}"
            prompt += f"\n\nIf this type of content is not included in the image, answer with 'Not Applicable'. Otherwise list all categories that are included in the image."
            inputs.append(
                {"prompt": prompt,
                "safety_category": safety_category,
                "image": im_path,
                'data_id': data_id,
                'out_path': out_path,
                'prediction_type': prompting_name
                }
                )
    print(f"Inference on {len(inputs)}, All Images: {len(im_paths)}, Num. Safety categories {len(safety_categories)}, Replace Existing: {replace_existing} (Existing: {len(im_paths) * len(safety_categories) - len(inputs)}/{len(im_paths) * len(safety_categories)})")
    return inputs


def run_sglang_batched(inputs, model, devices, batch_size):
    from llavaguard.sglang_server import SGLangServer
    server = SGLangServer(model, devices)
    rt = rtpt.RTPT(name_initials='LH', experiment_name=f'GraphGuard-Gen', max_iterations=len(inputs)//batch_size + 1)
    rt.start()
    for i in range(0, len(inputs), batch_size):
        print(f"Running batch {i//batch_size + 1}/{len(inputs)//batch_size + 1}")
        annotations = asyncio.run(server.request_async([{"image": input_['image'], "prompt": input_['prompt']} for input_ in inputs[i:i+batch_size]]))
        rt.step()
        for input_, annotation in zip(inputs[i:i+batch_size], annotations):
            os.makedirs(os.path.dirname(input_['out_path']), exist_ok=True)
            if i == 0 and 'Yes' not in annotation and 'No' not in annotation:
                example_prompt_dir = input_['out_path'].replace(input_['prediction_type'], f"invalid_prompt_responses_{input_['prediction_type']}")
                # save the prompt in the first file
                os.makedirs(os.path.dirname(example_prompt_dir), exist_ok=True)
                with open(example_prompt_dir, 'w') as f:
                    f.write(input_['prompt'] + '\n Response:' + annotation)
            with open(input_['out_path'], 'w') as f:
                f.write(annotation)
                # print(annotation)
    server.tearDownClass()
    
    
def run_lmdeploy_batched(inputs, model, devices, batch_size):
    from llavaguard.lmdeploy_server import lmdeployServer
    server = lmdeployServer(model, devices)
    rt = rtpt.RTPT(name_initials='LH', experiment_name=f'GraphGuard-Gen', max_iterations=len(inputs)//batch_size + 1)
    rt.start()
    pb = tqdm(total=len(inputs))
    for i in range(0, len(inputs), batch_size):
        i2 = min(i+batch_size, len(inputs))
        pb.set_description(f"Running batch {i//batch_size + 1}/{len(inputs)//batch_size + 1}")
        annotations = server.request([{"image": input_['image'], "prompt": input_['prompt']} for input_ in inputs[i:i2]])
        rt.step()
        for input_, annotation in zip(inputs[i:i2], annotations):
            os.makedirs(os.path.dirname(input_['out_path']), exist_ok=True)
            if i == 0 and 'Yes' not in annotation and 'No' not in annotation:
                example_prompt_dir = input_['out_path'].replace(input_['prediction_type'], f"invalid_prompt_responses_{input_['prediction_type']}")
                # save the prompt in the first file
                os.makedirs(os.path.dirname(example_prompt_dir), exist_ok=True)
                with open(example_prompt_dir, 'w') as f:
                    f.write(input_['prompt'] + '\n Response:' + annotation)
            with open(input_['out_path'], 'w') as f:
                f.write(annotation)
                print(annotation)
        pb.update(i2-i)
    pb.close()
        
def generate_annotations(model, devices, output_dir, replace_existing=False):
    # Load the template
    if output_dir is None:
        output_dir = f"{local_data_dir}/data/annotations/auto_generated_annotations/{model.split('/')[-1]}"
    print(f"Annotations will be saved at {output_dir}, Model: {model}, Devices: {devices}")

    im_paths = glob.glob(local_image_dirs['smid'] + '/*.jpg') + glob.glob(local_image_dirs['crawled'] + '/*/*.jpg') + glob.glob(local_image_dirs['synthetic'] + '/*/*.jpg')
    # im_paths = im_paths[:1]
    data_ids = [path.split('/')[-1].split('.')[0] for path in im_paths]
    assert len(set(data_ids)) == len(data_ids), 'Data IDs (image names) are not unique'  # if ids are not unique raise error
    content_categories = get_content_categories(policy_graph)
    batch_size = 1000
    
    random.seed(42)

    # Select 50 random elements from im_paths
    # im_paths = random.sample(im_paths, 150)


    inputs = []
    # inputs += prompt_for_every_content_category(im_paths, output_dir, replace_existing=replace_existing, describe_image=True)
    # inputs += prompt_for_every_content_category(im_paths, output_dir, replace_existing=replace_existing, describe_image=False)
    inputs += prompt_for_image_description(im_paths, output_dir, replace_existing=replace_existing)
    # inputs += prompt_for_all_content_categories(im_paths, output_dir, replace_existing=replace_existing)
    # inputs += prompt_for_every_safety_categories(im_paths, output_dir, replace_existing=replace_existing)
    # inputs += prompt_for_all_safety_categories(im_paths, output_dir, replace_existing=replace_existing)
    
    
    print(f"Total {len(inputs)} annotations will be generated in {len(inputs)//batch_size + 1} batches")
    if len(inputs) == 0:
        print("No new annotations to generate.")
        return
    if 'internvl' in model.lower():
        print("Running lmdeploy engine")
        run_lmdeploy_batched(inputs, model, devices, batch_size)
    else:
        print("Running sglang engine")
        run_sglang_batched(inputs, model, devices, batch_size)
    print("Annotations generated successfully to", output_dir)
    
    
if __name__ == "__main__":
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--output_dir', type=str, default=None)
    parser1.add_argument('--replace_existing', type=bool, default=True)
    parser1.add_argument('--device', type=str, default="0,1,2,3")
    parser1.add_argument(
        "--engine",
        help="Inference Engine to use",
        default="Sglang",
        choices=[
            "Sglang",
            "lmdeploy",
        ],
        required=False,
    )
    # parser1.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    # parser1.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen2-VL-72B-Instruct")
    # parser1.add_argument('--model_name_or_path', type=str, default="lmms-lab/llava-onevision-qwen2-72b-ov")
    args = parser1.parse_args()
    # models = ["lmms-lab/llava-onevision-qwen2-7b-ov", "lmms-lab/llava-onevision-qwen2-72b-ov", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-72B-Instruct", 'liuhaotian/llava-v1.6-34b','liuhaotian/llava-v1.6-vicuna-13b', "lmms-lab/llama3-llava-next-8b", "OpenGVLab/InternVL2-Llama3-76B",  ]
    models = ["lmms-lab/llava-onevision-qwen2-7b-ov",  "lmms-lab/llava-onevision-qwen2-72b-ov","Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-72B-Instruct", "lmms-lab/llama3-llava-next-8b", "OpenGVLab/InternVL2-Llama3-76B", "OpenGVLab/InternVL2-40B", "OpenGVLab/InternVL2-8B"]
    for model in models[3:4]:
       generate_annotations(model, args.device, args.output_dir, args.replace_existing)

