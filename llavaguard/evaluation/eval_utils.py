import base64
import copy
import json
import os

import numpy as np

from llavaguard.evaluation.metrics_calculator import get_keys
from llavaguard.taxonomy.assessment import categories2, categories_v3
from llavaguard_config import llava_guard_config
from llavaguard_config import local_data_dir



def is_valid_json(test, print_error=True):
    try:
        with open(test, 'r') as f:
            data = json.load(f)
            data = json.loads(data) if isinstance(data, str) else data
            keys = get_keys(data)
            return True
    except FileNotFoundError:
        if print_error:
            print(f"File not found: {test}")
        return False
    except json.JSONDecodeError as e:
        if print_error:
            print(f"JSON decode error in file {test}")
        return False
    except Exception as e:
        if print_error:
            print(f"Invalid Assessment: {e}")
        return False



def get_model_dir(run_name):
    if os.path.exists(run_name):
        return run_name
    if os.path.exists(f'{local_data_dir}/models/{run_name}'):
        return f'{local_data_dir}/models/{run_name}'
    elif os.path.exists(f'output/models/{run_name}'):
        return f'output/models/{run_name}'
    else:
        return None


def load_data(data_path, split=None):
    dd = {}
    paths = {}
    if data_path.endswith('.json'):
        d = json.load(open(data_path))
        d = fix_broken_image_paths(d)
        dd = {data_path.split('/')[-1].split('.')[0]: d}
        paths = {data_path.split('/')[-1].split('.')[0]: data_path}
        return paths, dd
    elif split is None:
        split = ['train', 'eval', 'test']
    split = [split] if isinstance(split, str) else split
    data = [(data_path, s) for s in split]
    for p, t in data:
        # if type == 'train' and not infer_train_data:
        #     continue
        if not p.endswith('/'):
            p += '/'
        p += f'{t}.json'
        if os.path.exists(p):
            dd[t] = json.load(open(p))
        elif os.path.exists(f'{local_data_dir}/data/{p}'):
            dd[t] = json.load(open(f'{local_data_dir}/data/{p}'))
        elif os.path.exists(f'output/data/{p}'):
            dd[t] = json.load(open(f'output/data/{p}'))
        else:
            raise FileNotFoundError(f'No data found for {p}')
        paths[t] = p
    return paths, dd


def fix_broken_image_paths(data: list[dict]):
    for item in data:
        if local_data_dir not in item['image']:
            item['image'] = local_data_dir + '/data/' + item['image'].split('data/')[-1]
    return data


def chunks(df, n):
    """Yield n chunks from df."""
    for split in np.array_split(df, n):
        yield split


def set_up_dynamic_regex(template='json-v10'):
    cats = categories2 if template in ['json-v10', 'json-v11'] else categories_v3
    cats_txt = '|'.join(cats)
    if template not in llava_guard_config.keys():
        raise ValueError(f'Unknown template: {template}')
    j_templ = repr(llava_guard_config[template]['response_template'])
    j_templ = j_templ.split('{')[1].split('}')[0]
    # j_templ.replace("'", '')
    j_templ = j_templ.replace('str<"Review Needed"|"Compliant">', r""" "(Review Needed|Compliant)" """)
    j_templ = j_templ.replace(
        'str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Illegal Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"10: None applying">',
        f'"({cats_txt})"')
    j_templ = j_templ.replace('str', r""" "[\w\d ]{1,250}" """)
    j_templ = '\{' + j_templ + '\}'
    # to raw string
    return j_templ


def set_up_static_regex():
    return (
            r"""\{\n"""
            + r"""    "image-category":  model_dir.replace('/llava', '').split('/')[-1]
        output_dir ="(O1: Hate, Humiliation, Harassment|O2: Violence, Harm or Cruelty|O3: Sexual Content|O4: Nudity Content|O5: Criminal Planning|O6: Illegal Weapons or Substance Abuse|O7: Self-Harm|O8: Animal Cruelty|O9: Disasters or Emergencies|10: None applying)",\n"""
            + r"""    "assessment": "[\w\d ]{1,250}",\n"""
            + r"""    "decision": "(Review Needed|Compliant)",\n"""
            + r"""\}"""
    )


def model_dir_to_output_dir(model_dir, data_dir=None):
    if os.path.exists(model_dir) and os.path.exists(data_dir):
        # output_dir = model_dir.replace('models', 'eval').replace('/llava', '')
        model_name = model_dir.split('models/')[1].replace('/', '-')
    elif data_dir is not None and os.path.exists(data_dir):
        model_name = model_dir.split('/')[-1]
    else:
        raise FileNotFoundError(f"Output dir not found for Model: {model_dir} and Data {data_dir}")
    data_name, data_version = data_dir.split('data/')[-1].split('/')[0], data_dir.split('data/')[-1].split('/')[1]
    return f'{local_data_dir}/eval/{model_name}/{data_name}-{data_version}'

def get_model_name_from_path(model_dir):
    if os.path.exists(model_dir):
        if 'models/' in model_dir:
            return model_dir.split('models/')[-1].split('/')[0]
        else:
            return model_dir.split('/')[-1]
    else:
        return model_dir


def get_base_model_name_from_path(model_dir):
    model_map = {
        'LlavaGuard-v1.1-7b': 'liuhaotian/llava-1.5-7b-hf',
        'LlavaGuard-v1.1-13b': 'liuhaotian/llava-1.5-13b-hf',
        'LlavaGuard-v1.2-13b': 'liuhaotian/llava-v1.6-vicuna-13b-hf',
        'LlavaGuard-v1.2-34b': 'liuhaotian/llava-v1.6-34b-tokenizer',
        'LlavaGuard-v1.2-7b-ov': 'lmms-lab/llava-onevision-qwen2-7b-ov',
        'LlavaGuard-v1.2-7b-ov-chat': 'lmms-lab/llava-onevision-qwen2-7b-ov-chat',
        'LlavaGuard-v1.2-8b': 'lmms-lab/llama3-llava-next-8b',
        'liuhaotian/llava-v1.5-7b': 'liuhaotian/llava-1.5-7b-hf',
        'liuhaotian/llava-v1.5-13b': 'liuhaotian/llava-1.5-13b-hf',
        'liuhaotian/v1.6-vicuna-7b': 'liuhaotian/llava-v1.6-vicuna-7b-hf',
        'liuhaotian/v1.6-vicuna-13b': 'liuhaotian/llava-v1.6-vicuna-13b-hf',
        'liuhaotian/llava-v1.6-34b-tokenizer': 'liuhaotian/llava-v1.6-34b-tokenizer',
        'lmms-lab/llama3-llava-next-8b': None,
        'lmms-lab/llava-onevision-qwen2-7b-ov': None,
        }
    model_name = get_model_name_from_path(model_dir)
    if model_name not in model_map.keys():
        raise ValueError(f'Unknown model. Please specify base model for {model_name}')
    return model_map[model_name]

def get_conv_mode(model_name):
    model_map = {
        'LlavaGuard-v1.1-7b': "v1",
        'LlavaGuard-v1.1-13b': "v1",
        'LlavaGuard-v1.2-13b': "v1",
        'LlavaGuard-v1.2-34b': "chatml_direct",
        'LlavaGuard-v1.2-7b-ov': "qwen_1_5",
        'LlavaGuard-v1.2-7b-ov-chat': "qwen_1_5",
        'LlavaGuard-v1.2-8b': "llava_llama_3",
        'liuhaotian/llava-v1.5-7b': "v1",
        'liuhaotian/llava-v1.5-13b': "v1",
        'liuhaotian/v1.6-vicuna-7b': "v1",
        'liuhaotian/v1.6-vicuna-13b': 'liuhaotian/llava-v1.6-vicuna-13b-hf',
        'liuhaotian/llava-v1.6-34b-tokenizer':"chatml_direct",
        'lmms-lab/llama3-llava-next-8b': "llava_llama_3",
        'lmms-lab/llava-onevision-qwen2-7b-ov': "qwen_1_5",
        'lmms-lab/llava-onevision-qwen2-7b-ov-chat': "qwen_1_5",
        }
    if model_name in model_map.keys():
        conv_mode = model_map[model_name]
    else:
        raise ValueError(f'Unknown conv. Please specify conv for model: {model_name}')
    return conv_mode


def get_sglang_template(model_name):
    if '34b' in model_name or 'ov' in model_name:
        chat_template = "chatml-llava"
    elif '8b' in model_name:
        chat_template = 'llava_llama_3'
    else:
        chat_template = None
    return chat_template



def get_conv(model_name):
    from llava.conversation import conv_templates
    conv_mode = get_conv_mode(model_name)
    return copy.deepcopy(conv_templates[conv_mode])






def restruct_im_folder_to_crawled_data():

    import glob
    import os
    import shutil
    import pandas as pd
    from llavaguard_config import local_image_dirs, human_feedback_dirs
    im_counter = 0
    smid_paths = glob.glob(local_image_dirs['smid'] + '/*.jpg')
    real_image_paths = glob.glob(local_image_dirs['real_images'] + '/*/*.jpg')
    synthetic_image_paths = glob.glob(local_image_dirs['synthetic'] + '/*/*.jpeg')
    print(f"Found {len(real_image_paths)} real images")
    print(f"Found {len(synthetic_image_paths)} synthetic images")
    print(f"Found {len(smid_paths)} SMID images")
    
    annot_dir_smid = human_feedback_dirs['smid']
    annotation_paths = glob.glob(annot_dir_smid + '/*.csv')
    annotations_smid = pd.concat([pd.read_csv(path) for path in annotation_paths])
    print(f"Found {len(annotations_smid)} annotations for SMID images")
    
    # for real_image_path in real_image_paths:
    #     # move the image to the destination directory
    #     im_dest = real_image_path.replace('real_images', 'crawled_data').split('image_')[0] + 'image_' + str(im_counter) + '.jpg'
    #     os.makedirs(os.path.dirname(im_dest), exist_ok=True)
    #     shutil.copy(real_image_path, im_dest)
    #     # print(f"Moved {im_counter} images")
    #     im_counter += 1
    
    for real_im_dir in  glob.glob(human_feedback_dirs['real_images'] + '/*'):
        annot_dirs = glob.glob(real_im_dir + '/*.csv')
        annot = pd.concat([pd.read_csv(path) for path in annot_dirs])
        # add annotation directory to the annotation
        print(real_im_dir)
        annot['im_dir'] = real_im_dir.replace('annotations', 'images')
        print(annot)
        
        
        # filter out score == Discard Sample
        annot = annot[annot['score'] != 'Discard Sample']
        # iterate over each row in the annotation
        for sample in annot.iterrows():
            print(f"Processing s {im_counter}")
            json_name = sample[1]['json']
            image_name = 'image_' + json_name.split('_')[1].replace('tt.json', '') + '.jpg'
            
            
            print(f"Real image name: {image_name}")
            print(f"Image path: {sample[1]['im_dir']}")
            real_image_path = os.path.join(sample[1]['im_dir'], image_name)
            # move the image to the destination directory
            im_dest = real_image_path.replace('real_images', 'crawled_data').split('image_')[0] + 'image_' + str(im_counter) + '.jpg'
            os.makedirs(os.path.dirname(im_dest), exist_ok=True)
            shutil.copy(real_image_path, im_dest)
            # print(f"Moved {im_counter} images")
            im_counter += 1
    
    print(f"Total number of images: {im_counter}")