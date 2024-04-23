import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

if '/workspace' not in sys.path:
    sys.path.insert(0, '/workspace')


def prepare_instruct_tuning_data_no_policy():
    import glob
    import os
    data_json = []
    dataset_name = 'smid_and_real_images_human_feedback'
    print(f'Preparing data {dataset_name} for LlavaGuard instructive tuning')
    ds_out = f'/common-repos/LlavaGuard/data/{dataset_name}'
    os.makedirs(ds_out, exist_ok=True)
    os.chmod(ds_out, 0o777)
    smid_images = '/common-repos/MultimodalExplanationLearning/datasets/SMID_images_400px/img'
    smid_feedback = '/workspace/data/smid_llava_guard_samplingv1_v1.5-13b_constrained_humanfeedback'
    shards = glob.glob(f'{smid_feedback}/*.csv')
    training_question_old = (
        "<image>\nAssess the inappropriateness of the content and classify if it is compliant to policies. "
        "Reply with one of the following labels: <\"Explicitly Inappropriate\", "
        "\"Highly Inappropriate\", \"Moderately Inappropriate\", \"Barely Acceptable\", "
        "\"Generally Acceptable\", \"Suitable & Respectful\", \"Positive & Engaging\", "
        "\"Exemplary & Inspiring\">")
    categories = ['None',
                  'Peace or Care',
                  'Respectful or Appropriate Content',
                  'Polite Communication and Gestures',
                  'Nature',
                  'Explicit Sexual Content',
                  'Nudity',
                  'Suicide or Self Harm',
                  'Substance Abuse or Weapons',
                  'Illegal Activities',
                  'Humiliation, Harassment or Hate',
                  'Disasters or Emergencies',
                  'Animal Cruelty',
                  'Violence, Harm or Cruelty', ]
    categories_prompt = ', '.join([f'"{x}"' for x in categories])
    scores_label = ["Highly Inappropriate", "Moderately Inappropriate", "Barely Acceptable", "Generally Acceptable"]
    training_question = (
        "<image>\nAssess the inappropriateness of the content and classify if it is compliant to policies. "
        "First Categorize the image's content into one of the following categories: "
        f"{categories_prompt}. "
        "Then, assess the inappropriateness of the content using one of the following labels: "
        "<\"Highly Inappropriate\", \"Moderately Inappropriate\", \"Barely Acceptable\", \"Generally Acceptable\">")

    for shard in shards:
        df = pd.read_csv(shard)
        for i, row in df.iterrows():
            sample = {}
            # remove last 2 characters from json name
            sample['id'] = row['json'].split(".")[0][:-2]
            try:
                im_path = glob.glob(f'{smid_images}/{sample["id"]}.*')[0]
                sample['image'] = im_path
            except FileNotFoundError:
                raise FileNotFoundError(f'Missing image: {sample["image"]}')
            sample['conversations'] = [
                {
                    "from": "human",
                    "value": training_question
                },
                {
                    "from": "gpt",
                    "value": row['score']
                }
            ]
            data_json.append(sample)

    real_images = '/common-repos/LlavaGuard/real_images_preselected_renamed'
    real_images_feedback = '/workspace/data/smid_llava_guard_samplingv1_v1.5-13b_constrained_real_images_v2_humanfeedback'
    shards = glob.glob(f'{real_images_feedback}/*/*.csv')
    for shard in shards:
        df = pd.read_csv(shard)
        image_folder = shard.split('/')[-2]
        for i, row in df.iterrows():
            sample = {}
            sample['id'] = row['json'].split('.')[0][:-2]
            sample['image'] = f'{real_images}/{image_folder}/{sample["id"]}.jpg'
            if not os.path.exists(sample['image']):
                raise FileNotFoundError(f'Missing image: {sample["image"]}')
            sample['conversations'] = [
                {
                    "from": "human",
                    "value": training_question
                },
                {
                    "from": "gpt",
                    "value": row['score']
                }
            ]
            data_json.append(sample)

    # save data as json
    with open(f'{ds_out}/all_data.json', 'w') as a, open(f'{ds_out}/train.json', 'w') as t, open(
            f'{ds_out}/eval.json', 'w') as v:
        json.dump(data_json, a, indent=4)
        data = train_test_split(data_json, test_size=0.1, random_state=42)
        json.dump(data[0], t, indent=4)
        json.dump(data[1], v, indent=4)