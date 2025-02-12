import json
import pickle
import random
import argparse
import shutil
import sys
from itertools import product

import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import os

from llavaguard.taxonomy.assessment import get_mapping
from llavaguard_config import llava_guard_config, \
    generated_rationales_dirs, local_image_dirs, human_feedback_dirs
from llavaguard.taxonomy.augmentation import create_samples_with_augmented_policies
from llavaguard.evaluation.metrics_calculator import get_keys, get_rating_values
from plots.ds_heatmap.dataset_heatmap import plot_ds_heatmap
from llavaguard_config import local_image_dirs, local_data_dir

def get_prompt_and_gt(data: dir):
    prompt = data["conversations"][0]["value"]
    gt = data["conversations"][1]["value"]
    # gt = gt if isinstance(gt, dict) else json.loads(gt)
    return prompt, gt


def get_safety_rating(data: dir):
    gt = data["conversations"][1]["value"]
    gt = json.loads(gt)
    _, _, rating_key = get_keys(gt)
    return gt[rating_key]


def get_safety_category(data: dir):
    gt = data["conversations"][1]["value"]
    gt = json.loads(gt)
    category_key, _, _ = get_keys(gt)
    if category_key is None:
        cat = data["category"]
        return cat
    return gt[category_key]


def get_rationale(data: dir):
    gt = data["conversations"][1]["value"]
    gt = json.loads(gt)
    _, rationale_key, _ = get_keys(gt)
    return gt[rationale_key]


def get_gt_assessment(data: dir):
    gt = data["conversations"][1]["value"]
    gt = json.loads(gt)
    return gt


def get_id(data: dir):
    return data["id"]


def get_safety_score_map():
    return {
        'Generally Safe': 'Generally Acceptable',
        'Barely Safe': 'Barely Acceptable',
        'Moderately Unsafe': 'Moderately Inappropriate',
        'Highly Unsafe': 'Highly Inappropriate'
    }


def oversample_minority_class(data):
    ex = data[0]['conversations'][1]["value"]
    _, _, rating_key = get_keys(ex)
    vals = get_rating_values(rating_key)

    compliant = [x for x in data if get_safety_rating(x) == vals[0]]
    len_compliant = len(compliant)
    review_needed = [x for x in data if get_safety_rating(x) == vals[1]]
    len_review_needed = len(review_needed)
    minority_class = compliant if len_compliant < len_review_needed else review_needed
    missing_samples = abs(len_compliant - len_review_needed)
    # oversample the minority class
    data = data.copy()
    data.extend(
        minority_class * (missing_samples // len(minority_class)) + minority_class[
                                                                    :missing_samples % len(minority_class)])
    # random shuffle the train data
    random.shuffle(data)
    return data


def filer_for_only_default_policy(data):
    filtered_data = []
    for sample in data:
        if '_v1' not in sample['id'] and '_v2' not in sample['id'] and '_v3' not in sample['id'] and '_v4' not in sample['id'] and '_v5' not in sample['id']:
            filtered_data.append(sample)
    return filtered_data
    


def filter_score(data, scores=None):
    if scores is None:
        return data
    return [x for x in data if x['score'] in scores]



def custom_serializer(obj):
    if callable(obj):
        return f"<function {obj.__name__}>"
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def split_data(data, test_size:int):
    if test_size > len(data):
        raise ValueError(f'Test size ({test_size}) is greater than the length of the data ({len(data)})')
    if test_size == 0:
        return data, []
    if test_size == len(data):
        return [], data
    return train_test_split(data, test_size=test_size, random_state=42)

def get_llava_prediction_dir(template_version):
    rationale_version = llava_guard_config[template_version]['rationale_version']
    if rationale_version not in generated_rationales_dirs.keys():
        raise ValueError(f'Invalid rationale version: {rationale_version}')
    return generated_rationales_dirs[rationale_version]


def load_smid(count_valid_explanations, template_version):
    # Load SMID data
    llava_prediction_dir = get_llava_prediction_dir(template_version)
    augmentation_strategy = llava_guard_config[template_version]['augmentation_strategy']
    smid_images = local_image_dirs['smid']
    smid_feedback = human_feedback_dirs['smid']
    shards = glob.glob(f'{smid_feedback}/*.csv')
    smid_scores = []
    smid_ratings = []
    smid_ids = []
    rating_vals = None
    smid_data = pd.DataFrame([], columns=['data', 'category', 'score'])
    for shard in shards:
        df_smid = pd.read_csv(shard)
        for i, row in df_smid.iterrows():
            s_id = row['json'].split(".")[0][:-2]
            pred_path = llava_prediction_dir['smid'] if isinstance(llava_prediction_dir, dict) else llava_prediction_dir
            samples = create_samples_with_augmented_policies(row, smid_images, pred_path, template_version,
                                                             augmentation_strategy, counter=count_valid_explanations)
            if len(samples) == 0:
                continue
            for s in samples:
                smid_ratings.append(get_safety_rating(s))
                smid_ids.append(s['id'])
            category = get_mapping(template_version)[str(row['category'])]
            score = row['score']
            smid_scores.extend([score] * len(samples))
            smid_data = pd.concat([smid_data, pd.DataFrame([[samples, category, score]], columns=['data', 'category', 'score'])])
    return smid_data, smid_scores, smid_ratings, smid_ids


def load_real_images(count_valid_explanations, template_version):
    llava_prediction_dir = get_llava_prediction_dir(template_version)
    augmentation_strategy = llava_guard_config[template_version]['augmentation_strategy']
    # Load Real Images data
    real_images = local_image_dirs['real_images']
    real_images_feedback = human_feedback_dirs['real_images']
    shards = glob.glob(f'{real_images_feedback}/*/*.csv')
    real_paths = []
    real_ids2 = []
    real_data = pd.DataFrame([], columns=['data', 'category', 'score'])
    real_scores, real_ratings, real_ids = [], [], []
    for shard in shards:
        df_ri = pd.read_csv(shard)
        image_folder = shard.split('/')[-2]
        for i, row in df_ri.iterrows():
            image_f = f'{real_images}/{image_folder}'
            pred_path = llava_prediction_dir['real_images'] if isinstance(llava_prediction_dir,
                                                                          dict) else llava_prediction_dir
            # data = row.to_dict()
            # data['score'] = get_safety_score_map()[data['score']]
            samples = create_samples_with_augmented_policies(row, image_f, pred_path, template_version,
                                                             augmentation_strategy,
                                                             counter=count_valid_explanations)
            if len(samples) == 0:
                continue
            real_paths.append(samples[0]['image'])
            real_ids2.append(samples[0]['id'])
            category = get_mapping(template_version)[str(row['category'])]
            score = row['score']
            for s in samples:
                real_ratings.append(get_safety_rating(s))
                real_ids.append(s['id'])

            real_scores.extend([score] * len(samples))
            real_data = pd.concat([real_data, pd.DataFrame([[samples, category, score]], columns=['data', 'category', 'score'])])
        
    # split in 4 chunks and copy images
    # chunk_size = len(real_paths) // 4
    # for i in range(4):
    #     chunk = real_paths[i * chunk_size: (i + 1) * chunk_size] if i < 3 else real_paths[i * chunk_size:]
    #     chunk_ids = real_ids2[i * chunk_size: (i + 1) * chunk_size] if i < 3 else real_ids2[i * chunk_size:]
    #     os.makedirs(f'{local_data_dir}/data/urls/{i}', exist_ok=True)
    #     for im, id in zip(chunk, chunk_ids):
    #         shutil.copy(im, f'{local_data_dir}/data/urls/{i}/{id}.jpg')
    #     #  create csv file with ids and empty urls
    #     with open(f'{local_data_dir}/data/urls/urls_{i}.csv', 'w') as f:
    #         f.write('id,url\n')
    #         for id in chunk_ids:
    #             f.write(f'{id},\n')
    return real_data, real_scores, real_ratings, real_ids


def load_sythetic(count_valid_explanations, template_version):
    llava_prediction_dir = get_llava_prediction_dir(template_version)
    augmentation_strategy = llava_guard_config[template_version]['augmentation_strategy']
    synthetic_images = local_image_dirs['synthetic']
    data_hf = human_feedback_dirs['synthetic']
    all_annot = glob.glob(f'{data_hf}/*.json')
    sythetic_data = pd.DataFrame([], columns=['data', 'category', 'score'])
    sythetic_scores, sythetic_ratings, sythetic_ids = [], [], []
    for file in all_annot:
        row = json.load(open(file, 'r'))
        category = row['category']
        if row['rating'] == 'Discard':
            continue
        score = get_safety_score_map()[row['rating']]
        id = file.split('/')[-1].split('.')[0]
        image = glob.glob(f'{synthetic_images}/*/{id}.jpeg')[0]
        samples = create_samples_with_augmented_policies({'category': category, 'score': score, 'id': id},
                                                         image, llava_prediction_dir, template_version,
                                                         augmentation_strategy, counter=count_valid_explanations)
        if len(samples) == 0:
            continue
        for s in samples:
            sythetic_ratings.append(get_safety_rating(s))
            sythetic_ids.append(s['id'])
        sythetic_scores.extend([score] * len(samples))
        sythetic_data = pd.concat([sythetic_data, pd.DataFrame([[samples, category, score]], columns=['data', 'category', 'score'])])
    return sythetic_data, sythetic_scores, sythetic_ratings, sythetic_ids

def prepare_instruct_tuning_with_policy_augmentation(version: int, ds_out: str):
    '''
    Prepare Humanfeedback dataset for instruction tuning with/without policy augmentation.
    :param template_version: Version of the template to use. Options: nl, json, json-v1, json-v2, ..., json-v8
    :param ds_out: Output directory to save the dataset
    1. We drop a random number of categories from the taxonomy that are not violated in the given example.
    2. We drop the violation category from the model prompt changing the safety label to “Compliant”.
    We then use the original and augmented examples to train and evaluate the model.
    Finally we save the dataset to the output dir
    '''
    # lg_v = int(template_version.split('-v')[-1].split('-')[0])
    template_version = f'json-v{version}'
    augmentation_strategy = llava_guard_config[template_version]['augmentation_strategy']
    datasets = llava_guard_config[template_version]['datasets']
    llava_prediction_dir = get_llava_prediction_dir(template_version)
    rationale_version = llava_guard_config[template_version]['rationale_version']
    
    ds_out += "" if augmentation_strategy is not None else "_no_augmentation"  # add with_augmented_policies to the directory name if augmentation is True
    ds_out += f'/v{version}'  # add template version to the directory name

    os.makedirs(ds_out, exist_ok=True)
    os.chmod(ds_out, 0o777)
    train_data_name, val_data_name, test_data_name = f'train', f'eval', f'test'


    # save data as json


    # if already exists, return
    if os.path.exists(f'{ds_out}/{val_data_name}.json') and os.path.exists(f'{ds_out}/{train_data_name}.json'):
        # and os.path.exists(f'{ds_out}/{test_data_name}.json')):
        print(f'Dataset already exists at: {ds_out} ({train_data_name}.json and {val_data_name}.json)')
        print('skipping dataset preparation')
        print('#################################################################################################')
        return
    data = pd.DataFrame([], columns=['data', 'category', 'score'])
    count_valid_explanations = [0, 0]
    all_scores, all_ratings, all_ids = [], [], []
    
    if 'smid' in datasets:
        smid_data, smid_scores, smid_ratings, smid_ids = load_smid(count_valid_explanations, template_version)
        data = pd.concat([data, smid_data])
        all_scores.extend(smid_scores)
        all_ratings.extend(smid_ratings)
        all_ids.extend(smid_ids)      

    if 'real_images' in datasets:
        real_data, real_scores, real_ratings, real_ids = load_real_images(count_valid_explanations, template_version)
        data = pd.concat([data, real_data])
        all_scores.extend(real_scores)
        all_ratings.extend(real_ratings)
        all_ids.extend(real_ids)
    
    if 'synthetic' in datasets:
        synthetic_data, synthetic_scores, synthetic_ratings, synthetic_ids = load_sythetic(count_valid_explanations, template_version)
        data = pd.concat([data, synthetic_data])
        all_scores.extend(synthetic_scores)
        all_ratings.extend(synthetic_ratings)
        all_ids.extend(synthetic_ids)
        

    # copy images to /common-repos/LlavaGuard/data/urls and
    # os.makedirs(f'{ds_out}/urls', exist_ok=True)
    # os.chmod(f'{ds_out}/urls', 0o777)




    train_split, validation_split, test_split = [], [], []
    categories, scores = data['category'].unique(), data['score'].unique()
    print('Splitting data into train, validation and test sets')

    for category, score in product(categories, scores):
        subset = data[(data['category'] == category) & (data['score'] == score)]['data'].values
        if len(subset) == 0:
            continue
        samples_per_image = len(subset[0])
        num_images = len(subset)
        total_num_samples = len(subset) * samples_per_image
        test_size, val_size = samples_per_image, samples_per_image
        if 'None applying' in category:
            test_size, val_size = 100, 5
        elif 'Acceptable' in score:
            test_size, val_size = 10, 1
        else:
            test_size = 5
            if 'Highly Inappropriate' in score:
                val_size = 1
            else:
                val_size = 0
        test_size = min(test_size, num_images)
        val_size = min(val_size, num_images - test_size)
        print(f'Category: {category}, Score: {score}, Total samples: {num_images*samples_per_image}, Test size: {test_size*samples_per_image}, Val size: {val_size*samples_per_image}, Train size: {(num_images - test_size - val_size)*samples_per_image}')
        train, test = split_data(subset, test_size=test_size)
        train, val = split_data(train, test_size=val_size)
        validation_split.extend(val)
        train_split.extend(train)
        test_split.extend(test)


    rating_vals = list(set(all_ratings))
    # print shape of train and test data
    # eval_split = [item for sublist in test_set for item in sublist]
    # train_split = [item for sublist in train_set for item in sublist]
    # print_txt += f'All data: ({len(data)}), Train data: ({len(train_set)}), Eval data: ({len(test_set)})')
    with (open(f'{ds_out}/all_data.json', 'w') as all_json,
          open(f'{ds_out}/{train_data_name}.json', 'w') as train_json,
          open(f'{ds_out}/{val_data_name}.json', 'w') as val_json,
          open(f'{ds_out}/{test_data_name}.json','w') as test_json,
          open(f'{ds_out}/{train_data_name}_oversampled.json', 'w') as train_oversampled_json,
          open(f'{ds_out}/{train_data_name}_default_policy.json', 'w') as train_default_policy_json):
        flattened_train_split = [item for sublist in train_split for item in sublist]
        flattened_train_split_oversampled = oversample_minority_class(flattened_train_split)
        flattened_train_split_default_policy = oversample_minority_class([item[0] for item in train_split])
        flattened_val_split = [item for sublist in validation_split for item in sublist]
        flattened_test_split = [item for sublist in test_split for item in sublist]
        all_data = flattened_train_split + flattened_val_split + flattened_test_split
        json.dump(all_data, all_json, indent=4)
        json.dump(flattened_train_split, train_json, indent=4)
        json.dump(flattened_train_split_oversampled, train_oversampled_json, indent=4)
        json.dump(flattened_train_split_default_policy, train_default_policy_json, indent=4)
        json.dump(flattened_test_split, test_json, indent=4)
        json.dump(flattened_val_split, val_json, indent=4)
    # print data statistics
    v1, v2, v3_safe, v3_unsafe, v4, v5_safe, v5_unsafe = 0, 0, 0, 0, 0, 0, 0
    safe, unsafe = 0, 0
    for i, score in zip(all_ids, all_scores):
        if '_v1' in i:
            v1 += 1
        elif '_v2' in i:
            v2 += 1
        elif '_v3' in i and 'Acceptable' in score:
            v3_safe += 1
        elif '_v3' in i and 'Inappropriate' in score:
            v3_unsafe += 1
        elif '_v4' in i:
            v4 += 1
        elif '_v5' in i and 'Acceptable' in score:
            v5_safe += 1
        elif '_v5' in i and 'Inappropriate' in score:
            v5_unsafe += 1
        elif 'Acceptable' in score:
            safe += 1
        elif 'Inappropriate' in score:
            unsafe += 1
            
    print_txt = f'''
#################################################################################################
DA Info:
Prepared dataset for instruction tuning using dataset sources: {datasets} from dir {ds_out}
Template version: {template_version}, Oversampling: Auto, Policy augmentation strategry: {augmentation_strategy}
Explanation taken from: {llava_prediction_dir} (version {rationale_version})
Unique images: {len(data)} ({len(data[data["score"] == "Generally Acceptable"])} Generally Acceptable, {len(data[data["score"] == "Barely Acceptable"])} Barely Acceptable, {len(data[data["score"] == "Moderately Inappropriate"])} Moderately Inappropriate, {len(data[data["score"] == "Highly Inappropriate"])} Highly Inappropriate)
Number of images in each category: {data.groupby(["category"]).size()}
#################################################################################################
All data: {len(all_data)} ({rating_vals[0]} {all_ratings.count(rating_vals[0])}, {rating_vals[1]} {all_ratings.count(rating_vals[1])})
Default policy {safe + unsafe} ({safe} safe, {unsafe} unsafe), Augmented Policy v1 ({v1}), v2 ({v2} unsafe -> safe), v3 ({v3_safe} safe, {v3_unsafe} unsafe), v4 ({v4} safe -> unsafe), v5 ({v5_safe} safe, {v5_unsafe} unsafe)
General Acceptable: {all_scores.count("Generally Acceptable")}, Barely Acceptable: {all_scores.count("Barely Acceptable")}, Moderately Inappropriate: {all_scores.count("Moderately Inappropriate")}, Highly Inappropriate: {all_scores.count("Highly Inappropriate")}
Train data: (unique {len(flattened_train_split)} -> oversampled {len(flattened_train_split_oversampled)}),
Eval data: ({len(flattened_val_split)}), Test data: ({len(flattened_test_split)})
Valid explanations: ({count_valid_explanations[0]}/{count_valid_explanations[1] + count_valid_explanations[0]})
Train Data saved at: {ds_out}/{train_data_name}.json
Eval Data saved at: {ds_out}/{val_data_name}.json
Test Data saved at: {ds_out}/{test_data_name}.json
#################################################################################################
'''
    print(print_txt)
    # save ds info
    with open(f'{ds_out}/ds_info.txt', 'w') as f:
        f.write(print_txt)
    with open(f'{ds_out}/ds_config.txt', 'w') as f:
        json.dump(llava_guard_config[template_version], f, default=custom_serializer, indent=4)
    plot_ds_heatmap(f'{ds_out}/all_data.json')
    plot_ds_heatmap(f'{ds_out}/{train_data_name}.json')
    plot_ds_heatmap(f'{ds_out}/{train_data_name}_oversampled.json')
    plot_ds_heatmap(f'{ds_out}/{train_data_name}_default_policy.json')
    plot_ds_heatmap(f'{ds_out}/{val_data_name}.json')
    plot_ds_heatmap(f'{ds_out}/{test_data_name}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for LlavaGuard instructive tuning with policy')
    parser.add_argument('--version', type=int, default=22,
                        help='LlavaGuard version to use for preparing the dataset')
    parser.add_argument('--ds_dir', default=f'{local_data_dir}/data/LlavaGuard-DS',
                        help='output directory to save the dataset')
    args = parser.parse_args()
    prepare_instruct_tuning_with_policy_augmentation(args.version, args.ds_dir)
