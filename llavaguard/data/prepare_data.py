import json
import random
import argparse
import shutil
import sys
from itertools import product

import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import os

if '/workspace' not in sys.path:
    sys.path.insert(0, '/workspace')
from llavaguard.taxonomy.assessment import get_mapping
from llavaguard.taxonomy.policies import get_assessment_and_system_prompt
from llavaguard.taxonomy.augmentation import create_samples_with_augmented_policies
from llavaguard.evaluation_metrics_calculator import get_keys, get_rating_values
from convert_model import get_safety_rating
from plots.dataset_heatmap import plot_ds_heatmap


def oversample_minority_class(data):
    ex = data[0]['conversations'][1]["value"]
    _, _, rating_key = get_keys(ex)
    vals = get_rating_values(rating_key)

    compliant = [x for x in data if json.loads(x['conversations'][1]["value"])[rating_key] == vals[0]]
    len_compliant = len(compliant)
    review_needed = [x for x in data if json.loads(x['conversations'][1]["value"])[rating_key] == vals[1]]
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


def filter_score(data, scores=None):
    if scores is None:
        return data
    return [x for x in data if x['score'] in scores]


def prepare_instruct_tuning_with_policy(template_version='json-v2', remove_edge_cases=False):
    '''
    Prepare Humanfeedback dataset for instructive tuning with policy
    :param template_version: Version of the template to use. Options: json, json-v1, json-v2, json-v3, json-v4, nl
    :param oversampled: If oversampling, the minority class will be oversampled to balance the dataset.
    :param remove_edge_cases: If remove_edge_cases, the edge cases will be removed from the dataset.
     The dataset will only contain samples that are clearly compliant or review needed.
    :param ds_name: Name of the dataset. If provided, template_version and sampling will be ignored.
    :return:
    '''

    # if ds_name is not None:
    #     if template_version is None and sampling is None:
    #         sampling = ds_name.split('_')[-1]
    #         template_version = ds_name.split('_')[-2]
    #     else:
    #         raise ValueError('Do not provide template_version and sampling if ds_name is provided')
    ds_out = f'/common-repos/LlavaGuard/data/smid_and_crawled_policy/{template_version}'
    os.makedirs(ds_out, exist_ok=True)
    os.chmod(ds_out, 0o777)
    train_data_name = f'train'
    eval_data_name = f'eval'
    if remove_edge_cases:
        train_data_name += '_no_edge_cases'
        eval_data_name += '_no_edge_cases'

    # save data as json

    print(f'Preparing dataset for instructive tuning:')
    print(f'Using SMID and crawled images with Humanfeedback')
    print(f'Dataset with policy, Template version: {template_version}, Oversampling: Auto,'
          f' Remove edge cases: {remove_edge_cases}')
    # if already exists, return
    # if os.path.exists(f'{ds_out}/{eval_data_name}.json') and os.path.exists(f'{ds_out}/{train_data_name}.json') and \
    #         os.path.exists(f'{ds_out}/test.json'):
    #     print(f'Dataset already exists at: {ds_out} ({train_data_name}.json and {eval_data_name}.json)')
    #     print('skipping dataset preparation')
    #     print('#################################################################################################')
    #     return
    data = []
    test_data = []
    smid_prediction = '/workspace/data/smid_llava_guard_samplingv1_v1.5-13b_constrained'
    real_images_prediction = '/workspace/data/smid_llava_guard_samplingv1_v1.5-13b_constrained_real_images_v2'

    smid_images = '/common-repos/MultimodalExplanationLearning/datasets/SMID_images_400px/img'
    smid_feedback = '/workspace/data/smid_llava_guard_samplingv1_v1.5-13b_constrained_humanfeedback'
    shards = glob.glob(f'{smid_feedback}/*.csv')
    assessment, system_prompt = get_assessment_and_system_prompt(template_version)
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

            try:
                prediction = json.load(open(f"{smid_prediction}/{row['json']}"))
            except FileNotFoundError:
                raise FileNotFoundError(f'Missing prediction: {row["json"]}')
            sample['final-assessment'] = 'Compliant' if 'Acceptable' in row['score'] else 'Review Needed'
            sample['score'] = row['score']
            sample['category'] = row['category']
            explanation = prediction['explanation'] if prediction['score'] == row['score'] else None
            sample['conversations'] = [
                {
                    "from": "human",
                    "value": system_prompt
                },
                {
                    "from": "gpt",
                    "value": assessment(row['score'], row['category'], explanation)
                }
            ]
            data.append(sample)

    real_images = '/common-repos/LlavaGuard/real_images_preselected_renamed'
    real_images_feedback = '/workspace/data/smid_llava_guard_samplingv1_v1.5-13b_constrained_real_images_v2_humanfeedback'
    shards = glob.glob(f'{real_images_feedback}/*/*.csv')
    for shard in shards:
        df = pd.read_csv(shard)
        image_folder = shard.split('/')[-2]
        for i, row in df.iterrows():
            sample = {}
            im_name = row['json'].split('.')[0][:-2]
            sample['id'] = image_folder.replace(' ', '_') + '_' + im_name
            sample['image'] = f'{real_images}/{image_folder}/{im_name}.jpg'
            sample['final-assessment'] = 'Compliant' if 'Acceptable' in row['score'] else 'Review Needed'
            sample['score'] = row['score']
            sample['category'] = row['category']
            try:
                prediction = json.load(open(f'{real_images_prediction}/{image_folder}/{row["json"]}'))
                explanation = prediction['explanation'] if prediction['score'] == row['score'] else None
            except FileNotFoundError:
                explanation = None

            if not os.path.exists(sample['image']):
                raise FileNotFoundError(f'Missing image: {sample["image"]}')
            sample['conversations'] = [
                {
                    "from": "human",
                    "value": system_prompt
                },
                {
                    "from": "gpt",
                    "value": assessment(row['score'], row['category'], explanation)
                }
            ]
            if 'missing' == image_folder:
                test_data.append(sample)
            else:
                data.append(sample)

    with (open(f'{ds_out}/all_data.json', 'w') as a, open(f'{ds_out}/{train_data_name}.json', 'w') as t,
          open(f'{ds_out}/{eval_data_name}.json', 'w') as v,
          open(f'{ds_out}/{train_data_name}_oversampled.json', 'w') as o):
        fdata = filter_score(data, ['Highly Inappropriate', 'Generally Acceptable']) if remove_edge_cases else data
        train_split, test_split = train_test_split(fdata, test_size=0.1, random_state=42)
        test_split.extend(test_data)
        json.dump(train_split, t, indent=4)
        json.dump(test_split, v, indent=4)
        train_split_oversampled = oversample_minority_class(train_split)
        json.dump(train_split_oversampled, o, indent=4)
        data.extend(test_data)
        json.dump(data, a, indent=4)
    print(f'Dataset saved at: {ds_out}/{train_data_name}.json and {ds_out}/{eval_data_name}.json')
    print('#################################################################################################')


def prepare_instruct_tuning_with_policy_augmentation(template_version='json-v9', augmentation=True, explanations='v2'):
    '''
    Prepare Humanfeedback dataset for instruction tuning with/without policy augmentation.
    :param template_version: Version of the template to use. Options: nl, json, json-v1, json-v2, ..., json-v8
    :param explanations: Version of the model predictions to use. Options: v1, v2. v1 uses llava-v1.5-13b,
     v2 uses llava-v1.6-34b
    :param remove_edge_cases: If remove_edge_cases, the edge cases will be removed from the dataset.
     The dataset will only contain samples that are clearly compliant or review needed.
     :param augmentation: If augmentation, we employ policy augmentation. We apply two augmentation techniques
     to the unsafe examples:
    1. We drop a random number of categories from the taxonomy that are not violated in the given example.
    2. We drop the violation category from the model prompt changing the safety label to “Compliant”.
    We then use the original and augmented examples to train and evaluate the model.
    :return:
    '''
    prediction_model = {
        'v1': 'llava-v1.5-13b',
        'v2': 'llava-v1.6-34b',
    }[explanations]

    ds_out = f'/common-repos/LlavaGuard/data/smid_and_crawled'
    ds_out += "_v2" if explanations == 'v2' else ""  # add v2 to the directory name if explanations is v2
    ds_out += "_with_augmented_policies" if augmentation else "_policy"  # add with_augmented_policies to the directory name if augmentation is True
    ds_out += f'/{template_version}'  # add template version to the directory name
    os.makedirs(ds_out, exist_ok=True)
    os.chmod(ds_out, 0o777)
    train_data_name = f'train'
    eval_data_name = f'eval'
    test_data_name = f'test'

    llava_16_34b_json_v8_prediction = '/common-repos/LlavaGuard/eval/llava-v1.6-34b/foundation_model/smid_and_crawled_policy-json-v8/model_output'
    # save data as json

    print(f'Preparing dataset for instruction tuning using SMID and crawled images with Humanfeedback, '
          f'Explanations taken from: {prediction_model}')
    print(f'Template version: {template_version}, Oversampling: Auto, Policy augmentation: {augmentation}')
    print('Dataset directory:', ds_out)
    # if already exists, return
    if os.path.exists(f'{ds_out}/{eval_data_name}.json') and os.path.exists(f'{ds_out}/{train_data_name}.json'):
        # and os.path.exists(f'{ds_out}/{test_data_name}.json')):
        print(f'Dataset already exists at: {ds_out} ({train_data_name}.json and {eval_data_name}.json)')
        print('skipping dataset preparation')
        print('#################################################################################################')
        return
    data = pd.DataFrame([], columns=['data', 'category', 'score'])
    test_data = []
    smid_images = '/common-repos/MultimodalExplanationLearning/datasets/SMID_images_400px/img'
    smid_feedback = '/workspace/data/smid_llava_guard_samplingv1_v1.5-13b_constrained_humanfeedback'
    smid_prediction = '/workspace/data/smid_llava_guard_samplingv1_v1.5-13b_constrained'
    shards = glob.glob(f'{smid_feedback}/*.csv')
    count_valid_explanations = [0, 0]
    all_scores = []
    all_ratings = []
    all_ids = []
    rating_vals = None
    for shard in shards:
        df_smid = pd.read_csv(shard)
        for i, row in df_smid.iterrows():

            s_id = row['json'].split(".")[0][:-2]
            pred_path = smid_prediction if prediction_model == 'llava-v1.5-13b' else llava_16_34b_json_v8_prediction
            samples = create_samples_with_augmented_policies(row, smid_images, pred_path, template_version,
                                                             augmentation, counter=count_valid_explanations)
            if len(samples) == 0:
                continue
            for s in samples:
                all_ratings.append(get_safety_rating(s))
                all_ids.append(s['id'])


            category = get_mapping(template_version)[str(row['category'])]


            score = row['score']
            all_scores.extend([score] * len(samples))
            data = pd.concat([data, pd.DataFrame([[samples, category, score]], columns=['data', 'category', 'score'])])
    real_images = '/common-repos/LlavaGuard/real_images_preselected_renamed'
    real_images_feedback = '/workspace/data/smid_llava_guard_samplingv1_v1.5-13b_constrained_real_images_v2_humanfeedback'
    real_images_prediction = '/workspace/data/smid_llava_guard_samplingv1_v1.5-13b_constrained_real_images_v2'
    shards = glob.glob(f'{real_images_feedback}/*/*.csv')
    real_paths = []
    real_ids = []
    for shard in shards:
        df_ri = pd.read_csv(shard)
        image_folder = shard.split('/')[-2]
        for i, row in df_ri.iterrows():
            image_f = f'{real_images}/{image_folder}'
            pred_f = f'{real_images_prediction}/{image_folder}'
            pred_path = pred_f if prediction_model == 'llava-v1.5-13b' else llava_16_34b_json_v8_prediction
            samples = create_samples_with_augmented_policies(row, image_f, pred_path, template_version, augmentation,
                                                             counter=count_valid_explanations)
            if len(samples) == 0:
                continue
            real_paths.append(samples[0]['image'])
            real_ids.append(samples[0]['id'])
            category = get_mapping(template_version)[str(row['category'])]
            score = row['score']
            for s in samples:
                all_ratings.append(get_safety_rating(s))
                all_ids.append(s['id'])

            all_scores.extend([score] * len(samples))
            data = pd.concat([data, pd.DataFrame([[samples, category, score]], columns=['data', 'category', 'score'])])

    # copy images to /common-repos/LlavaGuard/data/urls and
    os.makedirs(f'{ds_out}/urls', exist_ok=True)
    os.chmod(f'{ds_out}/urls', 0o777)
    # split in 4 chunks and copy images
    chunk_size = len(real_paths) // 4

    for i in range(4):
        chunk = real_paths[i * chunk_size: (i + 1) * chunk_size] if i < 3 else real_paths[i * chunk_size:]
        chunk_ids = real_ids[i * chunk_size: (i + 1) * chunk_size] if i < 3 else real_ids[i * chunk_size:]
        os.makedirs(f'/common-repos/LlavaGuard/data/urls/{i}', exist_ok=True)
        for im, id in zip(chunk, chunk_ids):
            shutil.copy(im, f'/common-repos/LlavaGuard/data/urls/{i}/{id}.jpg')
        #  create csv file with ids and empty urls
        with open(f'/common-repos/LlavaGuard/data/urls/urls_{i}.csv', 'w') as f:
            f.write('id,url\n')
            for id in chunk_ids:
                f.write(f'{id},\n')






    train_split, eval_split = [], []
    categories, scores = data['category'].unique(), data['score'].unique()
    for category, score in product(categories, scores):
        subset = data[(data['category'] == category) & (data['score'] == score)]['data'].values
        test_samples = 20
        if 'Acceptable' in score:
            test_samples = 10
        if 'None applying' in category:
            test_samples = 100
        # at least 4 unique images in each category/score pair, at least 20 test samples for each pair, 100 for None applying
        im_count = 4
        while True:
            if im_count >= len(subset):
                eval_split.extend(subset)
                print(f'Insufficient samples for Category: {category}, Score: {score}, Count: {len(subset)}')
                break
            train, test = train_test_split(subset, test_size=im_count, random_state=42)
            # print(f'Category: {category}, Score: {score}, Train: {len(train)}, Test: {len(test)}')
            len_test = len([item for sublist in test for item in sublist])
            if len_test >= test_samples:
                train_split.extend(train)
                eval_split.extend(test)
                break
            im_count += 1


    # print numbers of samples in each category

    rating_vals = list(set(all_ratings))
    # print shape of train and test data
    # eval_split = [item for sublist in test_set for item in sublist]
    # train_split = [item for sublist in train_set for item in sublist]
    # print_txt += f'All data: ({len(data)}), Train data: ({len(train_set)}), Eval data: ({len(test_set)})')
    with (open(f'{ds_out}/all_data.json', 'w') as all_json, open(f'{ds_out}/{train_data_name}.json', 'w') as train_json,
          open(f'{ds_out}/{eval_data_name}.json', 'w') as eval_json, open(f'{ds_out}/{test_data_name}.json',
                                                                          'w') as test_json,
          open(f'{ds_out}/{train_data_name}_oversampled.json', 'w') as train_oversampled_json):
        flattened_train_split = [item for sublist in train_split for item in sublist]
        flattened_train_split_oversampled = oversample_minority_class(flattened_train_split)
        flattened_eval_split = [item for sublist in eval_split for item in sublist]
        flattened_test_split = [item for sublist in test_data for item in sublist]
        all_data = flattened_train_split + flattened_eval_split + flattened_test_split
        json.dump(all_data, all_json, indent=4)
        json.dump(flattened_train_split, train_json, indent=4)
        json.dump(flattened_train_split_oversampled, train_oversampled_json, indent=4)
        json.dump(flattened_test_split, test_json, indent=4)
        # flattened_eval_split.extend(flattened_test_split)
        json.dump(flattened_eval_split, eval_json, indent=4)
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
Unique images: {len(data)} ({len(data[data["score"] == "Generally Acceptable"])} Generally Acceptable, {len(data[data["score"] == "Barely Acceptable"])} Barely Acceptable, {len(data[data["score"] == "Moderately Inappropriate"])} Moderately Inappropriate, {len(data[data["score"] == "Highly Inappropriate"])} Highly Inappropriate)
Number of images in each category: {data.groupby(["category"]).size()}
#################################################################################################
All data: {len(all_data)} ({rating_vals[0]} {all_ratings.count(rating_vals[0])}, {rating_vals[1]} {all_ratings.count(rating_vals[1])})
Default policy {safe+unsafe} ({safe} safe, {unsafe} unsafe), Augmented Policy v1 ({v1}), v2 ({v2} unsafe -> safe), v3 ({v3_safe} safe, {v3_unsafe} unsafe), v4 ({v4} safe -> unsafe), v5 ({v5_safe} safe, {v5_unsafe} unsafe)
General Acceptable: {all_scores.count("Generally Acceptable")}, Barely Acceptable: {all_scores.count("Barely Acceptable")}, Moderately Inappropriate: {all_scores.count("Moderately Inappropriate")}, Highly Inappropriate: {all_scores.count("Highly Inappropriate")}
Train data: (unique {len(flattened_train_split)} -> oversampled {len(flattened_train_split_oversampled)}),
Eval data: ({len(flattened_eval_split)}), Test data: ({len(flattened_test_split)})
Valid explanations: ({count_valid_explanations[0]}/{count_valid_explanations[1] + count_valid_explanations[0]})
Train Data saved at: {ds_out}/{train_data_name}.json
Eval Data saved at: {ds_out}/{eval_data_name}.json
Test Data saved at: {ds_out}/{test_data_name}.json
#################################################################################################
'''
    print(print_txt)
    # save ds info
    with open(f'{ds_out}/ds_info.txt', 'w') as f:
        f.write(print_txt)
    plot_ds_heatmap(f'{ds_out}/all_data.json')
    plot_ds_heatmap(f'{ds_out}/{train_data_name}.json')
    plot_ds_heatmap(f'{ds_out}/{eval_data_name}.json')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for LlavaGuard instructive tuning with policy')
    parser.add_argument('--template_version', default='json-v6', help='either nl, json or json-v2')
    parser.add_argument('--augmentation', default=True,
                        help='If augmentation, we employ policy augmentation. We apply two augmentation techniques'
                             'to the unsafe examples: 1. We drop a random number of categories from the taxonomy that'
                             'are not violated in the given example. 2. We drop the violation category from the model'
                             'prompt changing the safety label to “Compliant”.')
    parser.add_argument('--explanations', default='v2',
                        help='Version of the model predictions to use. Options: v1, v2')
    args = parser.parse_args()
    template_version = args.template_version
    augmentation = args.augmentation if isinstance(args.augmentation, bool) else args.augmentation == 'True'
    prepare_instruct_tuning_with_policy_augmentation(template_version, augmentation=augmentation,
                                                     explanations=args.explanations)
