import json
import os.path
import warnings

import pandas as pd

from llavaguard.taxonomy.policies import convert_category
from llavaguard.evaluation_metrics_calculator import get_keys
from plots.prepare_data import convert_to_dataset_ground_truth


def load_score_data(csv_path):
    if isinstance(csv_path, str):
        data = pd.read_csv(csv_path)
    elif isinstance(csv_path, list):
        data = pd.concat([pd.read_csv(path) for path in csv_path])
    else:
        raise ValueError('csv_path should be either a string or a list of strings')
    scores = ['Generally Acceptable', 'Barely Acceptable', 'Moderately Inappropriate', 'Highly Inappropriate']
    # remove all discarded samples
    data = data[data['score'].isin(scores)]
    data['score'] = data['score'].apply(lambda x: scores.index(x))
    data['score'] = data['score'].astype(int)
    return convert_to_dataset_ground_truth(data)


def read_csv_and_update_id(csv_paths):
    csv_paths = [csv_paths] if isinstance(csv_paths, str) else csv_paths
    csvs = []
    for csv_path in csv_paths:
        data = pd.read_csv(csv_path)
        if 'real_images' in csv_path:
            cat = csv_path.split('/')[-2]
            cat = cat.replace(' ', '_')
            data['json'] = data['json'].apply(lambda x: f'{cat}_{x}')
        csvs.append(data)
    return pd.concat(csvs)


def load_prediction_data(ds_path, model_dict):
    def add_predictions(data, pred_path):
        decisions_int = {'Review Needed': 1, 'Compliant': 0}
        # load the predictions
        data = data.copy()
        missing = []
        entries_added = []
        for entry in data['id']:
            # remove everything after the last underscore
            pred_file = os.path.join(pred_path, entry + '.json')
            pred_file = pred_file if os.path.isfile(pred_file) else os.path.join(pred_path, entry + '.txt')
            try:
                with open(pred_file, 'r') as f:
                    out = json.loads(f.read())
                pred = out['LlavaGuard'] if 'LlavaGuard' in out else out['prediction']
                # pred = pred_data['LlavaGuard'] if 'LlavaGuard' in pred_data else pred_data['prediction']
                # try to get the prediction from the LlavaGuard model

                # pred_category = pred['image-category']
                data.loc[data['id'] == entry, 'pred_decision'] = decisions_int[pred[final_assessment_key]]
                data.loc[data['id'] == entry, 'pred_category'] = convert_category(pred[category_key])
                data.loc[data['id'] == entry, 'pred_correct'] = (data.loc[data['id'] == entry, 'decision'] ==
                                                                 data.loc[data['id'] == entry, 'pred_decision']) * 100
                entries_added.append(entry)
            except:
                warnings.warn(f'No prediction found for {entry} at {pred_file}.')

                missing.append(entry)
        print(f'Predictions added for {len(entries_added)} entries and {len(missing)} entries were missing'
              f'for model {model}. Dropping the missing entries')
        # remove all the entries where the prediction is None
        data = data.dropna(subset=['pred_decision'])
        # print accuracy
        acc = data['pred_correct'].sum() / data.shape[0]
        print(f'Accuracy: {acc}')
        return data

    # data = read_csv_and_update_id(csv_path)
    eval_ds = pd.read_json(ds_path)
    scores = ['Generally Acceptable', 'Barely Acceptable', 'Moderately Inappropriate', 'Highly Inappropriate']
    score_to_decision = {'Generally Acceptable': 0, 'Barely Acceptable': 0, 'Moderately Inappropriate': 1,
                         'Highly Inappropriate': 1}
    scores_int = {'Generally Acceptable': 0, 'Barely Acceptable': 1, 'Moderately Inappropriate': 2,
                  'Highly Inappropriate': 3}
    # remove all discarded samples
    category_key, final_assessment_key = get_keys(eval_ds['conversations'][0][1]['value'])

    eval_ds = eval_ds[eval_ds['score'].isin(scores)]
    eval_ds['decision'] = eval_ds['score']
    eval_ds['decision'] = eval_ds['decision'].apply(lambda x: score_to_decision[x])
    eval_ds['decision'] = eval_ds['decision'].astype(int)
    eval_ds['score'] = eval_ds['score'].apply(lambda x: scores_int[x])
    eval_ds['score'] = eval_ds['score'].astype(int)
    eval_ds['category'] = eval_ds['conversations'].apply(lambda x: json.loads(x[1]['value'])[category_key])
    eval_ds['category'] = eval_ds['category'].apply(convert_category)

    # json.loads(eval_ds['conversations'][0][1]['value'])['image-category']
    eval_ds['pred_decision'] = [None] * eval_ds.shape[0]
    eval_ds['pred_category'] = [None] * eval_ds.shape[0]
    eval_ds['pred_correct'] = [None] * eval_ds.shape[0]
    eval_ds = eval_ds.drop(columns=['image', 'conversations', 'final-assessment'])

    # data = data[data['score'].isin(scores)]
    # # convert the scores to integers, 0 for Acceptable, 1 for inappropriate
    # data['decision'] = data['score']
    # data['decision'] = data['decision'].apply(lambda x: score_to_decision[x])
    # data['decision'] = data['decision'].astype(int)
    # data['score'] = data['score'].apply(lambda x: scores_int[x])
    # data['score'] = data['score'].astype(int)
    # # add the predictions to the data
    # data['pred_decision'] = [None] * data.shape[0]
    # data['pred_category'] = [None] * data.shape[0]
    # data['pred_correct'] = [None] * data.shape[0]
    eval_ids = eval_ds['id'].values
    data_dict = {}

    for model, pred_path in model_dict.items():
        print(f'Adding predictions for {model}')
        d = add_predictions(eval_ds, pred_path)
        data_dict[model] = d
    return data_dict
