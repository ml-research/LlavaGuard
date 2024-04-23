import numpy as np
import pandas as pd

appropriate_categories = ['Polite Communication and Gestures', 'Nature', 'Respectful or Appropriate Content',
                          'Peace or Care', 'NaN']

inappropriate_categories = ['Humiliation, Harassment or Hate',
                            'Explicit Sexual Content',
                            'Nudity',
                            'Illegal Activities',
                            'Animal Cruelty',
                            'Substance Abuse or Weapons',
                            'Violence, Harm or Cruelty',
                            'Disasters or Emergencies',
                            'Suicide or Self Harm',
                            'Disasters or Emergencies'
                            ]

categories_v6 = [
    'O1: Hate, Humiliation, Harassment',
    '02: Violence, Harm or Cruelty',
    'O3: Sexual Content',
    'O4: Nudity Content',
    'O5: Criminal Planning',
    'O6: Illegal Weapons or Substance Abuse',
    'O7: Self-Harm',
    'O8: Animal Cruelty',
    '09: Disasters or Emergencies',
    '10: None applying'
]
inappropriate_categories_v6 = [c[4:] for c in categories_v6[:-1]]


def calc_metrics(data):
    # Assuming 'score' is the ground truth and 'pred_score' is the prediction
    true_positives = ((data['decision'] == 1) & (data['pred_decision'] == 1)).sum()
    false_positives = ((data['decision'] == 0) & (data['pred_decision'] == 1)).sum()
    true_negatives = ((data['decision'] == 0) & (data['pred_decision'] == 0)).sum()
    false_negatives = ((data['decision'] == 1) & (data['pred_decision'] == 0)).sum()
    TPR = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    FPR = false_positives / (false_positives + true_negatives) if false_positives + true_negatives > 0 else 0
    FNR = false_negatives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    TNR = true_negatives / (false_positives + true_negatives) if false_positives + true_negatives > 0 else 0
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    bal_accuracy = (TPR + TNR) / 2
    precision, bal_accuracy = round(precision * 100, 2), round(bal_accuracy * 100, 2)
    TPR, FPR, FNR, TNR = round(TPR * 100, 2), round(FPR * 100, 2), round(FNR * 100, 2), round(TNR * 100, 2)
    detection_rate, false_alarm_rate = TPR, FPR
    return TPR, FPR, FNR, TNR, precision, bal_accuracy


def convert_to_dataset_ground_truth(data, metric='score'):
    '''
    This function will process the data and categorize it based on the category. Filters out the categories where
    the number of samples is less than 5
    :param metric: Either score_mean, score_median, score_max
    :return:
    '''
    categories = data['category'].unique()
    categorial_data = {'num_samples': [], 'score_mean': [], 'score_median': [], 'score_max': [], 'c_angle': [],
                       'category': []}
    c_id = 0
    for category in inappropriate_categories:
        c_data = data[data['category'] == category]
        num_samples = c_data.shape[0]
        # remove all the columns where samples are less than 5
        if num_samples < 5:
            continue
        score_mean = c_data[metric].mean()
        score_median = c_data[metric].median()
        score_max = c_data[metric].max()
        categorial_data['score_mean'].append(score_mean)
        categorial_data['score_median'].append(score_median)
        categorial_data['score_max'].append(score_max)
        categorial_data['num_samples'].append(num_samples)
        categorial_data['c_angle'].append(c_id)
        categorial_data['category'].append(category)
        c_id += 1
    # scale the angles between 0 and 2pi
    categorial_data['c_angle'] = np.array(categorial_data['c_angle']) * 2 * np.pi / c_id

    return pd.DataFrame(categorial_data)


def convert_to_performance_compass(data_dict):
    def data_preparation_for_llavaguard_compass(data):
        '''
        This function will process the data and categorize it based on the category. Filters out the categories where
        the number of samples is less than 5
        :param metric: Either score_mean, score_median, score_max
        :return:
        '''
        categories = data['category'].unique()

        num_samples_list = []
        detection_rate_list = []
        detection_rate_hi_list = []
        c_angle_list = []
        category_list = []
        bal_acc_list = []
        precision_list = []
        false_alert_list = []

        # cats = inappropriate_categories
        c_id = 0
        for category in inappropriate_categories_v6:
            c_data = data[data['category'] == category]
            inapropiate_samples = c_data[c_data['decision'] == 1]
            num_samples = c_data.shape[0]
            num_samples = inapropiate_samples.shape[0]
            # remove all the columns where samples are less than 5
            # if num_samples < 5:
            #     continue
            TPR, FPR, FNR, TNR, precision, bal_accuracy = calc_metrics(c_data)
            detection_rate_list.append(TPR)
            bal_acc_list.append(bal_accuracy)
            precision_list.append(precision)
            false_alert_list.append(FPR)
            # categorial_data['False Alarm'].append(false_alarm)
            HI_data = c_data[c_data['score'] == 3]
            TPR, FPR, FNR, TNR, precision, bal_accuracy = calc_metrics(HI_data)
            detection_rate_hi_list.append(TPR)
            num_samples_list.append(num_samples)
            c_angle_list.append(c_id)
            category_list.append(category)
            c_id += 1
            # scale the angles between 0 and 2pi
        c_angle_list = np.array(c_angle_list) * 2 * np.pi / c_id

        categorial_data = {
            # 'num_samples': num_samples_list,
            # 'Balanced Accuracy': bal_acc_list,
            # 'Precision': precision_list,
            # 'False Alert Rate': false_alert_list,
            'Detection Rate for \n Unsafe Data': detection_rate_list,
            'Detection Rate for \n Highly Unsafe Data': detection_rate_hi_list,
            'c_angle': c_angle_list,
            'category': category_list
        }
        return pd.DataFrame(categorial_data)

    compass_data = {}
    for model, data in data_dict.items():
        compass_data[model] = data_preparation_for_llavaguard_compass(data)
    return compass_data


def convert_to_dataset_compass(data_dict):
    data = data_dict['LlavaGuard']
    llavaguard_mean_list = []
    hf_mean_list = []
    c_angle_list = []
    category_list = []
    num_samples_list = []
    num_unsafe_samples = []
    llava_guard_num_samples = []
    llava_guard_unsafe_num_samples = []
    categories = data['category'].unique()
    pred_categories = data['pred_category'].unique()
    # normalize data score
    data['score'] = data['score'].apply(lambda x: x / 3)
    c_id = 0
    for category in inappropriate_categories_v6:
        c_data = data[data['category'] == category]
        llava_guard_data = data[data['pred_category'] == category]

        num_samples = c_data.shape[0]
        unsafe_samples = c_data[c_data['decision'] == 1]
        llava_guard_unsafe_samples = llava_guard_data[llava_guard_data['pred_decision'] == 1]

        score_mean = c_data['decision'].mean() * 100
        llavaguard_mean = c_data['pred_decision'].mean() * 100

        num_samples_list.append(num_samples)
        num_unsafe_samples.append(unsafe_samples.shape[0])
        # remove all the columns where samples are less than 5
        # if num_samples < 5:
        #     continue
        hf_mean_list.append(score_mean)
        llavaguard_mean_list.append(llavaguard_mean)
        llava_guard_num_samples.append(llava_guard_data.shape[0])
        llava_guard_unsafe_num_samples.append(llava_guard_unsafe_samples.shape[0])
        c_angle_list.append(c_id)
        category_list.append(category)
        c_id += 1
        print(c_data['pred_correct'].mean() * 100)
        # scale the angles between 0 and 2pi
    c_angle_list = np.array(c_angle_list) * 2 * np.pi / c_id

    # data = data_dict['LLaVA']

    categorial_data = {
        'LlavaGuard': pd.DataFrame({
            'Category Detections': llava_guard_num_samples,
            '# Unsafe Samples \n by Category': llava_guard_unsafe_num_samples,
            'Ø Safety Score \n by Category': llavaguard_mean_list,
            'c_angle': c_angle_list,
            'category': category_list
        }),
        'HumanFeedback': pd.DataFrame({
            'Category Detections': num_samples_list,
            '# Unsafe Samples \n by Category': num_unsafe_samples,
            'Ø Safety Score \n by Category': hf_mean_list,
            'c_angle': c_angle_list,
            'category': category_list
        })
    }
    return categorial_data
