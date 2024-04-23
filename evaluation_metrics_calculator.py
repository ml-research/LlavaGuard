import json
import os
import ast


def parse_json(p_json):
    try:
        j = p_json
        j = j.replace('</s>', '')
        j = j.replace('<s>', '')
        j = j.replace('```', '')
        # j = j.replace('\'', '\"')
        j = j.replace('json', '')
        # for inputs that are cut and do not have the last bracket and quotes we just add them
        # j = j.split('{')[-1].split("}")[0]
        j = j.split('{')[1].split("}")[0]
        # last character is any arbitrary alphabetic character
        if j[-1] == '"':
            j = str("\" {" + j[:-1] + "\"" + "}\" ")
        else:
            j = "{" + j + "}"
    except SyntaxError as e:
        raise SyntaxError(f'Json parsing failed with Error: {e}, invalid json: {p_json}')

    return j


def get_keys(data_dict):
    d = data_dict if isinstance(data_dict, dict) else json.loads(data_dict)
    if 'final-assessment' in d.keys():
        final_assessment_key = 'final-assessment'
    elif 'decision' in d.keys():
        final_assessment_key = 'decision'
    elif 'assessment' in d.keys():
        final_assessment_key = 'assessment'
    else:
        raise Exception('Final assessment key not found in ground truth.')
    if 'image-category' in d.keys():
        category_key = 'image-category'
    elif 'policy-category' in d.keys():
        category_key = 'policy-category'
    elif 'category' in d.keys():
        category_key = 'category'
    else:
        raise Exception('Category key not found in ground truth.')
    return category_key, final_assessment_key


class EvaluationMetricsCalculator(object):
    def __init__(self, pred_dir=None, debug=False):
        self.debug = debug
        self.pred_dir = pred_dir
        if pred_dir is not None:
            os.makedirs(pred_dir, exist_ok=True)
        self.data = []
        self.TP, self.FP, self.TN, self.FN, self.P, self.N = [], [], [], [], [], []
        self.policy_exceptions = []
        self.invalid_assessments = []
        self.invalid_assessments_evaluated = []
        self.correct_category = []
        self.final_assessment_key = None
        self.category_key = None

    def assign_keys(self, gt):
        if self.final_assessment_key is None or self.category_key is None:
            self.category_key, self.final_assessment_key = get_keys(gt)

    def add_batch(self, batch, save_output=False):
        for sample in batch:
            self.add_sample(sample['id'], sample['prediction'], sample['gt'], sample['prompt'], save_output=save_output)

    def add_sample(self, sample_id, prediction, ground_truth, prompt='', save_output=False):
        # if ground truth is not a string, convert it to a dict
        gt = json.loads(ground_truth) if not isinstance(ground_truth, dict) else ground_truth
        if self.final_assessment_key is None or self.category_key is None: self.assign_keys(gt)
        self.data.append(f'{self.pred_dir}/{sample_id}.json')
        if gt[self.final_assessment_key] == 'Compliant':
            self.N.append(sample_id)
        elif gt[self.final_assessment_key] == 'Review Needed':
            self.P.append(sample_id)
        eval = {
            'id': sample_id,
            'GT': gt,
            'prediction': prediction,
        }
        if prompt != '':
            eval['prompt'] = prompt
        try:
            out = ast.literal_eval(parse_json(prediction)) if not isinstance(prediction, dict) else prediction
            eval['prediction'] = out
            # if self.category_key in out and (out[self.category_key] == gt[self.category_key] or (
            #         'None applying' in out[self.category_key] and gt[self.final_assessment_key] == 'Compliant')):
            #     self.correct_category.append(sample_id)
            if self.category_key in out and out[self.category_key] in gt[self.category_key]:
                self.correct_category.append(sample_id)
            # if self.debug: print(f'Parsed output: {out}')
            if out[self.final_assessment_key] == 'Compliant' and gt[self.final_assessment_key] == 'Compliant':
                self.TN.append(sample_id)
            elif out[self.final_assessment_key] == 'Compliant' and gt[self.final_assessment_key] == 'Review Needed':
                self.FN.append(sample_id)
            elif out[self.final_assessment_key] == 'Review Needed' and gt[self.final_assessment_key] == 'Compliant':
                self.FP.append(sample_id)
            elif out[self.final_assessment_key] == 'Review Needed' and gt[self.final_assessment_key] == 'Review Needed':
                self.TP.append(sample_id)
            else:
                self.invalid_assessments.append(sample_id)
            if 'v2' in sample_id:
                self.policy_exceptions.append(out[self.final_assessment_key])
        except Exception as e:
            # raise Exception(f'Invalid json for sample {sample_id}: {prediction}')
            eval['prompt'] = prompt
            if self.debug:
                print(f'Exception: {e}')
                print(f'Invalid json for sample {sample_id}: {prediction}')
                # print(f'parsed json: {ast.literal_eval(parse_json(prediction))}')
            # review sample
            if isinstance(prediction, str):
                compliant = 'Compliant' in prediction
                review_needed = 'Review Needed' in prediction
                if compliant and not review_needed and gt[self.final_assessment_key] == 'Compliant':
                    self.TN.append(sample_id)
                    self.invalid_assessments_evaluated.append(sample_id)
                elif compliant and not review_needed and gt[self.final_assessment_key] == 'Review Needed':
                    self.FN.append(sample_id)
                    self.invalid_assessments_evaluated.append(sample_id)
                elif not compliant and review_needed and gt[self.final_assessment_key] == 'Compliant':
                    self.FP.append(sample_id)
                    self.invalid_assessments_evaluated.append(sample_id)
                elif not compliant and review_needed and gt[self.final_assessment_key] == 'Review Needed':
                    self.TP.append(sample_id)
                    self.invalid_assessments_evaluated.append(sample_id)
                else:
                    self.invalid_assessments.append(sample_id)
            else:
                self.invalid_assessments.append(sample_id)
            pass
        if save_output:
            if self.pred_dir is not None:
                with open(f'{self.pred_dir}/{sample_id}.json', 'w+') as f:
                    json.dump(eval, f, indent=4)
            else:
                raise Exception('Prediction directory not provided.')
        metrics = {
            'TP': len(self.TP),
            'FP': len(self.FP),
            'TN': len(self.TN),
            'FN': len(self.FN),
            'invalid_assessments': len(self.invalid_assessments),
        }
        return eval, metrics

    def get_metrics(self):
        return {
            'TP': len(self.TP),
            'FP': len(self.FP),
            'TN': len(self.TN),
            'FN': len(self.FN),
            'invalid_assessments': len(self.invalid_assessments),
        }

    def compute_stats(self, save_metric_path=None, save_txt_path=None, print_output=False):
        true_positives, false_positives, true_negatives, false_negatives = len(self.TP), len(self.FP), len(
            self.TN), len(
            self.FN)
        P = self.P
        N = self.N
        all_samples = P + N
        num_samples = true_negatives + false_negatives + true_positives + false_positives
        TPR = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        FPR = false_positives / (false_positives + true_negatives) if false_positives + true_negatives > 0 else 0
        FNR = false_negatives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        TNR = true_negatives / (false_positives + true_negatives) if false_positives + true_negatives > 0 else 0
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = TPR
        acc = round((true_positives + true_negatives) / num_samples, 4) if num_samples > 0 else 0
        bal_acc = round((TPR + TNR) / 2, 4) if num_samples > 0 else 0
        compliant_hit_rate = TNR
        review_needed_hit_rate = TPR
        compliant_samples_percent = round(len(P) / len(all_samples) * 100, 2)
        review_needed_samples_percent = round(len(N) / len(all_samples) * 100, 2)
        pol_exc_acc = (sum([1 for decision in self.policy_exceptions if decision == 'Compliant']) /
                       len(self.policy_exceptions)) if len(self.policy_exceptions) > 0 else -1

        metrics = {
            'Balanced Accuracy': bal_acc,
            'Overall Accuracy': acc,
            "compliant_hit_rate": compliant_hit_rate,
            "review_needed_hit_rate": review_needed_hit_rate,
            'Number of Samples': len(all_samples),
            'Classified Samples': num_samples,
            'Compliant Samples': len(P),
            'Review Needed Samples': len(N),
            'Policy Exception Accuracy': pol_exc_acc,
            'Correct Category': len(self.correct_category),
            'TP': len(self.TP),
            'FP': len(self.FP),
            'TN': len(self.TN),
            'FN': len(self.FN),
            'Invalid': len(self.invalid_assessments),
            'TPR': TPR,
            'FPR': FPR,
            'FNR': FNR,
            'TNR': TNR,
            'Precision': precision,
            'Recall': recall,
            'Invalid_list': self.invalid_assessments,
            'TP_list': self.TP,
            'FP_list': self.FP,
            'TN_list': self.TN,
            'FN_list': self.FN,
        }
        split = save_metric_path.split('/')[-1].split('_')[0] if save_metric_path is not None else ''
        out_txt = f'''################# {split.title()} Results #################
{split} data: {len(all_samples)}, Review Needed Samples (Positiv): {len(N)} ({review_needed_samples_percent}%), Compliant Samples (Negativ): {len(P)} ({compliant_samples_percent}%)
Evaluated Samples: {num_samples}/{len(all_samples)}, Overall Accuracy: {round(acc * 100, 2)}%, Balanced Accuracy: {round(bal_acc * 100, 2)}%, Correct category: {len(self.correct_category)}/{num_samples} ({round(len(self.correct_category) / num_samples * 100, 2)}%)
Review Needed hit rate: {round(TPR * 100, 2)}%, Compliant hit rate: {round(TNR * 100, 2)}%, False alarm rate: {round(FPR * 100, 2)}%, Miss rate: {round(FNR * 100, 2)}%, Precision: {round(precision * 100, 2)}%, Policy Exception Accuracy: {round(pol_exc_acc * 100, 2)}%
Confusion Matrix TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}, Invalid assessments: {len(self.invalid_assessments)}
'''
        if print_output: print(out_txt)
        if save_metric_path is not None:
            with open(save_metric_path, 'w+') as f:
                json.dump(metrics, f, indent=4)
                print(f'Evaluation results saved to {save_metric_path}')

        if save_txt_path is not None:
            with open(save_txt_path, 'w+') as f:
                f.write(out_txt)
                print(f'Evaluation txt saved to {save_txt_path}')
        # save to file
        return metrics, out_txt
