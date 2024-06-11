import json
import os
import ast

from llavaguard.taxonomy.policies import safety_assessment_functions, safety_assessment_json_keys


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
        if "{" in j:
            j = j.split('{')[1]
        if "}" in j:
            j = j.split("}")[0]
        # last character is any arbitrary alphabetic character
        if j[-1] == '"':
            j = str("\" {" + j[:-1] + "\"" + "}\" ")
        else:
            j = "{" + j + "}"
    except SyntaxError as e:
        raise SyntaxError(f'Json parsing failed with Error: {e}, invalid json: {p_json}')

    return j


def get_rating_values(rating_key):
    if rating_key == 'final-assessment':
        return ['Compliant', 'Review Needed']
    elif rating_key == 'decision':
        return ['Compliant', 'Review Needed']
    elif rating_key == 'assessment':
        return ['Compliant', 'Review Needed']
    elif rating_key == 'rating':
        return ['Safe', 'Unsafe']
    else:
        raise Exception(f'Rating values not found for key: {rating_key}')


def get_keys(data_dict):
    data = data_dict if isinstance(data_dict, dict) else json.loads(data_dict)
    for templ, func in list(safety_assessment_functions.items())[1:]:
        out = json.loads(func('rating', 'NaN', 'rationale'))
        # check if chosen assessment matches the data keys
        if out.keys() == data.keys():
            key = safety_assessment_json_keys[templ]

            return key['category'], key['rationale'], key['rating']
    raise Exception(f'Keys not found in ground truth: {data}')

    # if 'final-assessment' in d.keys():
    #     rating_key = 'final-assessment'
    # elif 'decision' in d.keys():
    #     rating_key = 'decision'
    # elif 'assessment' in d.keys():
    #     rating_key = 'assessment'
    # elif 'rating' in d.keys():
    #     rating_key = 'rating'
    # else:
    #     raise Exception('Final assessment key not found in ground truth.')
    # if 'image-category' in d.keys():
    #     category_key = 'image-category'
    # elif 'policy-category' in d.keys():
    #     category_key = 'policy-category'
    # elif 'category' in d.keys():
    #     category_key = 'category'
    # else:
    #     raise Exception('Category key not found in ground truth.')
    # if 'rationale' in d.keys():
    #     rationale_key = 'rationale'
    # else:
    #     rationale_key = 'explanation'
    # return category_key, rationale_key, rating_key


class EvaluationMetricsCalculator(object):
    def __init__(self, pred_dir=None, debug=False):
        self.debug = debug
        self.pred_dir = pred_dir
        if pred_dir is not None:
            os.makedirs(pred_dir, exist_ok=True)
        self.data = []
        self.TP, self.FP, self.TN, self.FN, self.P, self.N = [], [], [], [], [], []
        self.invalid_assessments = []
        self.invalid_assessments_evaluated = []
        self.correct_category = []
        self.final_assessment_key, self.category_key, self.rationale_key = None, None, None
        self.safe, self.unsafe = None, None

    def assign_keys(self, gt):
        if self.final_assessment_key is None or self.category_key is None:
            self.category_key, self.rationale_key, self.final_assessment_key = get_keys(gt)
            self.safe, self.unsafe = get_rating_values(self.final_assessment_key)

    def add_batch(self, batch, save_output=False):
        for sample in batch:
            self.add_sample(sample['id'], sample['prediction'], sample['gt'], sample['prompt'], save_output=save_output)

    def add_sample(self, sample_id, prediction, ground_truth, prompt='', save_output=False):
        # if ground truth is not a string, convert it to a dict
        gt = json.loads(ground_truth) if not isinstance(ground_truth, dict) else ground_truth
        if self.final_assessment_key is None or self.category_key is None: self.assign_keys(gt)
        self.data.append(f'{self.pred_dir}/{sample_id}.json')
        if gt[self.final_assessment_key] == self.safe:
            self.N.append(sample_id)
        elif gt[self.final_assessment_key] == self.unsafe:
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
            #         'None applying' in out[self.category_key] and gt[self.final_assessment_key] == self.safe)):
            #     self.correct_category.append(sample_id)
            if (self.category_key in out and out[self.category_key].lower() in gt[self.category_key].lower()):
                self.correct_category.append(sample_id)
            # if self.debug: print(f'Parsed output: {out}')
            if out[self.final_assessment_key] == self.safe and gt[self.final_assessment_key] == self.safe:
                self.TN.append(sample_id)
            elif out[self.final_assessment_key] == self.safe and gt[self.final_assessment_key] == self.unsafe:
                self.FN.append(sample_id)
            elif out[self.final_assessment_key] == self.unsafe and gt[self.final_assessment_key] == self.safe:
                self.FP.append(sample_id)
            elif out[self.final_assessment_key] == self.unsafe and gt[self.final_assessment_key] == self.unsafe:
                self.TP.append(sample_id)
            else:
                self.invalid_assessments.append(sample_id)

        except Exception as e:
            eval['prompt'] = prompt
            if self.debug:
                print(f'Invalid json for sample {sample_id} with Exception: {e}')
                print(prediction)
            # review sample
            if isinstance(prediction, dict):
                prediction = json.dumps(prediction)
            if isinstance(prediction, str):
                compliant = self.safe in prediction
                review_needed = self.unsafe in prediction
                if compliant and not review_needed and gt[self.final_assessment_key] == self.safe:
                    self.TN.append(sample_id)
                    self.invalid_assessments_evaluated.append(sample_id)
                elif compliant and not review_needed and gt[self.final_assessment_key] == self.unsafe:
                    self.FN.append(sample_id)
                    self.invalid_assessments_evaluated.append(sample_id)
                elif not compliant and review_needed and gt[self.final_assessment_key] == self.safe:
                    self.FP.append(sample_id)
                    self.invalid_assessments_evaluated.append(sample_id)
                elif not compliant and review_needed and gt[self.final_assessment_key] == self.unsafe:
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
                    try:
                        json.dump(eval, f, indent=4)
                    except:
                        print(f'Failed to save output for {sample_id} with prediction: {prediction}')
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
        if len(all_samples) == 0:
            print('No samples to evaluate.')
            return None, None
        num_samples = true_negatives + false_negatives + true_positives + false_positives
        TPR, FPR, FNR, TNR, precision, acc, bal_acc, f1, f2 = get_metrics(true_positives, false_positives,
                                                                          true_negatives,
                                                                          false_negatives)
        true_positives_default_policy, false_positives_default_policy, true_negatives_default_policy, false_negatives_default_policy = (
            len([x for x in self.TP if '_v' not in x]),
            len([x for x in self.FP if '_v' not in x]),
            len([x for x in self.TN if '_v' not in x]),
            len([x for x in self.FN if '_v' not in x]))
        (TPR_default_policy, FPR_default_policy, FNR_default_policy, TNR_default_policy, precision_default_policy,
         acc_default_policy, bal_acc_default_policy, f1_default_policy, f2_default_policy) = get_metrics(
            true_positives_default_policy, false_positives_default_policy, true_negatives_default_policy,
            false_negatives_default_policy)
        all_samples_default_policy = [x for x in all_samples if '_v' not in x]
        P_default_policy = [x for x in P if '_v' not in x]
        N_default_policy = [x for x in N if '_v' not in x]
        num_samples_default_policy = (true_positives_default_policy + false_positives_default_policy +
                                      true_negatives_default_policy + false_negatives_default_policy)

        policy_exception_true = len([x for x in self.TN if '_v2' in x]) + len([x for x in self.TP if '_v4' in x])
        policy_exception_false = len([x for x in self.FP if '_v2' in x]) + len([x for x in self.FN if '_v4' in x])
        pol_exc_rec = policy_exception_true / (
                policy_exception_true + policy_exception_false) if policy_exception_true + policy_exception_false > 0 else 0
        metrics = {
            'Balanced Accuracy': bal_acc,
            'Overall Accuracy': acc,
            "compliant_hit_rate": TNR,
            "review_needed_hit_rate": TPR,
            'Number of Samples': len(all_samples),
            'Classified Samples': num_samples,
            'Compliant Samples': len(N),
            'Review Needed Samples': len(P),
            'Policy Exception Recall': pol_exc_rec,
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
            'Recall': TPR,
            'Invalid_list': self.invalid_assessments,
            'TP_list': self.TP,
            'FP_list': self.FP,
            'TN_list': self.TN,
            'FN_list': self.FN,
        }
        split = save_metric_path.split('/')[-1].split('_')[0] if save_metric_path is not None else ''
        out_txt = f'''################# {split.title()} Results #################
{split} data: {len(all_samples)}, Unsafe Samples (Positiv): {len(P)} ({round(len(P) / len(all_samples) * 100, 2)}%), Safe Samples (Negativ): {len(N)} ({round(len(N) / len(all_samples) * 100, 2)}%)
Evaluated Samples: {num_samples}/{len(all_samples)}, Overall Accuracy: {round(acc * 100, 2)}%, Balanced Accuracy: {round(bal_acc * 100, 2)}%, Correct category: {len(self.correct_category)}/{num_samples} ({round(len(self.correct_category) / num_samples * 100, 2)}%)
Recall: {round(TPR * 100, 2)}%, Specificity: {round(TNR * 100, 2)}%, False alarm rate: {round(FPR * 100, 2)}%, Miss rate: {round(FNR * 100, 2)}%, Precision: {round(precision * 100, 2)}%, Policy Exception Recall: {round(pol_exc_rec * 100, 2)}%, F1: {round(f1 * 100, 2)}%, F2: {round(f2 * 100, 2)}%
Confusion Matrix TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}, invalid: {len(self.invalid_assessments)}, Unparsable samples: {len(self.invalid_assessments_evaluated) + len(self.invalid_assessments)}
'''
        out_txt_default_policy = f'''################# Default Policy Results #################
{split} data: {len(all_samples_default_policy)}, Unsafe Samples (Positiv): {len(P_default_policy)}, Safe Samples (Negativ): {len(N_default_policy)}
Evaluated Samples: {num_samples_default_policy}/{len(all_samples_default_policy)}, Overall Accuracy: {round(acc_default_policy * 100, 2)}%, Balanced Accuracy: {round(bal_acc_default_policy * 100, 2)}%
Recall: {round(TPR_default_policy * 100, 2)}%, specificity: {round(TNR_default_policy * 100, 2)}%, False alarm rate: {round(FPR_default_policy * 100, 2)}%, Miss rate: {round(FNR_default_policy * 100, 2)}%, Precision: {round(precision_default_policy * 100, 2)}%, F1: {round(f1_default_policy * 100, 2)}%, F2: {round(f2_default_policy * 100, 2)}%
Confusion Matrix TP: {true_positives_default_policy}, FP: {false_positives_default_policy}, TN: {true_negatives_default_policy}, FN: {false_negatives_default_policy}, Invalid assessments {len(all_samples_default_policy) - num_samples_default_policy}
'''
        if all_samples_default_policy != all_samples:
            out_txt += out_txt_default_policy

        if print_output:
            print(out_txt)

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


def get_metrics(true_positives, false_positives, true_negatives, false_negatives):
    TPR = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    FPR = false_positives / (false_positives + true_negatives) if false_positives + true_negatives > 0 else 0
    FNR = false_negatives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    TNR = true_negatives / (false_positives + true_negatives) if false_positives + true_negatives > 0 else 0
    TPR, FPR, FNR, TNR = round(TPR, 4), round(FPR, 4), round(FNR, 4), round(TNR, 4)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    num_samples = true_positives + false_positives + true_negatives + false_negatives
    acc = round((true_positives + true_negatives) / num_samples, 4) if num_samples > 0 else 0
    bal_acc = round((TPR + TNR) / 2, 4) if num_samples > 0 else 0
    F1 = 2 * (precision * TPR) / (precision + TPR) if precision + TPR > 0 else 0
    F2 = 5 * (precision * TPR) / (4 * precision + TPR) if 4 * precision + TPR > 0 else 0
    return TPR, FPR, FNR, TNR, precision, acc, bal_acc, F1, F2
