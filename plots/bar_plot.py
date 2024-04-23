import json

import pandas as pd

llava_16_34b_path = '/common-repos/LlavaGuard/eval/llava-v1.6-34b/foundation_model/smid_and_crawled_with_augmented_policies-json-v6/eval_metrics.json'
llava_15_13b_path = '/common-repos/LlavaGuard/eval/llava-v1.5-13b/foundation_model/smid_and_crawled_with_augmented_policies-json-v6/eval_metrics.json'
llavaguard_v1_13b_path = '/common-repos/LlavaGuard/eval/llava-v1.5-13b/LORA/smid_and_crawled_with_augmented_policies/json-v6/smid_and_crawled_with_augmented_policies-json-v6/eval_metrics.json'
llavaguard_v1_1_13b_path = '/common-repos/LlavaGuard/eval/LlavaGuard-v1.1-13b/smid_and_crawled_with_augmented_policies/json-v6/smid_and_crawled_with_augmented_policies-json-v6/eval_metrics.json'

llava_16_34b = json.load(open(llava_16_34b_path))
llava_15_13b = json.load(open(llava_15_13b_path))
llavaguard_v1_13b = json.load(open(llavaguard_v1_13b_path))
llavaguard_v1_1_13b = json.load(open(llavaguard_v1_1_13b_path))

# create a dataframe with the data
data = {
    'Model': ['Llava-v1.6-34b', 'Llava-v1.5-13b', 'LlavaGuard-v1.5-13b', 'LlavaGuard-v1.1-13b'],
    'Accuracy': [llava_16_34b['Balanced Accuracy'], llava_15_13b['Balanced Accuracy'], llavaguard_v1_13b['Balanced Accuracy'], llavaguard_v1_1_13b['Balanced Accuracy']],
    'Precision': [llava_16_34b['Precision'], llava_15_13b['Precision'], llavaguard_v1_13b['Precision'], llavaguard_v1_1_13b['Precision']],
    'Recall': [llava_16_34b['Recall'], llava_15_13b['Recall'], llavaguard_v1_13b['Recall'], llavaguard_v1_1_13b['Recall']],
}
# create a dataframe
df = pd.DataFrame(data)
# create a bar plot using seaborn
import seaborn as sns
import matplotlib.pyplot as plt
