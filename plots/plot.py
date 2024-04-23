import glob

from plots.category_distribution_heatmap import category_distribution_heat
from plots.compassess.dataset_compass import llavaguard_compass, score_compass
from plots.load_data import load_score_data, load_prediction_data
from plots.compassess.mm_compass import moral_mean_compass
from plots.prepare_data import convert_to_performance_compass, convert_to_dataset_compass

data_dir = 'data/smid_llava_guard_samplingv1_v1.5-13b_constrained_humanfeedback'
SMID_files = glob.glob(f'{data_dir}/*.csv')
data_dir = 'data/smid_llava_guard_samplingv1_v1.5-13b_constrained_real_images_v2_humanfeedback'
real_im_files = glob.glob(f'{data_dir}/*/*.csv')

# heatmaps for the datasets
category_distribution_heat(SMID_files, name='SMID')
category_distribution_heat(real_im_files, name='Webcrawler Images')
category_distribution_heat(SMID_files + real_im_files, name='SMID and Webcrawler Images')

# # get all the csv files in the directory
out_path = 'output/plots/compass/SMID.png'
data = load_score_data(SMID_files)
score_compass(data, out_path=out_path, title='SMID Inappropriateness Compass')

# # get the data
out_path = 'output/plots/compass/RealImages.png'
data = load_score_data(real_im_files)
score_compass(data, out_path=out_path, title='Real Images Inappropriateness Compass')

out_path = 'output/plots/compass/SMID_and_RealImages.png'
data = load_score_data(SMID_files + real_im_files)
score_compass(data, out_path=out_path, title='SMID and Real Images Inappropriateness Compass')

template_version = 'json-v6'
# # get all the csv files in the directory
if template_version == 'json-v4':
    pred_path = f'output/eval/llava-v1.5-13b/lora/{template_version}_oversampled-final/{template_version}/model_output'
else:
    pred_path = f'output/eval/llava-v1.5-13b/lora/{template_version}/{template_version}/model_output'

pred_path_llava = f'output/eval/llava-v1.5-13b/foundation_model/{template_version}/model_output'
# pred_path_llava = f'output/eval/llava-v1.6-34b/foundation_model/{template_version}/model_output'
ds_path = f'output/data/smid_and_crawled_policy/{template_version}/eval.json'
model_dict = {'LlavaGuard': pred_path, 'LLaVA': pred_path_llava}
out_path = 'output/plots/compass/LavaGuard_stats.pdf'
# data_dict = load_prediction_data(ds_path, model_dict)
# compass_data = convert_to_performance_compass(data_dict)
# llavaguard_compass(compass_data, out_path, 'LavaGuard on SMID+ dataset')


# template_version = 'json-v4'
# pred_path = f'output/eval/llava-v1.5-13b/lora/{template_version}_oversampled-final/{template_version}/model_output'
# pred_path = f'output/eval/llava-v1.5-13b/lora/{template_version}/{template_version}/model_output'
ds_path = f'output/data/smid_and_crawled_policy/{template_version}/all_data.json'
eval_ds_path = f'output/data/smid_and_crawled_policy/{template_version}/eval.json'
pred_path = f'output/eval/llava-v1.5-13b/lora/{template_version}/{template_version}/model_output'

out_path = 'output/plots/compass/dataset.pdf'
model_dict = {'LlavaGuard': pred_path}
# data_dict = load_prediction_data(eval_ds_path, model_dict)
# compass_data = convert_to_dataset_compass(data_dict)
# moral_mean_compass(compass_data, out_path)


