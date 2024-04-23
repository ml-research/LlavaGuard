import glob
import os.path
from itertools import product
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def category_distribution_heat(csv_files, name='SMID'):
    data = pd.concat([pd.read_csv(path) for path in csv_files])
    print(f'Number of samples: {data.shape[0]}')

    categories = ['Humiliation, Harassment or Hate', 'Nudity', 'Animal Cruelty',
                  'Substance Abuse or Weapons', 'Violence, Harm or Cruelty', 'Illegal Activities',
                  'Explicit Sexual Content', 'Disasters or Emergencies', 'Suicide or Self Harm']
    scores = ['Generally Acceptable', 'Barely Acceptable', 'Moderately Inappropriate', 'Highly Inappropriate']
    distributional_data = {'category': [], 'score': [], 'num_samples': []}
    for s, c in product(scores, categories):
        if s == 'Discard Sample':
            continue
        num_samples = data[(data['category'] == c) & (data['score'] == s)].shape[0]
        distributional_data['category'].append(c)
        distributional_data['score'].append(s)
        distributional_data['num_samples'].append(num_samples)
    # create fig
    fig, axes = plt.subplots(1, 1, sharey=True, figsize=(10, 8))
    data = pd.DataFrame(distributional_data)
    # pivot the data
    data = data.pivot(index='category', columns='score', values='num_samples')
    # reorder columns
    data = data.reindex(columns=scores)
    # make title
    plt.title(name + f' ({data.sum().sum()} images)')
    sns.heatmap(data, annot=True, annot_kws={"size": 20}, linewidths=1, cmap="Blues", vmin=0, vmax=50,
                cbar_kws={'format': '%%.f%%'}, fmt='d', cbar=False)
    out_path = f'output/plots/category_distribution/{name}.png'
    os.makedirs(os.path.dirname(out_path.replace(' ', '_')), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('png', 'pdf'), dpi=300, bbox_inches='tight')
