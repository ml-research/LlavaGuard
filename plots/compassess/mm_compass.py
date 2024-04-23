import os.path
import os.path

import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def moral_mean_compass(data_dict, out_path):
    sns.set_theme()
    paired = sns.color_palette("Paired")
    muted = sns.color_palette("muted")
    model_colors = {
        'LlavaGuard': (paired[0], paired[1]),
        'LLaVA': (paired[2], paired[3]),
        'Data': (paired[4], paired[5]),
        'HumanFeedback': (paired[6], paired[7]),
    }
    fontsize = 30
    labelsize = 22

    def remove_lower_letters(input_string):
        a = [char for char in input_string if char.isupper()]
        return ''.join(a)

    fig, axs = plt.subplots(1, data_dict['LlavaGuard'].shape[1] - 2, subplot_kw=dict(projection='polar'),
                            figsize=(20, 6))
    dist_overview_done = False
    for model, data in data_dict.items():
        score_type = [k for k in data.keys() if k not in ['c_angle', 'category']]
        ds_categories = data['category'].values

        # Create a figure with multiple subplots

        for i, score in enumerate(score_type):
            # if score == 'num_samples' and dist_overview_done:
            #     continue
            ax = axs[i]
            # make additional lineplot to connect the last and first point
            x = data['c_angle'].values
            y = data[score].values
            color_fill, color = model_colors[model]

            ax.fill(x, y, color=color_fill, alpha=0.3)
            # g = sns.scatterplot(data=data, x='c_angle', y=f'{score}', ax=ax, color=color1)
            g = sns.scatterplot(x=x, y=y, ax=ax, color=color)
            sns.lineplot(x=x, y=y, ax=ax, color=color, linestyle='--')
            sns.lineplot(x=[x[-1], x[0]], y=[y[-1], y[0]], ax=ax, color=color, linestyle='--')

            ax.set_xticks(data['c_angle'])
            # ax.set_xticklabels([*range(len(data))])
            ax.set_xticklabels([remove_lower_letters(cat) for cat in ds_categories], fontsize=labelsize)

            if score == 'Ø Safety Score \n by Category':
                ticks = [*range(0, 101, 25)]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f'{int(t/25)}' for t in ticks], fontsize=labelsize)
                ax.set_ylim(0, 110)
                # ax.set_ylabel('score')
            else:
                sup = 10
                max_m = 40
                ticks = [*range(0, max_m, sup)]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f'{t}' for t in ticks], fontsize=labelsize)
                ax.set_ylim(0, max_m)

            # remove labels from all subplots on x and y axis
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(score, fontsize=fontsize, fontweight='bold')

    # make title for the whole plot
    # plt.suptitle(title, fontsize=20, fontweight='bold')
    # create a legend for the whole plot and place it at the bottom enumerate the categories and place them in the legend
    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    ncols = 2

    models = list(data_dict.keys())
    legend1_txt = models + ([''] * 0 if len(models) % ncols == 0 else [''] * (ncols - len(models) % ncols))
    # legend1_txt += [''] * 0 if len(models) % ncols == 0 else [''] * (ncols - len(models) % ncols)

    # legend2_txt = [f'{i}: {cat}' for i, cat in enumerate(ds_categories)]
    legend2_txt = [f'{remove_lower_letters(cat)}: {cat}' for cat in ds_categories]
    legend2_txt += [''] * 0 if len(legend2_txt) % ncols == 0 else [''] * (ncols - len(legend2_txt) % ncols)

    txt = legend1_txt + legend2_txt
    # first_col_text = [f''] + [f'Categories:'] + [''] * ((len(txt) // ncols) - 2)
    first_col_text = []

    legend_txt = np.array(txt).reshape(-1, ncols)
    # legend_txt = np.concatenate((np.array(first_col_text).reshape(-1, 1), legend_txt), axis=1)
    legend_txt = legend_txt.T.flatten().tolist()
    # create two legends

    m_handles = [Patch(facecolor=model_colors[model][0], edgecolor=model_colors[model][1],
                       label=model) for model in models]
    # first_row_handel = white + m_handles + white * (ncols - len(m_handles))
    first_row_handel = m_handles + white * (ncols - len(m_handles))
    rest_handels = white * (len(legend_txt) - len(first_row_handel))
    first_row_handel = np.array(first_row_handel).reshape(-1, ncols)
    rest_handels = np.array(rest_handels).reshape(-1, ncols)
    handels = np.concatenate((first_row_handel, rest_handels), axis=0)
    # chain the handels
    handels = handels.T.flatten().tolist()

    leg = fig.legend(handels, legend_txt, loc='lower center',
                     bbox_to_anchor=(0.51, -0.55), ncol=ncols,
                     # handleheight=1, handlelength=1,
                     fontsize=labelsize+5)
    # save the plot
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('pdf', 'png'), dpi=300, bbox_inches='tight')
    # close the plot
    plt.close()
