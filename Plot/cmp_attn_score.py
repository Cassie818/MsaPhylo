import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functools import reduce
import matplotlib.colors as mcolors
import code


def load_data(base_path, default_file_name):
    """
    Loads attention score data for default, shuffled column, and shuffled covariance MSAs from a specified base path.
    """
    default_file_path = os.path.join(base_path, default_file_name)

    if not os.path.exists(default_file_path):
        raise FileNotFoundError(f"{default_file_path} doesn't exist!")

    # Load default MSA data
    attn_data = pd.read_csv(default_file_path)
    attn_data['ProteinDomain'] = attn_data['FileName'].str.extract(r'(PF\d+)_')
    attn_data['Layers'] = attn_data['FileName'].str.extract('default_(\d+)_')[0].astype(int)
    attn_data['Heads'] = attn_data['FileName'].str.extract('default_\d+_(\d+)')[0].astype(int)
    attn_data = attn_data.sort_values(by=['ProteinDomain', 'Layers', 'Heads']).drop(['FileName'], axis=1)
    attn_data = attn_data.reindex(columns=['ProteinDomain', 'Layers', 'Heads',
                                           'NJRFScore', 'MLRFScore', 'NJCID', 'MLCID'])

    # Initialize dictionaries to store shuffled column and covariance data
    sc_dict = {'NJRFScore': [], 'MLRFScore': [], 'NJCID': [], 'MLCID': []}
    scovar_dict = {'NJRFScore': [], 'MLRFScore': [], 'NJCID': [], 'MLCID': []}

    # Function to process files in directories
    def process_files(directory, is_scovar=False):
        attn_file_path = os.path.join(directory, default_file_name)
        if os.path.exists(attn_file_path):
            attn_file_data = pd.read_csv(attn_file_path)
            if is_scovar:
                filtered_data = attn_file_data[attn_file_data['FileName'].str.contains('scovar')]
                target_dict = scovar_dict
            else:
                filtered_data = attn_file_data[~attn_file_data['FileName'].str.contains('scovar')]
                target_dict = sc_dict

            for key in target_dict.keys():
                target_dict[key].append(filtered_data[['FileName', key]])

    # Load Shuffled column and covariance MSA data
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            process_files(dir_path, is_scovar=False)
            process_files(dir_path, is_scovar=True)

    return attn_data, sc_dict, scovar_dict


def merge_and_calculate_stats(dfs, typ):
    df = pd.DataFrame()
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['FileName'], how='outer'), dfs)
    df['ProteinDomain'] = merged_df['FileName'].str.extract(r'(PF\d+)_')
    df['Layers'] = merged_df['FileName'].str.extract(f'{typ}_(\d+)_')[0].astype(int)
    df['Heads'] = merged_df['FileName'].str.extract(f'{typ}_\d+_(\d+)')[0].astype(int)
    df['Mean'] = merged_df.mean(axis=1)
    df['Variance'] = merged_df.var(axis=1)
    df.sort_values(by=['ProteinDomain', 'Layers', 'Heads'], inplace=True)
    df = df.reindex(columns=['ProteinDomain', 'Layers', 'Heads', 'Mean', 'Variance'])
    return df


def plot_heatmap(ax, data, vmin, vmax):
    x_labels = [str(i) for i in range(1, 13)]
    y_labels = [str(i) for i in range(1, 13)]

    cdict = {'red': [(0.0, 1.0, 1.0),
                     (1.0, 1.0, 1.0)],
             'green': [(0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0)],
             'blue': [(0.0, 1.0, 1.0),
                      (1.0, 0.0, 0.0)]}

    red_white_cmap = mcolors.LinearSegmentedColormap('RedWhite', cdict)
    im = ax.imshow(data, cmap=red_white_cmap, aspect='equal', vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(0, 12, 2))
    ax.set_yticks(np.arange(0, 12, 2))
    ax.set_xticklabels(x_labels[::2])
    ax.set_yticklabels(y_labels[::2])

    return im


def overlay_variance(var_data, ax):
    max_variance = np.max(var_data)
    for i in range(var_data.shape[0]):
        for j in range(var_data.shape[1]):
            normalized_var = var_data[i, j] / max_variance
            circle_size = normalized_var * 100
            ax.scatter(j, i, s=circle_size, color='gray', alpha=0.5)


def extract_data(data, protein_domain, metrics, typ):
    if typ == 'default':
        df = data[data['ProteinDomain'] == protein_domain]
        val = df[metrics].to_numpy().reshape(12, 12)
        return val

    else:
        score = data[metrics]
        df = merge_and_calculate_stats(score, typ)
        data = df[df['ProteinDomain'] == protein_domain]
        mean = data['Mean'].to_numpy().reshape(12, 12)
        var = data['Variance'].to_numpy().reshape(12, 12)
        return mean, var


def plot_protein_domains(default_df, sc_dict, scovar_dict, prot_domains, metrics):
    fig, axs = plt.subplots(4, 3, figsize=(9, 12),
                            gridspec_kw={"width_ratios": [10, 10, 11.5]})

    typs = ['Default', 'Shuffled columns', 'Shuffled covariance']

    for i, protein_domain in enumerate(prot_domains):

        default = extract_data(default_df, protein_domain, metrics, 'default')
        sc_avg, sc_var = extract_data(sc_dict, protein_domain, metrics, 'sc')
        scovar_avg, scovar_var = extract_data(scovar_dict, protein_domain, metrics, 'scovar')

        vmax = np.max([default, sc_avg, scovar_avg])
        vmin = np.min([default, sc_avg, scovar_avg])

        for j, typ in enumerate(typs):

            if typ == 'Default':
                heatmap = plot_heatmap(axs[i, j], default, vmin, vmax)
            elif typ == 'Shuffled columns':
                heatmap = plot_heatmap(axs[i, j], sc_avg, vmin, vmax)
            else:
                heatmap = plot_heatmap(axs[i, j], scovar_avg, vmin, vmax)

            if i == 3:
                axs[i, j].set_xlabel("Head", fontsize=12)
            if i == 0:
                axs[0, j].set_title(typ, fontsize=12)
            if j == 0:
                axs[i, j].set_ylabel(f"{protein_domain}\nLayer", fontsize=12)
            if j == 2:
                fig.colorbar(heatmap, ax=axs[i, j])
                overlay_variance(scovar_var, axs[i, j])
            if j == 1:
                overlay_variance(sc_var, axs[i, j])

    plt.subplots_adjust(wspace=0.01, hspace=0.35)
    plt.show()


if __name__ == '__main__':
    base_path = 'score'
    default_file_name = 'attn_score.csv'
    attn_data, sc_dict, scovar_dict = load_data(base_path, default_file_name)
    prot_domains = ['PF00168', 'PF12172', 'PF14317', 'PF20171']
    metrics = 'NJCID'
    plot_protein_domains(attn_data, sc_dict, scovar_dict, prot_domains, metrics)
