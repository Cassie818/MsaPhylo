import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from functools import reduce
import matplotlib.colors as mcolors
import warnings

warnings.filterwarnings('ignore')


def load_data(base_path: str = 'score', default_file_name: str = 'attn_score.csv'):
    """
    Loads attention score data for default, shuffled column,
    shuffled covariance, and shuffled rows MSAs from a specified base path.
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
                                           'NJRFScore', 'MLRFScore', 'NJCI', 'MLCI'])

    # Initialize dictionaries to store shuffled column and covariance data
    sc_dict = {'NJRFScore': [], 'MLRFScore': [], 'NJCI': [], 'MLCI': []}
    scovar_dict = {'NJRFScore': [], 'MLRFScore': [], 'NJCI': [], 'MLCI': []}
    sr_dict = {'NJRFScore': [], 'MLRFScore': [], 'NJCI': [], 'MLCI': []}

    # Function to process shuffling MSA files
    def process_files(directory, typ='scovar'):
        attn_file_path = os.path.join(directory, default_file_name)

        if os.path.exists(attn_file_path):
            attn_file_data = pd.read_csv(attn_file_path)

            if typ == 'scovar':
                filtered_data = attn_file_data[attn_file_data['FileName'].str.contains('scovar')]
                target_dict = scovar_dict
            elif typ == 'sr':
                filtered_data = attn_file_data[attn_file_data['FileName'].str.contains('sr')]
                target_dict = sr_dict
            else:
                filtered_data = attn_file_data[
                    ~attn_file_data['FileName'].str.contains('scovar') & ~attn_file_data['FileName'].str.contains('sr')]
                target_dict = sc_dict

            for key in target_dict.keys():
                target_dict[key].append(filtered_data[['FileName', key]])

    # Load Shuffled column, covariance and rows MSA data
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            process_files(dir_path, 'sc')
            process_files(dir_path, 'scovar')
            process_files(dir_path, 'sr')

    return attn_data, sc_dict, scovar_dict, sr_dict


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


def plot_protein_domains(default_df, sc_dict, scovar_dict,
                         sr_dict, prot_domains, metrics):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['arial']
    fig, axs = plt.subplots(4, 4, figsize=(10, 10),
                            gridspec_kw={"width_ratios": [10, 10, 10, 13]})

    typs = ['Default', 'Shuffled Positions', 'Shuffled Covariance', 'Shuffled Rows']

    for i, protein_domain in enumerate(prot_domains):

        default = extract_data(default_df, protein_domain, metrics, 'default')
        sc_avg, sc_var = extract_data(sc_dict, protein_domain, metrics, 'sc')
        scovar_avg, scovar_var = extract_data(scovar_dict, protein_domain, metrics, 'scovar')
        sr_avg, sr_var = extract_data(sr_dict, protein_domain, metrics, 'sr')

        vmax = np.max([default, sc_avg, scovar_avg, sr_avg])
        vmin = np.min([default, sc_avg, scovar_avg, sr_avg])

        for j, typ in enumerate(typs):

            if typ == 'Default':
                heatmap = plot_heatmap(axs[i, j], default, vmin, vmax)
            elif typ == 'Shuffled Positions':
                heatmap = plot_heatmap(axs[i, j], sc_avg, vmin, vmax)
            elif typ == 'Shuffled Covariance':
                heatmap = plot_heatmap(axs[i, j], scovar_avg, vmin, vmax)
            else:
                heatmap = plot_heatmap(axs[i, j], sr_avg, vmin, vmax)

            if i == 3:
                axs[i, j].set_xlabel("Head", fontsize=12)
            if i == 0:
                axs[0, j].set_title(typ, fontsize=12)
            if j == 0:
                axs[i, j].set_ylabel(f"{protein_domain}\nLayer", fontsize=12)
            if j == 1:
                overlay_variance(sc_var, axs[i, j])
            if j == 2:
                overlay_variance(scovar_var, axs[i, j])
            if j == 3:
                fig.colorbar(heatmap, ax=axs[i, j])
                overlay_variance(sr_var, axs[i, j])

    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    plt.show()


if __name__ == '__main__':
    base_path = 'score'
    default_file_name = 'attn_score.csv'
    metrics = 'NJCI'
    prot_domains = ['PF00168', 'PF12172', 'PF14317', 'PF20171']
    attn_data, sc_dict, scovar_dict, sr_dict = load_data(base_path, default_file_name)
    plot_protein_domains(attn_data, sc_dict, scovar_dict, sr_dict, prot_domains, metrics)
