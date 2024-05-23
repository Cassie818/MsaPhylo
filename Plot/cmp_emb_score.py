import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.colors as mcolors
from pandas import DataFrame
from typing import List, Dict
from numpy.typing import ArrayLike


def load_data(base_path: str, default_file_name: str):
    """
    Loads attention score data for default, shuffled column, and shuffled covariance MSAs from a specified base path.
    """
    default_file_path = os.path.join(base_path, default_file_name)

    if not os.path.exists(default_file_path):
        raise FileNotFoundError(f"{default_file_path} doesn't exist!")

    # Load default MSA data
    data = pd.read_csv(default_file_path)
    data['ProteinDomain'] = data['FileName'].str.extract(r'(PF\d+)_')
    data['Layers'] = data['FileName'].str.extract('_(\d+)')[0].astype(int)
    data = data.sort_values(by=['ProteinDomain', 'Layers']).drop(['FileName'], axis=1)
    data = data.reindex(columns=['ProteinDomain', 'Layers', 'NJRFScore',
                                 'MLRFScore', 'NJCI', 'MLCI'])

    # Initialize dictionaries to store shuffled data
    sc_dict = {'NJRFScore': [], 'MLRFScore': [], 'NJCI': [], 'MLCI': []}
    scovar_dict = {'NJRFScore': [], 'MLRFScore': [], 'NJCI': [], 'MLCI': []}
    sr_dict = {'NJRFScore': [], 'MLRFScore': [], 'NJCI': [], 'MLCI': []}

    # Function to process shuffling files
    def process_files(directory, typ='scovar'):
        emb_file_path = os.path.join(directory, default_file_name)

        if os.path.exists(emb_file_path):
            emb_file_data = pd.read_csv(emb_file_path)

            if typ == 'scovar':
                filtered_data = emb_file_data[emb_file_data['FileName'].str.contains('scovar')]
                target_dict = scovar_dict
            elif typ == 'sr':
                filtered_data = emb_file_data[emb_file_data['FileName'].str.contains('sr')]
                target_dict = sr_dict
            else:
                filtered_data = emb_file_data[
                    ~emb_file_data['FileName'].str.contains('scovar') & ~emb_file_data['FileName'].str.contains('sr')]
                target_dict = sc_dict

            for key in target_dict.keys():
                target_dict[key].append(filtered_data[['FileName', key]])

    # Load Shuffled column and covariance MSA data
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            process_files(dir_path, 'sc')
            process_files(dir_path, 'scovar')
            process_files(dir_path, 'sr')

    return data, sc_dict, scovar_dict, sr_dict


def merge_and_calculate_stats(dfs: DataFrame, typ: str) -> DataFrame:
    merged_df = dfs[0]
    for i in range(1, len(dfs)):
        suffix_1 = f'_df{i}'
        suffix_2 = f'_df{i + 1}'
        while any(col + suffix_1 in merged_df.columns for col in dfs[i].columns if col != 'FileName'):
            suffix_1 += '_'
        while any(col + suffix_2 in dfs[i].columns for col in merged_df.columns if col != 'FileName'):
            suffix_2 += '_'
        merged_df = merged_df.merge(dfs[i], on='FileName', suffixes=(suffix_1, suffix_2))
    df = pd.DataFrame()
    df['ProteinDomain'] = merged_df['FileName'].str.extract(r'(PF\d+)_')
    df['Layers'] = merged_df['FileName'].str.extract(f'_{typ}(\d+)')[0].astype(int)
    df['Mean'] = merged_df.iloc[:, -5:].mean(axis=1)
    df['Std'] = merged_df.iloc[:, -5:].std(axis=1)
    df.sort_values(by=['ProteinDomain', 'Layers'], inplace=True)
    df = df.reindex(columns=['ProteinDomain', 'Layers', 'Mean', 'Std'])
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


def overlay_variance(var_data: ArrayLike, ax):
    max_variance = np.max(var_data)

    for i in range(var_data.shape[0]):
        for j in range(var_data.shape[1]):
            normalized_var = var_data[i, j] / max_variance
            circle_size = normalized_var * 100
            ax.scatter(j, i, s=circle_size, color='gray', alpha=0.5)


def extract_data(data: DataFrame, protein_domain: List[str], metrics: str, typ: str):
    if typ == 'default':
        df = data[data['ProteinDomain'] == protein_domain]
        val = df[metrics].to_numpy()
        return val

    else:
        score = data[metrics]
        df = merge_and_calculate_stats(score, typ)
        data = df[df['ProteinDomain'] == protein_domain]
        mean = data['Mean'].to_numpy()
        std = data['Std'].to_numpy()
        return mean, std


def plot_protein_domains(default_df: DataFrame, sc_dict: Dict[str, int],
                         scovar_dict: Dict[str, int], sr_dict: Dict[str, int],
                         prot_domains: List[str], metrics: str, reference_csv: str):
    ref = pd.read_csv(reference_csv)
    if 'CI' in metrics:
        ref_dict = ref.set_index('ProteinDomain')['CI'].to_dict()
    else:
        ref_dict = ref.set_index('ProteinDomain')['RFScore'].to_dict()

    # plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['arial']

    fig, axs = plt.subplots(4, 5, figsize=(14, 8))
    axs = axs.flatten()

    typs = ['Default', 'Shuffled columns', 'Shuffled covariance', 'Shuffled rows']
    layers = list(range(1, 13))

    for i, protein_domain in enumerate(prot_domains):

        default = extract_data(default_df, protein_domain, metrics, 'default')
        sc_avg, sc_std = extract_data(sc_dict, protein_domain, metrics, 'sc')
        scovar_avg, scovar_std = extract_data(scovar_dict, protein_domain, metrics, 'scovar')
        sr_avg, sr_std = extract_data(sr_dict, protein_domain, metrics, 'sr')

        ax = axs[i]
        ax.plot(layers, default, '-^', markersize=4, color='#505050', label='Default')
        ax.errorbar(layers, sc_avg, yerr=sc_std, fmt='-x', markersize=4, color='lightpink',
                    label='Shuffled Positions')
        ax.errorbar(layers, scovar_avg, yerr=scovar_std, fmt='-*', markersize=4, color='lightskyblue',
                    label='Shuffled Covariance')
        ax.errorbar(layers, sr_avg, yerr=sr_std, fmt='-o', markersize=4, color='darkseagreen',
                    label='Shuffled Rows')

        ref_val = ref_dict.get(protein_domain, 0)
        ax.axhline(y=ref_val, linestyle='--', color='dimgray', label='Reference', linewidth=1)
        ax.text(11, ref_val + 0.05, f'{ref_val:.2f}', color='dimgray', ha='center', va='bottom', fontsize=12)
        ax.set_title(protein_domain, fontsize=14)
        ax.set_xticks(range(1, 13, 2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        if i >= 15:
            ax.set_xlabel('Layer', fontsize=14)

        if i % 5 == 0:

            if 'CI' in metrics:
                ax.set_ylabel('CI Score', fontsize=14)
            else:
                ax.set_ylabel('RF Score', fontsize=14)
        else:
            ax.set_yticklabels([])
            ax.set_yticks([])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0), fontsize=12)

    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


if __name__ == '__main__':
    base_path = 'score'
    default_file_name = 'emb_score.csv'
    reference_csv = 'score/nj_ml_score.csv'
    metrics = 'MLRFScore'
    with open('./data/Pfam/protein_domain.txt', 'r') as file:
        lines = file.readlines()
    prot_domains = [line.strip() for line in lines]
    attn_data, sc_dict, scovar_dict, sr_dict = load_data(base_path, default_file_name)
    plot_protein_domains(attn_data, sc_dict, scovar_dict, sr_dict, prot_domains, metrics, reference_csv)
