import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functools import reduce
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
                                 'MLRFScore', 'NJCID', 'MLCID'])

    # Initialize dictionaries to store shuffled positions and covariance data
    sc_dict = {'NJRFScore': [], 'MLRFScore': [], 'NJCID': [], 'MLCID': []}
    scovar_dict = {'NJRFScore': [], 'MLRFScore': [], 'NJCID': [], 'MLCID': []}

    # Function to process shuffling files
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

    return data, sc_dict, scovar_dict


def merge_and_calculate_stats(dfs: DataFrame, typ: str) -> DataFrame:
    df = pd.DataFrame()
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['FileName'], how='outer'), dfs)
    df['ProteinDomain'] = merged_df['FileName'].str.extract(r'(PF\d+)_')
    df['Layers'] = merged_df['FileName'].str.extract(f'_{typ}(\d+)')[0].astype(int)
    df['Mean'] = merged_df.mean(axis=1)
    df['Std'] = merged_df.std(axis=1)
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
                         scovar_dict: Dict[str, int], prot_domains: List[str],
                         metrics: str, reference_csv: str):
    ref = pd.read_csv(reference_csv)
    if 'CID' in metrics:
        ref_dict = ref.set_index('ProteinDomain')['CID'].to_dict()
    else:
        ref_dict = ref.set_index('ProteinDomain')['RFScore'].to_dict()

    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['arial']

    fig, axs = plt.subplots(4, 5, figsize=(12, 9), sharex=False, sharey=False)
    axs = axs.flatten()

    typs = ['Default', 'Shuffled columns', 'Shuffled covariance']
    layers = list(range(1, 13))

    for i, protein_domain in enumerate(prot_domains):

        default = extract_data(default_df, protein_domain, metrics, 'default')
        sc_avg, sc_std = extract_data(sc_dict, protein_domain, metrics, 'sc')
        scovar_avg, scovar_std = extract_data(scovar_dict, protein_domain, metrics, 'scovar')

        ax = axs[i]
        ax.plot(layers, default, '--', markersize=3, color='#505050', label='Default')
        ax.errorbar(layers, sc_avg, yerr=sc_std, fmt='--', markersize=3, color='#E75480',
                    label='Shuffled Positions')
        ax.errorbar(layers, scovar_avg, yerr=scovar_std, fmt='--', markersize=3, color='#3399FF',
                    label='Shuffled Covariance')
        # Plot the reference line
        ref_corr = ref_dict.get(protein_domain, 0)
        ax.axhline(y=ref_corr, color='#808080', label='Reference', linewidth=1)
        ax.text(11, ref_corr + 0.05, f'{ref_corr:.2f}', color='black', ha='center', va='bottom', fontsize=12)
        ax.set_title(protein_domain, fontsize=14)
        ax.set_xticks(range(1, 13, 2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        if i >= 15:
            ax.set_xlabel('Layer', fontsize=12)

        if i % 5 == 0:

            if 'CID' in metrics:
                ax.set_ylabel('CID', fontsize=12)
            else:
                ax.set_ylabel('RF Score', fontsize=12)
        else:
            ax.set_yticklabels([])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0), fontsize=12)

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
    attn_data, sc_dict, scovar_dict = load_data(base_path, default_file_name)
    plot_protein_domains(attn_data, sc_dict, scovar_dict, prot_domains, metrics, reference_csv)
