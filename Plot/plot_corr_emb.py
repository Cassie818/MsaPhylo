import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def load_correlation_data(base_path: str, domain_name: str, analysis_type: str):
    """
    Load correlation data for a given analysis type.
    """
    data_dict = {}
    file_suffix = f"{domain_name}_{analysis_type}_ev_and_euclidean_analysis_emb.csv"

    if analysis_type == 'default':
        file_path = os.path.join(base_path, file_suffix)
        if os.path.exists(file_path):
            data_dict['default'] = pd.read_csv(file_path)["Correlation"].to_list()
    else:
        for root, dirs, files in os.walk(base_path):
            for dir in dirs:
                file_path = os.path.join(root, dir, file_suffix)
                if os.path.exists(file_path):
                    data_dict[dir] = pd.read_csv(file_path)["Correlation"].to_list()

    return data_dict


def calculate_stats(data_dict: Dict):
    """
    Calculate mean and standard deviation for data in data_dict.
    """
    list_length = len(next(iter(data_dict.values())))
    layer_data = {i: [data_dict[key][i] for key in data_dict] for i in range(list_length)}
    means = [np.mean(values) for values in layer_data.values()]
    stds = [np.std(values) for values in layer_data.values()]
    return means, stds


def plot_correlations(protein_domains: List, base_path: str, reference_csv: str):
    """
    Plot correlation data for multiple protein domains.
    """
    ref = pd.read_csv(reference_csv)
    ref_dict = ref.set_index('ProteinDomain')['Corr'].to_dict()

    plt.rcParams['font.family'] = ['DejaVu Sans']
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    axes = axes.flatten()

    for i, domain in enumerate(protein_domains):
        ax = axes[i]
        for analysis_type, linestyle, color, label in [
            ('default', '--', 'navy', 'Default'),
            ('sc', '-.', '#ff7f0e', 'Shuffled columns'),
            ('scovar', '--', 'blue', 'Shuffled covariance')]:
            data_dict = load_correlation_data(base_path, domain, analysis_type)
            if analysis_type == 'default':
                if 'default' in data_dict:
                    ax.plot(range(1, 13), data_dict['default'], linestyle=linestyle, color=color, label=label)
            else:
                means, stds = calculate_stats(data_dict)
                ax.errorbar(range(1, 13), means, yerr=stds, linestyle=linestyle, color=color, label=label, elinewidth=1)

        ref_corr = ref_dict.get(domain, 0)
        ax.axhline(y=ref_corr, linestyle='-', color='r', label='Reference', linewidth=1)
        ax.set_title(domain, fontsize=12)

        if i % 5 == 0:
            ax.set_ylabel('Rho', fontsize=10)
        if i // 5 == 3:
            ax.set_xlabel('Layer', fontsize=10)

        ax.set_xticks(list(range(1, 13, 2)))
        ax.grid(False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0), fontsize=10)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.show()


def load_protein_domains(file_path: str):
    """
    Load a list of protein domains from a file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]


if __name__ == "__main__":
    protein_domains = load_protein_domains('./data/Pfam/protein_domain.txt')
    base_path = 'Results'
    reference_csv = 'score/nj_ml_corr.csv'
    plot_correlations(protein_domains, base_path, reference_csv)
