import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_domain_abs_vmax(domain_name):
    all_data = []
    for data in data_dict[domain_name].values():
        all_data.append(data)
        domain_vmins = min([d.min() for d in all_data])
        domain_vmaxs = max([d.max() for d in all_data])
        domain_abs_vmax = max(abs(domain_vmins), abs(domain_vmaxs))

    return domain_abs_vmax


def load_data(domain_name):
    domain_dict = {}
    base_path = 'Results'

    # load default data
    default_corr_file = os.path.join('Results', f'{domain_name}_default_ev_analysis_col_attention.csv')
    if os.path.exists(default_corr_file):
        default_corr = pd.read_csv(default_corr_file)["Correlation"].to_numpy().reshape(12, 12)
        domain_dict[f'{domain_name}_Default'] = default_corr

    # load sc data
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            file_path = os.path.join(root, dir, f'{domain_name}_sc_ev_analysis_col_attention.csv')
            if os.path.exists(file_path):
                sc = pd.read_csv(file_path)["Correlation"].to_numpy().reshape(12, 12)
                domain_dict[f'{domain_name}_Shuffled_columns'] = sc

    # load scovar data
    scovar_dict = {}
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            file_path = os.path.join(root, dir, f'{domain_name}_scovar_ev_analysis_col_attention.csv')
            if os.path.exists(file_path):
                scovar = pd.read_csv(file_path)["Correlation"].to_numpy().reshape(12, 12)
                domain_dict[f'{domain_name}_Shuffled_covariance'] = scovar

    return domain_dict


def create_domain_heatmaps(data_dict, protein_domains, domain_abs_max, num_protein=4):
    """
    Create and display heatmaps for protein domains.

    :param data_dict: Dictionary containing the heatmap data for each protein domain.
    :param protein_domains: List of protein domain names.
    :param domain_abs_max: Dictionary of maximum absolute value for each domain.
    :param num_protein: Number of proteins (default is 4).
    """
    typs = ['Default', 'Shuffled_columns', 'Shuffled_covariance']
    x_labels = [str(i) for i in range(1, 13)]
    y_labels = [str(i) for i in range(1, 13)]

    fig, axes = plt.subplots(nrows=num_protein, ncols=len(typs), figsize=(9, 12),
                             gridspec_kw={"width_ratios": [10, 10, 10]},
                             constrained_layout=True)

    for i, protein_domain in enumerate(protein_domains):
        pf_info = data_dict[protein_domain]
        for j, typ in enumerate(typs):
            data_key = f'{protein_domain}_{typ}'
            data = pf_info[data_key]
            heatmap = create_heatmap(data, axes[i, j], x_labels, y_labels, vmin=-domain_abs_max[protein_domain],
                                     vmax=domain_abs_max[protein_domain])

            if j == 0:
                axes[i, j].set_ylabel(f"{protein_domain}\nLayer", fontsize=12)
            if i == num_protein - 1:
                axes[i, j].set_xlabel("Head", fontsize=12)
            if j == 2:
                fig.colorbar(heatmap, ax=axes[i, j])

    # Set column titles
    for j, typ in enumerate(typs):
        axes[0, j].set_title(typ, fontsize=12)

    plt.subplots_adjust(wspace=0.01, hspace=0.35)
    plt.show()


def create_heatmap(data, ax, x_labels, y_labels, vmin=None, vmax=None):
    heatmap = ax.imshow(data, cmap='bwr', aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(0, len(x_labels), 2))
    ax.set_yticks(np.arange(0, len(y_labels), 2))
    ax.set_xticklabels(x_labels[::2])
    ax.set_yticklabels(y_labels[::2])
    return heatmap


data_dict = {}
domain_abs_vmax = {}
protein_domains = ['PF00066', 'PF00168', 'PF00484', 'PF00672']
for domain in protein_domains:
    data_dict[domain] = load_data(domain)
    domain_abs_vmax[domain] = load_domain_abs_vmax(domain)
create_domain_heatmaps(data_dict, protein_domains, domain_abs_vmax)
