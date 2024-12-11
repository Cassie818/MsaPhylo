import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(domain_name):
    default_dict = {}
    sc_dict = {}
    scovar_dict = {}
    sr_dict = {}
    base_path = 'Results'

    default_corr_file = os.path.join(base_path, f'{domain_name}_default_ev_analysis_col_attention.csv')
    if os.path.exists(default_corr_file):
        default_corr = pd.read_csv(default_corr_file)["Correlation"].to_numpy().reshape(12, 12)
        default_dict[f'{domain_name}_Default'] = default_corr

    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)

            sc_file = os.path.join(dir_path, f'{domain_name}_sc_ev_analysis_col_attention.csv')
            if os.path.exists(sc_file):
                sc = pd.read_csv(sc_file)["Correlation"].to_numpy().reshape(12, 12)
                sc_dict[f'{dir}_{domain_name}_Shuffled_columns'] = sc

            scovar_file = os.path.join(dir_path, f'{domain_name}_scovar_ev_analysis_col_attention.csv')
            if os.path.exists(scovar_file):
                scovar = pd.read_csv(scovar_file)["Correlation"].to_numpy().reshape(12, 12)
                scovar_dict[f'{dir}_{domain_name}_Shuffled_covariance'] = scovar

            sr_file = os.path.join(dir_path, f'{domain_name}_sr_ev_analysis_col_attention.csv')
            if os.path.exists(sr_file):
                sr = pd.read_csv(sr_file)["Correlation"].to_numpy().reshape(12, 12)
                sr_dict[f'{dir}_{domain_name}_Shuffled_rows'] = sr

    return default_dict, sc_dict, scovar_dict, sr_dict


def calculate_mean_variance(domain_dict):
    first_key = next(iter(domain_dict))
    rows, cols = domain_dict[first_key].shape

    mean_matrix = np.zeros((rows, cols))
    variance_matrix = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            values = [matrix[i, j] for matrix in domain_dict.values()]
            mean_matrix[i, j] = np.mean(values)
            variance_matrix[i, j] = np.var(values)

    return mean_matrix, variance_matrix


def load_domain_abs_vmax(domain_name, data_dict):
    all_data = []

    for data in data_dict[domain_name].values():
        all_data.append(data)
        domain_vmins = min([d.min() for d in all_data])
        domain_vmaxs = max([d.max() for d in all_data])
        domain_abs_vmax = max(abs(domain_vmins), abs(domain_vmaxs))

    return domain_abs_vmax


def create_domain_heatmaps(mean_dict, var_dict, protein_domains, domain_abs_max, num_protein=4):
    """
    Create and display heatmaps for protein domains with variance overlay.

    :param mean_dict: Dictionary containing the mean heatmap data for each protein domain.
    :param var_dict: Dictionary containing the variance data for each protein domain.
    :param protein_domains: List of protein domain names.
    :param domain_abs_max: Dictionary of maximum absolute value for each domain.
    :param num_protein: Number of proteins (default is 4).
    """
    typs = ['Default', 'Shuffled_columns', 'Shuffled_covariance', 'Shuffled_rows']
    x_labels = [str(i) for i in range(1, 13)]
    y_labels = [str(i) for i in range(1, 13)]

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['arial']
    fig, axes = plt.subplots(nrows=num_protein, ncols=len(typs), figsize=(10, 10),
                             gridspec_kw={"width_ratios": [10, 10, 10, 12.5]},
                             )

    for i, protein_domain in enumerate(protein_domains):
        pf_info_mean = mean_dict[protein_domain]
        pf_info_var = var_dict[protein_domain]
        for j, typ in enumerate(typs):
            mean_key = f'{protein_domain}_{typ}'
            mean_data = pf_info_mean[mean_key]
            if typ != 'Default':
                var_key = f'{protein_domain}_{typ}'
                var_data = pf_info_var[var_key]
            heatmap = create_heatmap(mean_data,
                                     axes[i, j],
                                     x_labels,
                                     y_labels,
                                     vmin=-domain_abs_max[protein_domain],
                                     vmax=domain_abs_max[protein_domain])

            if j == 0:
                axes[i, j].set_ylabel(f"{protein_domain}\nLayer", fontsize=12)
            if i == num_protein - 1:
                axes[i, j].set_xlabel("Head", fontsize=12)
            if j != 0:
                overlay_variance(var_data, axes[i, j])
            if j == 3:
                fig.colorbar(heatmap, ax=axes[i, j])

    # Set column titles
    axes[0, 0].set_title("Original", fontsize=12)
    axes[0, 1].set_title("Shuffled Columns", fontsize=12)
    axes[0, 2].set_title("Shuffled within Columns", fontsize=12)
    axes[0, 3].set_title("Shuffled within Rows", fontsize=12)

    plt.subplots_adjust(wspace=0.2, hspace=0.15)
    plt.show()


def create_heatmap(data,
                   ax,
                   x_labels,
                   y_labels,
                   vmin=None,
                   vmax=None):
    heatmap = ax.imshow(data, cmap='bwr', aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(0, len(x_labels), 2))
    ax.set_yticks(np.arange(0, len(y_labels), 2))
    ax.set_xticklabels(x_labels[::2], size=12)
    ax.set_yticklabels(y_labels[::2], size=12)
    return heatmap


def overlay_variance(var_data, ax):
    max_variance = np.max(var_data)
    for i in range(var_data.shape[0]):
        for j in range(var_data.shape[1]):
            normalized_var = var_data[i, j] / max_variance
            circle_size = normalized_var * 100
            ax.scatter(j, i, s=circle_size, color='gray', alpha=0.5)


if __name__ == '__main__':
    mean_dict = {}
    var_dict = {}
    domain_abs_vmax = {}

    protein_domains = ['PF13377', 'PF13466', 'PF14317', 'PF20171']
    for domain in protein_domains:
        default_dict, sc_dict, scovar_dict, sr_dict = load_data(domain)
        mean_matrix_sc, variance_matrix_sc = calculate_mean_variance(sc_dict)
        mean_matrix_scovar, variance_matrix_scovar = calculate_mean_variance(scovar_dict)
        mean_matrix_sr, variance_matrix_sr = calculate_mean_variance(sr_dict)

        mean_dict[f'{domain}'] = {f'{domain}_Default': default_dict[f'{domain}_Default'],
                                  f'{domain}_Shuffled_columns': mean_matrix_sc,
                                  f'{domain}_Shuffled_covariance': mean_matrix_scovar,
                                  f'{domain}_Shuffled_rows': mean_matrix_sr}
        var_dict[f'{domain}'] = {f'{domain}_Shuffled_columns': variance_matrix_sc,
                                 f'{domain}_Shuffled_covariance': variance_matrix_scovar,
                                 f'{domain}_Shuffled_rows': variance_matrix_sr}
        domain_abs_vmax[domain] = load_domain_abs_vmax(domain, mean_dict)
    create_domain_heatmaps(mean_dict, var_dict, protein_domains, domain_abs_vmax)