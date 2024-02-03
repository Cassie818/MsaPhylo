import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager


def load_data(domain_name):
    base_path = 'Results2'

    # load default data
    default_corr_file = os.path.join(base_path, f'{domain_name}_default_ev_and_euclidean_analysis_emb.csv')
    if os.path.exists(default_corr_file):
        default_corr = pd.read_csv(default_corr_file)["Correlation"].to_list()
    else:
        default_corr = []

    # load sc data
    sc_dict = {}
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            file_path = os.path.join(root, dir, f'{domain_name}_sc_ev_and_euclidean_analysis_emb.csv')
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)["Correlation"].to_list()
                sc_dict[dir] = data

    # load scovar data
    scovar_dict = {}
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            file_path = os.path.join(root, dir, f'{domain_name}_scovar_ev_and_euclidean_analysis_emb.csv')
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)["Correlation"].to_list()
                scovar_dict[dir] = data

    return default_corr, sc_dict, scovar_dict


def calculate_stats(data_dict):
    list_length = len(next(iter(data_dict.values())))
    layer_data = {i: [data_dict[key][i] for key in data_dict] for i in range(list_length)}
    means = [np.mean(values) for values in layer_data.values()]
    stds = [np.std(values) for values in layer_data.values()]
    return means, stds


with open('./data/Pfam/protein_domain.txt', 'r') as file:
    lines = file.readlines()
    protein_domains = [line.strip() for line in lines]

matplotlib.rcParams['font.family'] = ['DejaVu Sans']
fig, axes = plt.subplots(4, 5, figsize=(12, 9))
axes = axes.flatten()

for i, domain in enumerate(protein_domains):

    default_corr, sc_dict, scovar_dict = load_data(domain)
    means1, stds1 = calculate_stats(sc_dict)
    means2, stds2 = calculate_stats(scovar_dict)

    ax = axes[i]
    ax.plot(range(1, 13), default_corr, color='navy', linestyle='--', label='Default')
    ax.errorbar(range(1, 13), means1, yerr=stds1, color='#ff7f0e',
                linestyle='-.', label='Shuffled columns', elinewidth=1)
    ax.errorbar(range(1, 13), means2, yerr=stds2, color='blue',
                linestyle='--', label='Shuffled covariance', elinewidth=1)

    ax.set_title(domain, fontsize=12)
    if i >= 15:
        ax.set_xlabel('Layer', fontsize=10)
    if i % 5 == 0:
        ax.set_ylabel('Rho', fontsize=10)

    x_labels = list(range(1, 13, 2))
    ax.set_xticks(x_labels)
    ax.grid(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0), fontsize=10)

plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
