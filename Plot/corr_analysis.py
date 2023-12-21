import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(domain_name):
    base_path = 'Results'

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
    std_devs = [np.std(values) for values in layer_data.values()]
    return means, std_devs


with open('./data/Pfam/protein_domain.txt', 'r') as file:
    lines = file.readlines()
    protein_domains = [line.strip() for line in lines]

fig, axes = plt.subplots(4, 5, figsize=(12, 9))
axes = axes.flatten()

for i, domain in enumerate(protein_domains):
    default_corr, sc_dict, scovar_dict = load_data(domain)

    means1, std_devs1 = calculate_stats(sc_dict)
    means2, std_devs2 = calculate_stats(scovar_dict)

    x_labels = list(range(len(means1)))

    ax = axes[i]
    ax.plot(x_labels, default_corr, '-*', label='Default', color='purple', markersize=3)
    ax.errorbar(x_labels, means1, yerr=std_devs1, fmt='-o', ecolor='orange', markersize=3, capsize=2, color='orange',
                label='Shuffled columns')
    ax.errorbar(x_labels, means2, yerr=std_devs2, fmt='-^', ecolor='blue', markersize=3, capsize=2, color='blue',
                label='Shuffled covariance')

    ax.set_title(domain, fontsize=10)
    ax.set_xlabel('Layers', fontsize=9)
    ax.set_ylabel('Correlation', fontsize=9)

    ax.grid(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0), fontsize=10)

plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()