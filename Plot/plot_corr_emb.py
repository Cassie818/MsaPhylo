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

    # load shuffle data
    sc_dict = {}
    scovar_dict = {}
    sr_dict = {}

    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            sc_file_path = os.path.join(root, dir, f'{domain_name}_sc_ev_and_euclidean_analysis_emb.csv')
            scovar_file_path = os.path.join(root, dir, f'{domain_name}_scovar_ev_and_euclidean_analysis_emb.csv')
            sr_file_path = os.path.join(root, dir, f'{domain_name}_sr_ev_and_euclidean_analysis_emb.csv')

            if os.path.exists(sc_file_path):
                sc_data = pd.read_csv(sc_file_path)["Correlation"].to_list()
                sc_dict[dir] = sc_data

            if os.path.exists(scovar_file_path):
                scovar_data = pd.read_csv(scovar_file_path)["Correlation"].to_list()
                scovar_dict[dir] = scovar_data

            if os.path.exists(sr_file_path):
                sr_data = pd.read_csv(sr_file_path)["Correlation"].to_list()
                sr_dict[dir] = sr_data

    return default_corr, sc_dict, scovar_dict, sr_dict


def calculate_stats(data_dict):
    list_length = len(next(iter(data_dict.values())))
    layer_data = {i: [data_dict[key][i] for key in data_dict] for i in range(list_length)}
    means = [np.mean(values) for values in layer_data.values()]
    stds = [np.std(values) for values in layer_data.values()]
    return means, stds


with open('./data/Pfam/protein_domain.txt', 'r') as file:
    lines = file.readlines()
    protein_domains = [line.strip() for line in lines]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['arial']
fig, axes = plt.subplots(4, 5, figsize=(14, 8))
ref = pd.read_csv('score/nj_ml_corr.csv')
ref_dict = ref.set_index('ProteinDomain')['Corr'].to_dict()
axes = axes.flatten()

for i, domain in enumerate(protein_domains):

    default_corr, sc_dict, scovar_dict, sr_dict = load_data(domain)
    means1, stds1 = calculate_stats(sc_dict)
    means2, stds2 = calculate_stats(scovar_dict)
    means3, stds3 = calculate_stats(sr_dict)
    ref = ref_dict[domain]

    ax = axes[i]
    ax.plot(range(1, 13), default_corr, '-^', label='Original', color='#505050', markersize=4)
    ax.errorbar(range(1, 13), means1, yerr=stds1, color='lightpink',
                fmt='-x', label='Shuffled Columns', elinewidth=1, markersize=4)
    ax.errorbar(range(1, 13), means2, yerr=stds2, color='lightskyblue',
                fmt='-*', label='Shuffled within Columns', elinewidth=1, markersize=4)
    ax.errorbar(range(1, 13), means3, yerr=stds3, color='darkseagreen',
                fmt='-o', markersize=4, label='Shuffled within Rows', elinewidth=1)
    ax.axhline(y=ref, linestyle='--', color='dimgray',
               label='Reference', linewidth=1)
    ax.text(11, ref + 0.05, f'{ref:.2f}', color='dimgray', ha='center', va='bottom', fontsize=12)

    ax.set_title(domain, fontsize=14)
    if i >= 15:
        ax.set_xlabel('Layer', fontsize=14)
    if i % 5 == 0:
        ax.set_ylabel('Spearman\'s rho', fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
    else:
        ax.set_yticklabels([])
        ax.set_yticks([])

    x_labels = list(range(1, 13, 2))
    ax.set_xticks(x_labels)
    ax.tick_params(axis='x', labelsize=14)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0), fontsize=10)

plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
