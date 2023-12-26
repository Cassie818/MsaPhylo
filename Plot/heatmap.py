import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


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


# Function to create a heatmap for the given data and axis
def create_heatmap(data, ax, title, x_labels, y_labels, vmin=None, vmax=None):
    heatmap = ax.imshow(data, cmap='bwr', aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_title(title, size=10, loc='center')
    ax.set_xticks(np.arange(0, len(x_labels), 2))
    ax.set_yticks(np.arange(0, len(y_labels), 2))
    ax.set_xticklabels(x_labels[::2])
    ax.set_yticklabels(y_labels[::2])
    return heatmap


protein_domains = ['PF00066', 'PF00168', 'PF00484', 'PF00672']

data_dict = {}
domain_vmins = {}
domain_vmaxs = {}

for domain in protein_domains:
    data_dict[domain] = load_data(domain)

for domain in protein_domains:
    all_data = []
    for key, data in data_dict[domain].items():
        all_data.append(data)
    domain_vmins[domain] = min([d.min() for d in all_data])
    domain_vmaxs[domain] = max([d.max() for d in all_data])

um_protein = 4
typs = ['Default', 'Shuffled_columns', 'Shuffled_covariance']
# Define the labels for the x and y axes
x_labels = [str(i) for i in range(1, 13)]
y_labels = [str(i) for i in range(1, 13)]
fig, axes = plt.subplots(nrows=num_protein, ncols=len(typs), figsize=(10, 12))

for i, protein_domain in enumerate(protein_domains):
    pf_info = data_dict[protein_domain]
    # Iterate over the rows and columns and populate the subplots
    for j, typ in enumerate(typs):
        title = f'{protein_domain}_{typ}'
        if typ == 'Default':
            data = pf_info[f'{protein_domain}_Default']
        elif typ == 'Shuffled_columns':
            data = pf_info[f'{protein_domain}_Shuffled_columns']
        elif typ == 'Shuffled_covariance':
            data = pf_info[f'{protein_domain}_Shuffled_covariance']
        heatmap = create_heatmap(data, axes[i, j], title, x_labels, y_labels, vmin=domain_vmins[protein_domain],
                                 vmax=domain_vmaxs[protein_domain])
        # Add color bars to the rightmost column
        if j == len(typs) - 1:
            fig.colorbar(heatmap, ax=axes[i, j])

        axes[i, j].set_xlabel('Head')
        axes[i, j].set_ylabel('Layer')

# Adjust layout to prevent overlap
plt.subplots_adjust(wspace=0.01, hspace=0.4)

# Display the plot
plt.show()
