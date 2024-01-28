import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functools import reduce
import matplotlib.colors as mcolors


def load_csv_and_process(file_path, pattern, type_name):
    df = pd.read_csv(file_path)
    df = df[df['FileName'].str.contains(pattern)]
    df = df[df['FileName'].str.count('_') > 1]
    df['ProteinDomain'] = df['FileName'].str.extract(r'(PF\d+)_')
    df['Layers'] = df['FileName'].str.extract(f'{type_name}_(\d+)_')[0].astype(int)
    df['Heads'] = df['FileName'].str.extract(f'{type_name}_\d+_(\d+)')[0].astype(int)
    df = df.sort_values(by=['ProteinDomain', 'Layers', 'Heads']).drop(['FileName'], axis=1)
    df = df.reindex(columns=['ProteinDomain', 'Layers', 'Heads', 'RFScore'])
    return df


def merge_and_calculate_stats(dfs):
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['ProteinDomain', 'Layers', 'Heads'], how='outer'),
                       dfs)
    df_reduced = merged_df.drop(columns=['ProteinDomain', 'Layers', 'Heads'])
    merged_df['RFScore'] = df_reduced.mean(axis=1)
    merged_df['Variance'] = df_reduced.var(axis=1)
    return merged_df[['ProteinDomain', 'RFScore', 'Variance']]


def load_data(typ):
    base_path = 'RFscore'

    # Load default data
    default_rf_file = os.path.join(base_path, f'{typ}_rf_score_default.csv')
    final_default = None
    if os.path.exists(default_rf_file):
        final_default = load_csv_and_process(default_rf_file, 'default', 'default')

    # Load sc data
    filepaths = [f'RFscore/{typ}_rf_score_rep{i}.csv' for i in range(1, 6)]
    sc_list = [load_csv_and_process(path, 'sc_', 'sc') for path in filepaths]
    final_sc = merge_and_calculate_stats(sc_list)

    # Load scovar data
    scovar_list = [load_csv_and_process(path, 'scovar', 'scovar') for path in filepaths]
    final_scovar = merge_and_calculate_stats(scovar_list)

    return final_default, final_sc, final_scovar


def extract_data(df, protein_domain):
    data = df[df['ProteinDomain'] == protein_domain]
    return data['RFScore'].to_numpy().reshape(12, 12)


def extract_var(df, protein_domain):
    var = df[df['ProteinDomain'] == protein_domain]
    return var['Variance'].to_numpy().reshape(12, 12)


def plot_heatmap(ax, data, vmax):
    x_labels = [str(i) for i in range(1, 13)]
    y_labels = [str(i) for i in range(1, 13)]
    cdict = {'red': [(0.0, 1.0, 1.0),
                     (1.0, 1.0, 1.0)],
             'green': [(0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0)],
             'blue': [(0.0, 1.0, 1.0),
                      (1.0, 0.0, 0.0)]}
    red_white_cmap = mcolors.LinearSegmentedColormap('RedWhite', cdict)
    im = ax.imshow(data, cmap=red_white_cmap, aspect='equal', vmin=0, vmax=vmax)
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


def plot_protein_domains(default_df, sc_df, scovar_df, prot_domains):
    fig, axs = plt.subplots(4, 3, figsize=(9, 12),
                            gridspec_kw={"width_ratios": [10, 10, 10]},
                            constrained_layout=True)

    typs = ['Default', 'Shuffled columns', 'Shuffled covariance']

    for i, protein_domain in enumerate(prot_domains):
        default = extract_data(default_df, protein_domain)
        sc = extract_data(sc_df, protein_domain)
        scovar = extract_data(scovar_df, protein_domain)
        vmax = np.max([default, sc, scovar])

        var_of_sc = extract_var(sc_df, protein_domain)
        var_of_scovar = extract_var(scovar_df, protein_domain)

        for j, typ in enumerate(typs):
            if typ == 'Default':
                heatmap = plot_heatmap(axs[i, j], default, vmax)
            elif typ == 'Shuffled columns':
                heatmap = plot_heatmap(axs[i, j], sc, vmax)
            else:
                heatmap = plot_heatmap(axs[i, j], scovar, vmax)

            if i == 3:
                axs[i, j].set_xlabel("Head", fontsize=12)
            if i == 0:
                axs[0, j].set_title(typ, fontsize=12)
            if j == 0:
                axs[i, j].set_ylabel(f"{protein_domain}\nLayer", fontsize=12)
            if j == 2:
                fig.colorbar(heatmap, ax=axs[i, j])
                overlay_variance(var_of_scovar, axs[i, j])
            if j == 1:
                overlay_variance(var_of_sc, axs[i, j])

    plt.subplots_adjust(wspace=0.01, hspace=0.35)
    plt.show()


if __name__ == '__main__':
    selected_prot_domains = ['PF00066', 'PF00168', 'PF00484', 'PF00672']
    final_default, final_sc, final_scovar = load_data('ml')
    plot_protein_domains(final_default, final_sc, final_scovar, selected_prot_domains)
