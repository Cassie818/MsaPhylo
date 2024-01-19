from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd
import numpy as np


def load_data(typ):
    base_path = 'RFscore'

    # load default data for column attention
    default_rf_file = os.path.join(base_path, f'{typ}_rf_score_default.csv')

    if os.path.exists(default_rf_file):
        default_rf = pd.read_csv(default_rf_file)
        filtered_rf = default_rf[default_rf['FileName'].str.contains('default')]
        filtered_rf['ProteinDomain'] = filtered_rf['FileName'].str.extract(r'(PF\d+)_')
        filtered_rf['Layers'] = filtered_rf['FileName'].str.extract(r'default_(\d+)_\d+').astype(int)
        filtered_rf['Heads'] = filtered_rf['FileName'].str.extract(r'default_\d+_(\d+)').astype(int)
        sorted_rf = filtered_rf.sort_values(by=['ProteinDomain', 'Layers', 'Heads'])
        final_default = sorted_rf.drop(['FileName'], axis=1)
        new_order = ['ProteinDomain', 'Layers', 'Heads', 'RFScore']
        final_default = final_default.reindex(columns=new_order)

    else:

        return None

    # load sc data for column attention
    sc_list = []

    filepaths = [f'RFscore/{typ}_rf_score_rep{i}.csv' for i in range(1, 6)]

    for path in filepaths:
        sc_rf = pd.read_csv(path)

        sc_rf = sc_rf[~sc_rf['FileName'].str.contains('scovar')]
        sc_rf = sc_rf[sc_rf['FileName'].str.count('_') > 1]
        sc_rf['ProteinDomain'] = sc_rf['FileName'].str.extract(r'(PF\d+)_')
        sc_rf['Layers'] = sc_rf['FileName'].str.extract(r'sc_(\d+)_')[0].astype(int)
        sc_rf['Heads'] = sc_rf['FileName'].str.extract(r'sc_\d+_(\d+)')[0].astype(int)
        sorted_rf = sc_rf.sort_values(by=['ProteinDomain', 'Layers', 'Heads'])
        sorted_rf = sorted_rf.drop(['FileName'], axis=1)
        new_order = ['ProteinDomain', 'Layers', 'Heads', 'RFScore']
        sorted_rf = sorted_rf.reindex(columns=new_order)
        sc_list.append(sorted_rf)

    dataframes = [df for df in sc_list]
    final_sc = reduce(lambda left, right: pd.merge(left, right, on=['ProteinDomain', 'Layers', 'Heads'], how='outer'),
                      dataframes)
    df_reduced = final_sc.drop(columns=['ProteinDomain', 'Layers', 'Heads'])
    reps_means = df_reduced.mean(axis=1)
    reps_variances = df_reduced.var(axis=1)
    final_sc['Mean'] = reps_means
    final_sc['Variance'] = reps_variances

    final_sc = final_sc[['ProteinDomain', 'Mean', 'Variance']]

    # load scovar data
    scovar_list = []
    for path in filepaths:
        scovar_rf = pd.read_csv(path)
        scovar_rf = scovar_rf[scovar_rf['FileName'].str.contains('scovar')]
        scovar_rf = scovar_rf[scovar_rf['FileName'].str.count('_') > 1]
        scovar_rf['ProteinDomain'] = scovar_rf['FileName'].str.extract(r'(PF\d+)_')
        scovar_rf['Layers'] = scovar_rf['FileName'].str.extract(r'scovar_(\d+)_')[0].astype(int)
        scovar_rf['Heads'] = scovar_rf['FileName'].str.extract(r'scovar_\d+_(\d+)')[0].astype(int)
        sorted_rf = scovar_rf.sort_values(by=['ProteinDomain', 'Layers', 'Heads'])
        sorted_rf = sorted_rf.drop(['FileName'], axis=1)
        new_order = ['ProteinDomain', 'Layers', 'Heads', 'RFScore']
        sorted_rf = sorted_rf.reindex(columns=new_order)
        scovar_list.append(sorted_rf)

    dataframes = [df for df in scovar_list]
    final_scovar = reduce(
        lambda left, right: pd.merge(left, right, on=['ProteinDomain', 'Layers', 'Heads'], how='outer'), dataframes)
    df_reduced = final_scovar.drop(columns=['ProteinDomain', 'Layers', 'Heads'])
    reps_means = df_reduced.mean(axis=1)
    reps_variances = df_reduced.var(axis=1)
    final_scovar['Mean'] = reps_means
    final_scovar['Variance'] = reps_variances

    final_scovar = final_scovar[['ProteinDomain', 'Mean', 'Variance']]

    return final_default, final_sc, final_scovar


def extract_data(df, protein_domain):
    data = df[df['ProteinDomain'] == protein_domain]
    return data['RFScore'].to_numpy().reshape(12, 12)


def extract_var(df, protein_domain):
    var = df[df['ProteinDomain'] == protein_domain]
    return var['Variance'].to_numpy().reshape(12, 12)


def plot_heatmap(ax, data, vmax):
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


def plot_protein_domains(default_df, sc_df, scovar_df):
    # fig, axs = plt.subplots()
    fig, axs = plt.subplots(4, 3, figsize=(9, 12), gridspec_kw={"width_ratios": [10, 10, 10]})

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
            elif typ == 'Shuffled_columns':
                heatmap = plot_heatmap(axs[i, j], sc, vmax)
            else:
                heatmap = plot_heatmap(axs[i, j], scovar, vmax)

            if i == 3:
                axs[i, j].set_xlabel("Heads", fontsize=12)
            if i == 0:
                axs[0, j].set_title(typ, fontsize=12)
            if j == 0:
                axs[i, j].set_ylabel(f"{protein_domain}\nLayers", fontsize=12)
            if j == 2:
                fig.colorbar(heatmap, ax=axs[i, j])
            if j != 0:
                overlay_variance(var_of_sc, axs[i, j])
                overlay_variance(var_of_scovar, axs[i, j])

    plt.subplots_adjust(wspace=0.01, hspace=0.35)
    plt.show()


if __name__ == '__main__':
    prot_domains = ['PF00066', 'PF00168', 'PF00484', 'PF00672']
    typs = ['Default', 'Shuffled_columns', 'Shuffled_covariance']
    x_labels = [str(i) for i in range(1, 13)]
    y_labels = [str(i) for i in range(1, 13)]
    cdict = {'red': [(0.0, 1.0, 1.0),
                     (1.0, 1.0, 1.0)],
             'green': [(0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0)],
             'blue': [(0.0, 1.0, 1.0),
                      (1.0, 0.0, 0.0)]}
    red_white_cmap = mcolors.LinearSegmentedColormap('RedWhite', cdict)
    final_default, final_sc, final_scovar = load_data('ml')
    plot_protein_domains(final_default, final_sc, final_scovar, prot_domains)
