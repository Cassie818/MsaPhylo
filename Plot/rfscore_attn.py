from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd


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


def plot_protein_domains(default_df, sc_df, scovar_df, prot_domains):
    fig, axs = plt.subplots(4, 3, figsize=(9, 12),
                            gridspec_kw={"width_ratios": [10, 10, 10]},
                            constrained_layout=True)

    cdict = {'red': [(0.0, 1.0, 1.0),
                     (1.0, 1.0, 1.0)],
             'green': [(0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0)],
             'blue': [(0.0, 1.0, 1.0),
                      (1.0, 0.0, 0.0)]}

    red_white_cmap = mcolors.LinearSegmentedColormap('RedWhite', cdict)

    for i, protein_domain in enumerate(prot_domains):
        default_data = default_df[default_df['ProteinDomain'] == protein_domain]
        sc_data = sc_df[sc_df['ProteinDomain'] == protein_domain]
        scovar_data = scovar_df[scovar_df['ProteinDomain'] == protein_domain]
        default = default_data['RFScore'].to_numpy().reshape(12, 12)
        sc = sc_data['Mean'].to_numpy().reshape(12, 12)
        scovar = scovar_data['Mean'].to_numpy().reshape(12, 12)
        axs[i, 0].imshow(default, cmap=red_white_cmap, aspect='equal', vmin=0, vmax=0.5)
        axs[i, 1].imshow(sc, cmap=red_white_cmap, aspect='equal', vmin=0, vmax=0.5)
        axs[i, 2].imshow(scovar, cmap=red_white_cmap, aspect='equal', vmin=0, vmax=0.5)

    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


if __name__ == '__main__':
    prot_domains = ['PF00066', 'PF00168', 'PF00484', 'PF00672']
    final_default, final_sc, final_scovar = load_data('ml')
    plot_protein_domains(final_default, final_sc, final_scovar, prot_domains)
