from functools import reduce
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(typ):
    base_path = 'RFscore'

    # load default data
    default_rf_file = os.path.join(base_path, f'{typ}_rf_score_default.csv')

    if os.path.exists(default_rf_file):
        default_rf = pd.read_csv(default_rf_file)
        filtered_rf = default_rf[~default_rf['FileName'].str.contains('default')]
        filtered_rf[['ProteinDomain', 'Layer']] = filtered_rf['FileName'].str.split('_', expand=True)
        filtered_rf['Layer'] = filtered_rf['Layer'].astype(int)
        sorted_rf = filtered_rf.sort_values(by=['ProteinDomain', 'Layer'])
        final_default = sorted_rf.drop(['FileName'], axis=1).reset_index(drop=True)
        new_order = ['ProteinDomain', 'Layer', 'RFScore']
        final_default = final_default.reindex(columns=new_order)

    else:
        return None

    # load sc data
    sc_list = []

    filepaths = [f'RFscore/{typ}_rf_score_rep{i}.csv' for i in range(1, 6)]
    for path in filepaths:
        sc_rf = pd.read_csv(path)

        sc_rf = sc_rf[~sc_rf['FileName'].str.contains('scovar')]
        sc_rf = sc_rf[sc_rf['FileName'].str.count('_') == 1]
        sc_rf['ProteinDomain'] = sc_rf['FileName'].str.extract(r'(PF\d+)_sc')
        sc_rf['Layer'] = sc_rf['FileName'].str.extract(r'_sc(\d+)').astype(int)

        # sort and drop auxiliary columns
        sorted_rf = sc_rf.sort_values(by=['ProteinDomain', 'Layer'])
        sorted_rf = sorted_rf.drop(columns=['FileName'])

        # append to list
        sc_list.append(sorted_rf)


    dataframes = [df for df in sc_list]
    final_sc = reduce(lambda left, right: pd.merge(left, right, on=['ProteinDomain', 'Layer'], how='outer'), dataframes)
    df_reduced = final_sc.drop(columns=['ProteinDomain', 'Layer'])
    reps_means = df_reduced.mean(axis=1)
    reps_std = df_reduced.std(axis=1)
    final_sc['Mean'] = reps_means
    final_sc['Std'] = reps_std

    final_sc = final_sc[['ProteinDomain', 'Layer', 'Mean', 'Std']]

    # load scovar data
    scovar_list = []
    for path in filepaths:
        scovar_rf = pd.read_csv(path)

        scovar_rf = scovar_rf[scovar_rf['FileName'].str.contains('scovar')]
        scovar_rf = scovar_rf[scovar_rf['FileName'].str.count('_') == 1]
        scovar_rf['ProteinDomain'] = scovar_rf['FileName'].str.extract(r'(PF\d+)_scovar')
        scovar_rf['Layer'] = scovar_rf['FileName'].str.extract(r'_scovar(\d+)').astype(int)

        # sort and drop auxiliary columns
        sorted_rf = scovar_rf.sort_values(by=['ProteinDomain', 'Layer'])
        sorted_rf = sorted_rf.drop(columns=['FileName'])

        # append to list
        scovar_list.append(sorted_rf)

    dataframes = [df for df in scovar_list]
    final_scovar = reduce(lambda left, right: pd.merge(left, right, on=['ProteinDomain', 'Layer'], how='outer'),
                          dataframes)
    df_reduced = final_scovar.drop(columns=['ProteinDomain', 'Layer'])
    reps_means = df_reduced.mean(axis=1)
    reps_std = df_reduced.var(axis=1)
    final_scovar['Mean'] = reps_means
    final_scovar['Std'] = reps_std

    final_scovar = final_scovar[['ProteinDomain', 'Layer', 'Mean', 'Std']]

    return final_default, final_sc, final_scovar

def plot_protein_domains(default_df, sc_df, scovar_df):
    fig, axs = plt.subplots(4, 5, figsize=(12, 9), sharex=False, sharey=False)
    axs = axs.flatten()  # Flatten the 2D array of axes for easier access

    for i, (protein_domain, group) in enumerate(final_default.groupby('ProteinDomain')):
        group = group.sort_values('Layer')
        ax = axs[i]
        layers = list(range(1, 13))
        sc_data = final_sc[final_sc['ProteinDomain'] == protein_domain]
        scovar_data = final_scovar[final_scovar['ProteinDomain'] == protein_domain]

        ax.plot(layers, group['RFScore'], color='dodgerblue', linestyle='--', label='Default')
        ax.errorbar(layers, sc_data['Mean'], yerr=sc_data['Std'], color='red',
                    linestyle='-.', label='Shuffled columns', elinewidth=1)
        ax.errorbar(layers, scovar_data['Mean'], yerr=scovar_data['Std'], color='darkviolet',
                    linestyle='--', label='Shuffled covariance', elinewidth=1)
        ax.set_title(protein_domain, fontsize=12)
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('RF score', fontsize=10)

        ax.set_xticks(range(1, 13, 2))

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0), fontsize=10)

    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


if __name__ == '__main__':
    final_default, final_sc, final_scovar = load_data('ml')
    plot_protein_domains(final_default, final_sc, final_scovar)
