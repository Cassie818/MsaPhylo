from functools import reduce


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
    reps_variances = df_reduced.var(axis=1)
    final_sc['Mean'] = reps_means
    final_sc['Variance'] = reps_variances

    final_sc = final_sc[['ProteinDomain', 'Mean', 'Variance']]

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
    reps_variances = df_reduced.var(axis=1)
    final_scovar['Mean'] = reps_means
    final_scovar['Variance'] = reps_variances

    final_scovar = final_scovar[['ProteinDomain', 'Mean', 'Variance']]

    return final_default, final_sc, final_scovar
