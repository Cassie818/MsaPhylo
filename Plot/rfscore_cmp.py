import pandas as pd

ml_rf_score_default = pd.read_csv('RFscore/ml_rf_score_default.csv')
filtered_ml_default = ml_rf_score_default[~ml_rf_score_default['FileName'].str.contains('default')]
filtered_ml_default[['Prefix', 'Suffix']] = filtered_ml_default['FileName'].str.split('_', expand=True)
filtered_ml_default['Suffix'] = filtered_ml_default['Suffix'].astype(int)
sorted_filtered_ml_default = filtered_ml_default.sort_values(by=['Prefix', 'Suffix'])
emb_ml_default = sorted_filtered_ml_default.drop(['Prefix', 'Suffix'], axis=1).reset_index(drop=True)
