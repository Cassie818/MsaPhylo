import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from functools import reduce

warnings.filterwarnings('ignore')


def extract_number(s):
    parts = s.split('_')[-1]
    return int(''.join(filter(str.isdigit, parts)))


def extract_protein_family(s):
    return s.split('_')[0]


def create_heatmap(data, ax, x_labels, y_labels, vmin=None, vmax=None):
    heatmap = ax.imshow(data, cmap='bwr', aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(0, len(x_labels), 2))
    ax.set_yticks(np.arange(0, len(y_labels), 2))
    ax.set_xticklabels(x_labels[::2])
    ax.set_yticklabels(y_labels[::2])
    return heatmap


def merge_dfs(typ):
    file_paths = ['score/cmp_emb_score_rep1.csv', 'score/cmp_emb_score_rep2.csv',
                  'score/cmp_emb_score_rep3.csv', 'score/cmp_emb_score_rep4.csv',
                  'score/cmp_emb_score_rep5.csv']
    if typ == 'sc':
        dfs = [pd.read_csv(path)[
                   ~pd.read_csv(path)['FileName2'].str.contains('sr') & ~pd.read_csv(path)['FileName2'].str.contains(
                       'scovar')] for path in file_paths]
    elif typ == 'scovar':
        dfs = [pd.read_csv(path)[pd.read_csv(path)['FileName2'].str.contains('scovar')] for path in file_paths]
    else:
        dfs = [pd.read_csv(path)[pd.read_csv(path)['FileName2'].str.contains('sr')] for path in file_paths]

    merge_func = lambda left, right: pd.merge(left, right, on=['FileName1', 'FileName2'])
    df = reduce(merge_func, dfs)
    new_column_names = ['FileName1', 'FileName2', 'RF_1', 'CI_1', 'RF_2', 'CI_2',
                        'RF_3', 'CI_3', 'RF_4', 'CI_4', 'RF_5', 'CI_5']
    df.columns = new_column_names
    RF_columns = [col for col in df.columns if 'RF' in col]
    CI_columns = [col for col in df.columns if 'CI' in col]
    df["RF_mean"] = df[RF_columns].mean(axis=1)
    df["RF_var"] = df[RF_columns].var(axis=1)
    df["CI_mean"] = df[CI_columns].mean(axis=1)
    df["CI_var"] = df[CI_columns].var(axis=1)
    df['FileName1_num'] = df['FileName1'].apply(extract_number)
    df['FileName2_num'] = df['FileName2'].apply(extract_number)
    df['ProteinFamily'] = df['FileName1'].apply(extract_protein_family)

    # Sort the dataframe by these new numeric columns
    df_sorted = df.sort_values(by=['ProteinFamily', 'FileName1_num', 'FileName2_num'])
    return df_sorted


def overlay_variance(var_data, ax):
    max_variance = np.max(var_data)
    for i in range(var_data.shape[0]):
        for j in range(var_data.shape[1]):
            normalized_var = var_data[i, j] / max_variance
            circle_size = normalized_var * 100
            ax.scatter(j, i, s=circle_size, color='gray', alpha=0.5)


def plot_heatmap(ax, data):
    x_labels = [str(i) for i in range(1, 13)]
    y_labels = [str(i) for i in range(1, 13)]
    im = ax.imshow(data, cmap='bwr', aspect='equal')
    ax.set_xticks(np.arange(0, 12, 2))
    ax.set_yticks(np.arange(0, 12, 2))
    ax.set_xticklabels(x_labels[::2])
    ax.set_yticklabels(y_labels[::2])
    return im


def plot_protein_domains(sc_df, scovar_df, sr_df, protein_domains, metrics):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['arial']
    fig, axs = plt.subplots(4, 3, figsize=(10, 10),
                            gridspec_kw={"width_ratios": [10, 10, 10]})

    def cal_avg_std(df, protein_domain):

        if metrics == 'RF':
            selec = df[['ProteinFamily', 'RF_mean', 'RF_var']]
            selec_avg = selec[selec["ProteinFamily"] == protein_domain]['RF_mean'].to_numpy().reshape(12, 12)
            selec_std = selec[selec["ProteinFamily"] == protein_domain]['RF_var'].to_numpy().reshape(12, 12)

        else:
            selec = df[['ProteinFamily', 'CI_mean', 'CI_var']]
            selec_avg = selec[selec["ProteinFamily"] == protein_domain]['CI_mean'].to_numpy().reshape(12, 12)
            selec_std = selec[selec["ProteinFamily"] == protein_domain]['CI_var'].to_numpy().reshape(12, 12)

        return selec_avg, selec_std

    for i, protein_domain in enumerate(protein_domains):

        sc_avg, sc_var = cal_avg_std(sc_df, protein_domain)
        scovar_avg, scovar_var = cal_avg_std(scovar_df, protein_domain)
        sr_avg, sr_var = cal_avg_std(sr_df, protein_domain)

        for j in list(range(3)):

            if j == 0:
                heatmap = plot_heatmap(axs[i, j], sc_avg)
                overlay_variance(sc_var, axs[i, j])
                axs[i, j].set_ylabel(f"{protein_domain}\nDefault", fontsize=12)
                fig.colorbar(heatmap, ax=axs[i, j])

            elif j == 1:
                heatmap = plot_heatmap(axs[i, j], scovar_avg)
                overlay_variance(scovar_var, axs[i, j])
                fig.colorbar(heatmap, ax=axs[i, j])

            else:
                heatmap = plot_heatmap(axs[i, j], sr_avg)
                overlay_variance(sr_var, axs[i, j])
                fig.colorbar(heatmap, ax=axs[i, j])

        axs[0, 0].set_title('Shuffled Positions', fontsize=12)
        axs[0, 1].set_title('Shuffled Covariance', fontsize=12)
        axs[0, 2].set_title('Shuffled Rows', fontsize=12)

    plt.subplots_adjust(wspace=0.01, hspace=0.15)
    plt.show()


if __name__ == '__main__':
    protein_domains = ['PF00066', 'PF00168', 'PF00484', 'PF00672']
    sc_df = merge_dfs('sc')
    scovar_df = merge_dfs('scovar')
    sr_df = merge_dfs('sr')
    plot_protein_domains(sc_df, scovar_df, sr_df, protein_domains, 'RF')
    plot_protein_domains(sc_df, scovar_df, sr_df, protein_domains, 'CI')
