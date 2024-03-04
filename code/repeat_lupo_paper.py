"""Adapted from https://github.com/Bitbol-Lab/Phylogeny-MSA-Transformer"""
from params import MSA_PATH, ATTN_PATH, MSA_TYPE_MAP, ATTN_TYPE_MAP
import torch
import numpy as np
from esm import pretrained
from Bio import SeqIO
import string
from typing import List, Tuple
import warnings
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from patsy import dmatrices

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
import statsmodels.api as sm

SEED = 100
rng = np.random.default_rng(seed=SEED)


class Extractor:
    """Class for extracting embeddings and column attention heads."""

    def __init__(
            self,
            prot_family: str,
            msa_typ: str, ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "esm_msa1b_t12_100M_UR50S"
        self.encoding_dim, self.encoding_layer, self.max_seq_len, self.max_seq_depth = 768, 12, 1024, 1024
        self.prot_family = prot_family
        self.msa_type = msa_typ if msa_typ in MSA_TYPE_MAP else "default"
        self.msa_fasta_file = f'{MSA_PATH}{prot_family}{MSA_TYPE_MAP[self.msa_type]}'
        self.attn = f'{ATTN_PATH}{self.prot_family}{ATTN_TYPE_MAP[self.msa_type]}{self.model_name}.pt'

    @staticmethod
    def remove_insertions(sequence: str) -> str:
        """
        Removes any insertions into the sequence. Needed to load aligned sequences in an MSA.
        Utilities from https://github.com/facebookresearch/esm
        """
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        translation = str.maketrans(deletekeys)
        return sequence.translate(translation)

    def read_msa(self) -> List[Tuple[str, str]]:
        """ Reads MSA file. """
        return [(record.description, Extractor.remove_insertions(str(record.seq)))
                for record in SeqIO.parse(self.msa_fasta_file, "fasta")]

    def get_dist(self):
        model, alphabet = pretrained.load_model_and_alphabet(self.model_name)
        batch_converter = alphabet.get_batch_converter()

        attn = f'{ATTN_PATH}{self.prot_family}{ATTN_TYPE_MAP[self.msa_type]}{self.model_name}.pt'

        model.eval()
        msa_data = [self.read_msa()]
        msa_labels, msa_strs, msa_tokens = batch_converter(msa_data)

        hamming_dist = squareform(pdist(msa_tokens.cpu().numpy()[0, :, 1:], metric='hamming'))

        # Load column attention
        col_attn = torch.load(self.attn)

        # Remove start token and  then save averaged and symmetrized column attention matrices
        attn_mean_on_cols_symm = col_attn["col_attentions"].cpu().numpy()[0, :, :, 1:, :, :].mean(axis=2)
        attn_mean_on_cols_symm += attn_mean_on_cols_symm.transpose(0, 1, 3, 2)

        return attn_mean_on_cols_symm, hamming_dist


class CorAns:

    def __init__(self, attn, dist):
        self.attn = attn
        self.dist = dist

    def create_train_test_sets(self,
                               normalize_dists=False,
                               train_size=0.7,
                               ensure_same_size=False,
                               zero_attention_diagonal=False):

        """Attentions assumed averaged across column dimensions, i.e. 4D tensors"""
        if zero_attention_diagonal:
            self.attn[:, :, np.arange(self.attn.shape[2]), np.arange(self.attn.shape[2])] = 0
        assert self.attn.shape[2] == self.attn.shape[3]
        if normalize_dists:
            self.dist = self.dist.astype(np.float64)
            self.dist /= np.max(self.dist)
        if ensure_same_size:
            self.dist = self.dist[:self.attn.shape[2], :self.attn.shape[2]]
        assert len(self.dist) == self.attn.shape[2]
        depth = len(self.dist)
        n_layers, n_heads, depth, _ = self.attn.shape

        # Train-test split
        n_train = int(train_size * depth)
        train_idxs = rng.choice(depth, size=n_train, replace=False)
        split_mask = np.zeros(depth, dtype=bool)
        split_mask[train_idxs] = True

        attns_train, attns_test = self.attn[:, :, split_mask, :][:, :, :, split_mask], self.attn[:, :, ~split_mask, :][
                                                                                       :, :, :, ~split_mask]
        dists_train, dists_test = self.dist[split_mask, :][:, split_mask], self.dist[~split_mask, :][:, ~split_mask]

        n_rows_train, n_rows_test = attns_train.shape[-1], attns_test.shape[-1]
        triu_indices_train = np.triu_indices(n_rows_train)
        triu_indices_test = np.triu_indices(n_rows_test)

        attns_train = attns_train[..., triu_indices_train[0], triu_indices_train[1]]
        attns_test = attns_test[..., triu_indices_test[0], triu_indices_test[1]]
        dists_train = dists_train[triu_indices_train]
        dists_test = dists_test[triu_indices_test]

        attns_train = attns_train.transpose(2, 0, 1).reshape(-1, n_layers * n_heads)
        attns_test = attns_test.transpose(2, 0, 1).reshape(-1, n_layers * n_heads)

        return (attns_train, dists_train), (attns_test, dists_test), (n_rows_train, n_rows_test)

    def perform_regressions_msawise(self,
                                    normalize_dists=False,
                                    ensure_same_size=False,
                                    zero_attention_diagonal=False):
        ((attns_train, dists_train), (attns_test, dists_test),
         (n_rows_train, n_rows_test)) = self.create_train_test_sets(
            normalize_dists=normalize_dists,
            ensure_same_size=ensure_same_size,
            zero_attention_diagonal=zero_attention_diagonal)
        df_train = pd.DataFrame(attns_train,
                                columns=[f"lyr{i}_hd{j}" for i in range(12) for j in range(12)])
        df_train["dist"] = dists_train
        df_test = pd.DataFrame(attns_test,
                               columns=[f"lyr{i}_hd{j}" for i in range(12) for j in range(12)])
        df_test["dist"] = dists_test

        # Carve out the training matrices from the training and testing data frame using the regression formula
        formula = "dist ~ " + " + ".join([f'lyr{i}_hd{j}' for i in range(12) for j in range(12)])
        y_train, X_train = dmatrices(formula, df_train, return_type="dataframe")
        y_test, X_test = dmatrices(formula, df_test, return_type="dataframe")

        binom_model = sm.GLM(y_train, X_train, family=sm.families.Binomial(), cov_type="H0")
        binom_model_results = binom_model.fit()

        y_train = y_train["dist"].to_numpy()
        y_test = y_test["dist"].to_numpy()
        y_pred_train = binom_model_results.predict(X_train).to_numpy()
        y_pred_test = binom_model_results.predict(X_test).to_numpy()

        regr_results[protein_family] = {
            "bias": binom_model_results.params[0],
            "coeffs": binom_model_results.params.to_numpy()[-12 * 12:].reshape(12, 12),
            "y_train": y_train,
            "y_pred_train": y_pred_train,
            "y_test": y_test,
            "y_pred_test": y_pred_test,
            "depth": self.dist.shape[0],
            "n_rows_train": n_rows_train,
            "n_rows_test": n_rows_test
        }

        return regr_results


def plot_protein_domain_correlations(protein_domain_list, regr_results_hamming_msawise, cmap=cm.bwr, vpad=10):
    """
    Plots correlation coefficients for protein domains.

    Parameters:
    - protein_domain_list: List of protein family identifiers.
    - regr_results_hamming_msawise: Dictionary containing regression results for each protein family.
    - cmap: Colormap for the plots.
    - vpad: Padding above the title.
    """
    x_vals_coeffs = np.arange(0, 12, 2)
    y_vals_coeffs = np.arange(0, 12, 2)

    fig, axs = plt.subplots(
        figsize=(9, 12),
        nrows=len(protein_domain_list),
        ncols=1,
        gridspec_kw={"width_ratios": [10]},
        constrained_layout=True
    )

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['arial']

    for i, pfam_family in enumerate(protein_domain_list):
        res = regr_results_hamming_msawise[pfam_family]
        for key in res:
            exec(f"{key} = res['{key}']")

        im = axs[i].imshow(coeffs, norm=colors.CenteredNorm(), cmap=cmap)
        cbar = fig.colorbar(im, ax=axs[i], fraction=0.05, pad=0.03)
        axs[i].set_ylabel(fr"{pfam_family}" + "\n Layer", fontsize=12)
        axs[i].set_xticks(x_vals_coeffs)
        axs[i].set_yticks(y_vals_coeffs)
        axs[i].set_xticklabels(list(map(str, x_vals_coeffs + 1)))
        axs[i].set_yticklabels(list(map(str, y_vals_coeffs + 1)))

    axs[0].set_title("Default", pad=vpad, fontsize=12)
    axs[-1].set_xlabel("Head", fontsize=12)

    plt.show()


if __name__ == '__main__':
    regr_results = {}
    msa_type_list = ['default']
    protein_domain_list = ['PF13377', 'PF13466', 'PF14317', 'PF20171']

    for protein_family in protein_domain_list:
        for msa_type in msa_type_list:
            ex = Extractor(protein_family, msa_type)
            attns_mean_on_cols_symm, hamming_dist = ex.get_dist()
            co = CorAns(attns_mean_on_cols_symm, hamming_dist)
            regr_results_hamming_msawise = co.perform_regressions_msawise()

    plot_protein_domain_correlations(protein_domain_list, regr_results_hamming_msawise)
