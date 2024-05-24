import torch
import numpy as np
from Bio import SeqIO
from typing import List
from code.params import MSA_PATH, MSA_TYPE_MAP, EMB_PATH, EMB_TYPE_MAP, ATTN_PATH, ATTN_TYPE_MAP, TREE_PATH, LAYER, HEAD


class PlmTree:
    """Class for building treeS from the MSA Transformer"""

    def __init__(self,
                 prot_family: str,
                 msa_type: str
                 ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "esm_msa1b_t12_100M_UR50S"
        self.prot_family = prot_family
        self.msa_type = msa_type
        self.msa_fasta_file = f'{MSA_PATH}{prot_family}{MSA_TYPE_MAP[self.msa_type]}'
        self.emb = f'{EMB_PATH}{self.prot_family}{EMB_TYPE_MAP[self.msa_type]}{self.model_name}.pt'
        self.col_attn = f'{ATTN_PATH}{self.prot_family}{ATTN_TYPE_MAP[self.msa_type]}{self.model_name}.pt'

    @staticmethod
    def euc_distance(a, b):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((a - b) ** 2))

    @staticmethod
    def neighbor_joining(distmat, names):
        """
        Builds a tree from a distance matrix using the NJ algorithm using the original algorithm published by Saitou and Nei.
        Utilities from https://github.com/esbgkannan/chumby

        Parameters
        ----------
        distmat : np.ndarray, a square, symmetrical distance matrix of size (n, n)
        names : list of str list of size (n) containing names corresponding to the distance matrix

        Returns
        -------
        tree : str a newick-formatted tree
        """

        def join_ndx(D, n):
            # calculate the Q matrix and find the pair to join
            Q = np.zeros((n, n))
            Q += D.sum(1)
            Q += Q.T
            Q *= -1.
            Q += (n - 2.) * D
            np.fill_diagonal(Q, 1.)  # prevent from choosing the diagonal
            return np.unravel_index(Q.argmin(), Q.shape)

        def branch_lengths(D, n, i, j):
            i_to_j = float(D[i, j])
            i_to_u = float((.5 * i_to_j) + ((D[i].sum() - D[j].sum()) / (2. * (n - 2.))))
            if i_to_u < 0.:
                i_to_u = 0.
            j_to_u = i_to_j - i_to_u
            if j_to_u < 0.:
                j_to_u = 0.
            return i_to_u, j_to_u

        def update_distance(D, n1, mask, i, j):
            D1 = np.zeros((n1, n1))
            D1[0, 1:] = 0.5 * (D[i, mask] + D[j, mask] - D[i, j])
            D1[0, 1:][D1[0, 1:] < 0] = 0
            D1[1:, 0] = D1[0, 1:]
            D1[1:, 1:] = D[:, mask][mask]
            return D1

        t = names
        D = distmat.copy()
        np.fill_diagonal(D, 0.)

        while True:
            n = D.shape[0]
            if n == 3:
                break
            ndx1, ndx2 = join_ndx(D, n)
            len1, len2 = branch_lengths(D, n, ndx1, ndx2)
            mask = np.full(n, True, dtype=bool)
            mask[[ndx1, ndx2]] = False
            t = [f"({t[ndx1]}:{len1:.6f},{t[ndx2]}:{len2:.6f})"] + [i for b, i in zip(mask, t) if b]
            D = update_distance(D, n - 1, mask, ndx1, ndx2)

        len1, len2 = branch_lengths(D, n, 1, 2)
        len0 = 0.5 * (D[1, 0] + D[2, 0] - D[1, 2])
        if len0 < 0:
            len0 = 0
        newick = f'({t[1]}:{len1:.6f},{t[0]}:{len0:.6f},{t[2]}:{len2:.6f});'
        return newick

    @staticmethod
    def pairwise_euclidean_distance(emb):
        emb = np.array(emb)
        m, n, p = emb.shape

        distances = np.zeros((m, m))

        for i in range(m):
            for j in range(i + 1, m):
                distance = PlmTree.euc_distance(emb[i].flatten(), emb[j].flatten())
                distances[i, j] = distance
                distances[j, i] = distance

        return distances

    def build_embedding_tree(self):

        # Load the sequence names
        prot_sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
        # Load embeddings
        all_embeddings = torch.load(self.emb)

        for layer in range(LAYER):
            euc_dist = PlmTree.pairwise_euclidean_distance(all_embeddings[layer])
            phylo_path = f"{TREE_PATH}{self.prot_family}_{layer}.nwk"
            tree = PlmTree.neighbor_joining(euc_dist, prot_sequences)

            # Save the tree
            with open(phylo_path, "w") as file:
                file.write(tree)

    def build_attention_tree(self):

        # Load the sequence names
        prot_sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
        # Load column attention
        attention = torch.load(self.col_attn)

        # Remove start token
        attn_mean_on_cols_symm = attention["col_attentions"].cpu().numpy()[0, :, :, 1:, :, :].mean(axis=2)
        # Compute and save symmetrized column attention matrices
        attn_mean_on_cols_symm += attn_mean_on_cols_symm.transpose(0, 1, 3, 2)

        for layer in range(LAYER):
            for head in range(HEAD):
                attn = attn_mean_on_cols_symm[layer, head, :, :]
                phylo_path = f"{TREE_PATH}{self.prot_family}_{layer}_{head}.nwk"
                # Build attention tree for each column attention head
                tree = PlmTree.neighbor_joining(attn, prot_sequences)
                # Save the tree
                with open(phylo_path, "w") as file:
                    file.write(tree)


if __name__ == "__main__":
    msa_type_list = ['default', 'sc', 'scovar', 'sr']
    with open('./data/Pfam/protein_domain.txt', 'r') as file:
        lines = file.readlines()
    protein_domain_list = [line.strip() for line in lines]

    for protein_family in protein_domain_list:
        for msa_type in msa_type_list:
            plmtree = PlmTree(protein_family, msa_type)
            plmtree.build_embedding_tree()
            plmtree.build_attention_tree()
