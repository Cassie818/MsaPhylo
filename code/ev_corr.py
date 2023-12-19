import torch
import numpy as np
import os
from ete3 import Tree
from scipy import stats
from Bio import SeqIO
import csv
from extracting import MSA_PATH, MSA_TYPE_MAP, EMB_PATH, EMB_TYPE_MAP, ATTN_PATH, ATTN_TYPE_MAP

TREE_PATH = './Trees/NJ/'

LAYER = 12
HEAD = 12


class EvDist:

    def __init__(self, protein_domain, msa_typ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "esm_msa1b_t12_100M_UR50S"
        self.protein_family = protein_domain
        self.msa_type = msa_typ if msa_typ in MSA_TYPE_MAP else "default"
        self.msa_fasta_file = f'{MSA_PATH}{protein_domain}{MSA_TYPE_MAP[self.msa_type]}'
        self.emb = f'{EMB_PATH}{self.protein_family}{EMB_TYPE_MAP[self.msa_type]}{self.model_name}.pt'
        self.attn = f'{ATTN_PATH}{self.protein_family}{ATTN_TYPE_MAP[self.msa_type]}{self.model_name}.pt'
        self.tree = os.path.join(TREE_PATH, f"{self.protein_family}.tree")

    @staticmethod
    def euc_distance(a, b):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((a - b) ** 2))

    def evolutionary_distance(self, phylo_tree, seq_labels):

        """
        Calculate the evolutionary distance between sequences based on the phylogenetic tree.
        """

        phylo_tree = Tree(self.tree)
        ev_distances = []

        for ref_seq_name in seq_labels:
            ref_seq_node = phylo_tree & ref_seq_name

            current_seq_distances = []
            for ex_seq_name in seq_labels:
                ex_seq_node = phylo_tree & ex_seq_name
                distance = ref_seq_node.get_distance(ex_seq_node)
                current_seq_distances.append(distance)

            ev_distances.append(current_seq_distances)

        return np.array(ev_distances)

    def pairwise_euclidean_distance(self, emb):
        emb = np.array(emb)
        m, n, p = emb.shape

        distances = np.zeros((m, m))

        for i in range(m):
            for j in range(i + 1, m):
                distance = self.euc_distance(emb[i].flatten(), emb[j].flatten())
                distances[i, j] = distance
                distances[j, i] = distance

        return distances

    def compute_embedding_correlation(self):
        """
        Calculate the correlation between evolutionary distances from the trees and pairwise Euclidean distances of embeddings
        """
        output_file = os.path.join('./Results',
                                   f"{self.protein_family}_{self.msa_type}_ev_and_euclidean_analysis_emb.csv")

        # Load embeddings
        embeddings = torch.load(self.emb)

        # Load sequence names and extract shorter names
        sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
        sequence_list = [seq.split(' ')[0] for seq in sequences]

        # Compute evolutionary distances
        ev_dist = self.evolutionary_distance(Tree(self.tree), sequence_list)
        spear_ev_dist_corr = []

        # Compute Euclidean distances and their correlation with evolutionary distances
        for layer in range(LAYER):
            euc_dist = self.pairwise_euclidean_distance(embeddings[layer])
            spear_corr = stats.spearmanr(ev_dist.flatten(), euc_dist.flatten())
            spear_ev_dist_corr.append([self.protein_family, layer, spear_corr.correlation, spear_corr.pvalue])

        # Save to CSV file
        with open(output_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Protein domain', 'Layer', 'Correlation', 'P value'])
            writer.writerows(spear_ev_dist_corr)

    def compute_attention_correlation(self):
        """
        Calculate the correlation between evolutionary distances from the trees and column attention
        """
        output_file = os.path.join('./Results', f"{self.protein_family}_{self.msa_type}_ev_analysis_col_attention.csv")
        spear_ev_dist_corr = []

        # Load sequence names and extract shorter names
        sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
        sequence_list = [seq.split(' ')[0] for seq in sequences]

        # Compute evolutionary distances
        ev_dist = self.evolutionary_distance(Tree(self.tree), sequence_list)

        # Load column attention
        col_attn = torch.load(self.attn)
        # Remove start token and  then save averaged and symmetrized column attention matrices
        attn_mean_on_cols_symm = col_attn["col_attentions"].cpu().numpy()[0, :, :, 1:, :, :].mean(axis=2)
        attn_mean_on_cols_symm += attn_mean_on_cols_symm.transpose(0, 1, 3, 2)

        for layer in range(LAYER):
            for head in range(HEAD):
                attn = attn_mean_on_cols_symm[layer, head, :, :]
                sp_corr = stats.spearmanr(ev_dist.flatten(), attn.flatten())
                spear_ev_dist_corr.append([self.protein_family, layer, head, sp_corr.correlation, sp_corr.pvalue])

        # Save to CSV file
        with open(output_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Protein domain', 'Layer', 'Head', 'Correlation', 'P-value'])
            writer.writerows(spear_ev_dist_corr)


if __name__ == '__main__':
    msa_type_list = ['default', 'sc', 'scovar']
    with open('./data/Pfam/protein_domain.txt', 'r') as file:
        lines = file.readlines()
    protein_domain_list = [line.strip() for line in lines]

    for protein_domain in protein_domain_list:
        for msa_type in msa_type_list:
            evd = EvDist(protein_domain, msa_type)
            evd.compute_embedding_correlation()
            evd.compute_attention_correlation()
