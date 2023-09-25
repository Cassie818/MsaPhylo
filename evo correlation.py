import torch
import numpy as np
import os

LAYER = 12
HEAD = 12


class EvDist:
    """Class for evolutionary distance processing"""

    def __init__(self, protein_family, msa_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "esm_msa1b_t12_100M_UR50S"
        self.protein_family = protein_family
        self.msa_type = msa_type if msa_type in MSA_TYPE_MAP else "default"
        self.msa_fasta_file = f'{MSA_PATH}{protein_family}{MSA_TYPE_MAP[self.msa_type]}'
        self.emb = f'{EMB_PATH}{self.protein_family}{EMB_TYPE_MAP[self.msa_type]}{self.model_name}.pt'
        self.attn = f'{ATTN_PATH}{self.protein_family}{ATTN_TYPE_MAP[self.msa_type]}{self.model_name}.pt'
        self.tree = os.path.join('/content/drive/MyDrive/PhD/tree', f"{self.protein_family}.tree")

    @staticmethod
    def euc_distance(a, b):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((a - b) ** 2))

    def evolutionary_distance(self, phylo_tree, seq_labels):
        """Calculate the evolutionary distance between sequences based on the phylogenetic tree."""
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
        output_file = os.path.join('./Results', f"{self.protein_family}_ev_and_euclidean_analysis.csv")

        # Load embeddings
        embeddings = torch.load(self.emb)

        # Load sequence names and extract shorter names
        sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
        sequence_list = [seq.split(' ')[0] for seq in sequences]

        # Compute evolutionary distances
        ev_distances = self.evolutionary_distance(Tree(self.tree), sequence_list)
        correlations = []

        # Compute Euclidean distances and their correlation with evolutionary distances
        for layer in range(LAYER):
            euc_distances = self.pairwise_euclidean_distance(embeddings[layer].mean(1))
            spear_corr = stats.spearmanr(ev_distances.flatten(), euc_distances.flatten())
            correlations.append([self.protein_family, layer, spear_corr.correlation, spear_corr.pvalue])

        # Save to CSV file
        with open(output_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Protein family', 'Layer', 'Correlation', 'P value'])
            writer.writerows(correlations)

    def compute_attention_correlation(self):
        """
        Calculate the correlation between evolutionary distances from the trees and column attention
        """
        output_file = os.path.join('./Results', f"{self.protein_family}_ev_and_euclidean_analysis.csv")
        spear_ev_dist_corr = []

        # Load sequence names and extract shorter names
        sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
        sequence_list = [seq.split(' ')[0] for seq in sequences]
        # Compute evolutionary distances
        ev_distances = self.evolutionary_distance(Tree(self.tree), sequence_list)

        # Load column attention
        attn = torch.load(self.attn)
        # Remove start token
        attn_mean_on_cols_symm = attn["col_attentions"].cpu().numpy()[0, :, :, 1:, :, :].mean(axis=2)
        attn_mean_on_cols_symm += attn_mean_on_cols_symm.transpose(0, 1, 3, 2)
        # Generate the row and column indices of the upper triangular part of the attention matrix
        tri_indices = np.triu_indices(attn_mean_on_cols_symm.shape[-1])
        # Select the upper triangle of attention and distance matrix
        attn = attn_mean_on_cols_symm[..., tri_indices[0], tri_indices[1]]  # (12,12, M * (M+1) / 2)
        ev = ev_distances[tri_indices]
        # Reshape the attention matrix
        attn = attn.transpose(2, 0, 1).reshape(-1, 12 * 12)
        df_attn = pd.DataFrame(attn, columns=[f"lyr{i}_hd{j}" for i in range(12) for j in range(12)])

        for layer in range(LAYER):
            for head in range(HEAD):
                attns = df_attn[f"lyr{layer}_hd{head}"].values
                sp_corr = stats.spearmanr(ev, attns)
                spear_ev_dist_corr.append([self.pfam, layer, head, sp_corr.correlation, sp_corr.pvalue])
        # field names
        fields = ['Protein family', 'Layer', 'Head', 'Correlation', 'P-value']
        # save csv file
        with open(output_file, 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(spear_ev_dist_corr)
