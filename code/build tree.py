import torch
import numpy as np
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio import SeqIO
from scipy.spatial.distance import pdist, squareform

EMB_PATH = './Embeddings/Case/'
ATTN_PATH = './Attentions/Case/'
MSA_PATH = './data/Case/RdRp/'
TREE_PATH = './Trees/'

MSA_TYPE_MAP = {
    "default": ".fasta",
    "sc": "_shuffle_column.fasta",
    "sa": "_shuffle_all.fasta",
    "mc": "_mix_column.fasta"
}

EMB_TYPE_MAP = {
    "default": "_emb_",
    "sc": "_emb_shuffle_column_",
    "sa": "_emb_shuffle_all_",
    "mc": "_emb_mix_column_"
}

ATTN_TYPE_MAP = {
    "default": "_attn_",
    "sc": "_attn_shuffle_column_",
    "sa": "_attn_shuffle_all_",
    "mc": "_attn_mix_column_"
}

LAYER = 12
HEAD = 12


class NJtree:

    def __init__(self, protein_family, msa_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "esm_msa1b_t12_100M_UR50S"
        self.protein_family = protein_family
        self.msa_type = msa_type if msa_type in MSA_TYPE_MAP else "default"
        self.msa_fasta_file = msa_type if msa_type in MSA_TYPE_MAP else "default"
        self.emb = f'{EMB_PATH}{self.protein_family}{EMB_TYPE_MAP[self.msa_type]}{self.model_name}.pt'
        self.col_attn = f'{ATTN_PATH}{self.protein_family}{ATTN_TYPE_MAP[self.msa_type]}{self.model_name}.pt'

    @staticmethod
    def euc_distance(a, b):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((a - b) ** 2))

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

    def build_embedding_tree(self):

        # load embeddings
        embeddings = torch.load(self.emb)
        # load the sequence names
        sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]

        for layer in range(LAYER):
            euc_distances = self.pairwise_euclidean_distance(embeddings[layer])
            phylo_path = f"{TREE_PATH}{self.protein_family}_{layer}.nwk"
            # transform the distance matrix into lower-triangle format
            dist = [di[:idx + 1] for idx, di in enumerate(euc_distances.tolist())]
            dm = DistanceMatrix(sequences, dist)
            constructor = DistanceTreeConstructor()
            # Build the NJ trees
            tree = constructor.nj(dm)
            # save the tree
            Phylo.write(tree, phylo_path, "newick")

    def build_attention_tree1(self):
        # load the sequence names
        prot_sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
        # load attention
        attention = torch.load(self.col_attn)
        # remove start token
        attns_mean_on_cols_symm = attention["col_attentions"].cpu().numpy()[0, :, :, 1:, :, :].mean(axis=2)
        attns_mean_on_cols_symm += attns_mean_on_cols_symm.transpose(0, 1, 3, 2)
        phylo_path = '/content/drive/MyDrive/Plmphylo/emb/tree/' + f"{self.pfam}_0_4.nwk"
        attns = attns_mean_on_cols_symm[0, 4, :, :]
        dist = [di[:idx + 1] for idx, di in enumerate(attns.tolist())]
        dm = DistanceMatrix(prot_sequences, dist)
        constructor = DistanceTreeConstructor()
        tree = constructor.nj(dm)  # build the tree
        # save the tree
        Phylo.write(tree, phylo_path, "newick")

    def build_attention_tree2(self):
        # load the sequence names
        prot_sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
        # load attention
        attention = torch.load(self.col_attn)
        # remove start token
        attns_mean_on_cols_symm = attention["col_attentions"].cpu().numpy()[0, :, :, 1:, :, :].mean(axis=2)
        attns_mean_on_cols_symm += attns_mean_on_cols_symm.transpose(0, 1, 3, 2)
        phylo_path = '/content/drive/MyDrive/Plmphylo/emb/tree/' + f"{self.pfam}_11_9.nwk"
        attns = attns_mean_on_cols_symm[11, 9, :, :]
        dist = [di[:idx + 1] for idx, di in enumerate(attns.tolist())]
        dm = DistanceMatrix(prot_sequences, dist)
        constructor = DistanceTreeConstructor()
        tree = constructor.nj(dm)  # build the tree
        # save the tree
        Phylo.write(tree, phylo_path, "newick")
