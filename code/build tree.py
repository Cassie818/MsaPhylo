import torch
import numpy as np
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from scipy.spatial.distance import pdist, squareform


class NJtree:

    def __init__(self, protein_family):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "esm_msa1b_t12_100M_UR50S"
        self.protein_family = protein_family
        self.file_location = '/content/drive/MyDrive/Plmphylo/emb/'
        self.msa_fasta_file = '/content/drive/MyDrive/Plmphylo/emb/tf2s.fasta'
        self.plm_emb_file_name = self.file_location + self.protein_family + '_all_layers_' + self.model_name + '.pt'
        self.col_attn = self.file_location + self.pfam + '_attention_' + self.model_name + '.pt'


    def eucli_distance(self,a,b):
        return np.sqrt(sum((b - a)**2))

    def pairwise_euclidean_distance(self,emb):
      emb = np.array(emb)
      m, n, p = emb.shape

      distances = np.zeros((m, m))

      for i in range(m):
        for j in range(i+1, m):
          distance = self.eucli_distance(emb[i].flatten(), emb[j].flatten())
          distances[i, j] = distance
          distances[j, i] = distance

      return distances

    def build_embedding_tree(self):
      # load the sequence names
      prot_sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
      # load embeddings
      all_embeddings  = torch.load(self.plm_emb_file_name)
      for layer in range(2,3):
        phylo_path = '/content/drive/MyDrive/Plmphylo/emb/tree/'+ f"{self.pfam}_{layer}.nwk"
        euc_array = self.pairwise_euclidean_distance(all_embeddings[layer])
        # transform the distance matrix into lower-triangle format
        dist = [di[:idx+1] for idx, di in enumerate(euc_array.tolist())]
        dm = DistanceMatrix(prot_sequences,dist)
        constructor = DistanceTreeConstructor()
        # Build the NJ trees
        tree = constructor.nj(dm)
        # save the tree
        Phylo.write(tree, phylo_path, "newick")

    def build_attention_tree1(self):
      # load the sequence names
      prot_sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
      # load attention
      attention  = torch.load(self.col_attn)
      # remove start token
      attns_mean_on_cols_symm = attention["col_attentions"].cpu().numpy()[0,:,:,1:,:,:].mean(axis=2)
      attns_mean_on_cols_symm += attns_mean_on_cols_symm.transpose(0, 1, 3, 2)
      phylo_path = '/content/drive/MyDrive/Plmphylo/emb/tree/' + f"{self.pfam}_0_4.nwk"
      attns = attns_mean_on_cols_symm[0,4,:,:]
      dist = [di[:idx+1] for idx, di in enumerate(attns.tolist())]
      dm = DistanceMatrix(prot_sequences,dist)
      constructor = DistanceTreeConstructor()
      tree = constructor.nj(dm) # build the tree
      # save the tree
      Phylo.write(tree, phylo_path, "newick")

    def build_attention_tree2(self):
      # load the sequence names
      prot_sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
      # load attention
      attention  = torch.load(self.col_attn)
      # remove start token
      attns_mean_on_cols_symm = attention["col_attentions"].cpu().numpy()[0,:,:,1:,:,:].mean(axis=2)
      attns_mean_on_cols_symm += attns_mean_on_cols_symm.transpose(0, 1, 3, 2)
      phylo_path = '/content/drive/MyDrive/Plmphylo/emb/tree/' + f"{self.pfam}_11_9.nwk"
      attns = attns_mean_on_cols_symm[11,9,:,:]
      dist = [di[:idx+1] for idx, di in enumerate(attns.tolist())]
      dm = DistanceMatrix(prot_sequences,dist)
      constructor = DistanceTreeConstructor()
      tree = constructor.nj(dm) # build the tree
      # save the tree
      Phylo.write(tree, phylo_path, "newick")
