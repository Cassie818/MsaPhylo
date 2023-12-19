import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from Bio import SeqIO

MSA_PATH = './data/Case/RdRp/'
EMB_PATH = './Embeddings/'


def euclidean_distance(a, b):
    return np.sqrt(sum((b - a) ** 2))


def pairwise_euclidean_distance(emb):
    emb = np.array(emb)
    m, n, p = emb.shape

    distances = np.zeros((m, m))

    for i in range(m):
        for j in range(i + 1, m):
            distance = euclidean_distance(emb[i].flatten(), emb[j].flatten())
            distances[i, j] = distance
            distances[j, i] = distance

    return distances


class NJtree:

    def __init__(self, protein_family):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "esm_msa1b_t12_100M_UR50S"
        self.protein_family = protein_family
        self.file_location = '/content/drive/MyDrive/Plmphylo/emb/'
        self.msa_fasta_file = f'{MSA_PATH}{protein_family}.fasta'
        self.plm_emb_file_name = f'{EMB_PATH}{self.protein_family}{EMB_TYPE_MAP[self.msa_type]}{self.model_name}.pt'

    def build_embedding_tree(self):
        # load the sequence names
        prot_sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
        excel_file_path = '/content/drive/MyDrive/Plmphylo/emb/label2.xlsx'
        labels = pd.read_excel(excel_file_path, header=None).iloc[:, 0].to_list()
        print(len(labels))
        # load embeddings
        all_embeddings = torch.load(self.plm_emb_file_name)
        emb = all_embeddings[11]
        emb = emb.reshape(emb.shape[0], -1)
        umap_model = umap.UMAP(min_dist=0.1,
                               n_components=3,
                               metric='euclidean',
                               init='spectral',
                               random_state=42)
        umap_embedding = umap_model.fit_transform(emb)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        unique_labels = set(labels)
        print(unique_labels)
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta']
        label_colors = {label: color for label, color in zip(unique_labels, colors)}
        print(label_colors)

        for i in range(umap_embedding.shape[0]):
            label = labels[i]
            ax.scatter(
                umap_embedding[i, 0],
                umap_embedding[i, 1],
                umap_embedding[i, 2],
                c=label_colors.get(label)
            )

        plt.title('3D UMAP Visualization')
        plt.legend()
        plt.show()

        emb = emb.reshape(emb.shape[0], -1)
        umap_model = umap.UMAP(n_neighbors=15,
                               min_dist=0.1,
                               n_components=2,
                               metric='euclidean',
                               init='spectral',
                               random_state=42)
        umap_embedding = umap_model.fit_transform(emb)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        unique_labels = set(labels)
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta']
        label_colors = {label: color for label, color in zip(unique_labels, colors)}

        for i in range(umap_embedding.shape[0]):
            label = labels[i]
            ax.scatter(
                umap_embedding[i, 0],
                umap_embedding[i, 1],
                c=label_colors.get(label)
            )

        plt.title('2D UMAP Visualization')
        plt.legend()
        plt.show()
