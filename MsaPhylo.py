import argparse
import torch
import string
from typing import List, Tuple
from esm import pretrained
from Bio import SeqIO
from code.build_plm_tree import PlmTree
from code.extracting import Extractor


class EmbeddingTree(PlmTree, Extractor):
    """Class for building trees from embeddings."""

    def __init__(
            self,
            msa: str,
            name: str,
            output_tree_path: str,
            layer: int
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.msa_fasta_file = msa
        self.name = name
        self.output_tree_path = output_tree_path
        self.layer = int(layer) - 1
        self.model_name = "esm_msa1b_t12_100M_UR50S"
        self.encoding_dim, self.encoding_layer, self.max_seq_len, self.max_seq_depth = 768, 12, 1024, 1024

    def read_msa(self) -> List[Tuple[str, str]]:
        """ Reads MSA file. """
        return [(record.description, Extractor.remove_insertions(str(record.seq)))
                for record in SeqIO.parse(self.msa_fasta_file, "fasta")]

    def get_embedding(self):
        """ Extracts embeddings """
        plm_embedding = {}
        model, alphabet = pretrained.load_model_and_alphabet(self.model_name)
        batch_converter = alphabet.get_batch_converter()

        model.eval()
        msa_data = [self.read_msa()]
        msa_labels, msa_strs, msa_tokens = batch_converter(msa_data)
        seq_num = len(msa_labels[0])
        seq_len = len(msa_strs[0][0])
        if seq_len > self.max_seq_depth or seq_num > self.max_seq_len:
            raise ValueError("It exceeds the capacity of the MSA transformer!")

        with torch.no_grad():
            out = model(msa_tokens, repr_layers=[self.layer], return_contacts=False)
            token_representations = out["representations"][self.layer].view(seq_num, -1, self.encoding_dim)
            # Remove the start token
            token_representations = token_representations[:, 1:, :]
            print(f"Finish extracting embeddings from layer {self.layer + 1}.")
            plm_embedding[self.layer] = token_representations

        return plm_embedding

    def build_emb_tree(self):
        # Load the sequence names
        prot_sequences = [record.id for record in SeqIO.parse(self.msa_fasta_file, "fasta")]
        plm_embedding = self.get_embedding()

        euc_dist = PlmTree.pairwise_euclidean_distance(plm_embedding[self.layer])
        phylo_path = f"{self.output_tree_path}{self.name}_{self.layer + 1}.nwk"
        tree = PlmTree.neighbor_joining(euc_dist, prot_sequences)

        # Save the tree
        with open(phylo_path, "w") as file:
            file.write(tree)

        print(f"Finish building embedding trees from layer {self.layer + 1}.")


def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Building the phylogenetic trees using the MSA Transformer.'
    )
    parser.add_argument(
        '--i',
        type=str,
        required=True,
        help='Input FASTA file path'
    )
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Name of output tree'
    )
    parser.add_argument(
        '--o',
        type=str,
        required=True,
        help='Output path to save the phylogenetic trees'
    )
    parser.add_argument(
        '--l',
        type=int,
        required=False,
        default=3,
        choices=range(1, 13),
        help='Specify the layer of the MSA Transformer (1-12)'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    msa_file = args.i
    layer = args.l
    name = args.name
    output_tree_path = args.o
    embtree = EmbeddingTree(msa_file, name, output_tree_path, layer)
    embtree.build_emb_tree()


if __name__ == '__main__':
    main()
