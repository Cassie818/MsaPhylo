import os
import pandas as pd
from Bio import SeqIO
from ete3 import Tree
import glob


class PhyloStat:
    def __init__(self,
                 msa_folder: str,
                 nj_tree_folder: str,
                 ml_tree_folder: str,
                 output_file: str,
                 ):
        self.msa_folder = msa_folder
        self.nj_tree_folder = nj_tree_folder
        self.ml_tree_folder = ml_tree_folder
        self.output_file = output_file

    @staticmethod
    def get_alignment_stats(alignments_file):
        sequences = list(SeqIO.parse(alignments_file, "fasta"))
        sequence_lengths = [len(seq.seq) for seq in sequences]
        no_gap_lengths = [len(str(seq.seq).replace("-", "")) for seq in sequences]
        num_sequences = len(sequences)
        alignment_length = sequence_lengths[0] if sequence_lengths else 0
        no_gap_length_avg = sum(no_gap_lengths) / num_sequences if num_sequences else 0
        return num_sequences, alignment_length, no_gap_length_avg

    @staticmethod
    def calculate_total_distance_to_root(node):
        distance = 0
        while node.up:
            distance += node.dist
            node = node.up
        return distance

    @staticmethod
    def get_mean_extant_to_root(tree_file):
        """Calculate the extant to root distance for the tree."""
        tree = Tree(tree_file)
        tree_heights = [PhyloStat.calculate_total_distance_to_root(leaf) for leaf in tree.get_leaves()]
        average_height = sum(tree_heights) / len(tree_heights) if tree_heights else 0
        return average_height

    def calc_stats(self):
        data = []

        for alignments_file in glob.glob(os.path.join(self.msa_folder, "*.fasta")):
            base_name = os.path.splitext(os.path.basename(alignments_file))[0]
            nj_tree = os.path.join(self.nj_tree_folder, base_name + ".tree")
            ml_tree = os.path.join(self.ml_tree_folder, base_name + ".tree")

            if os.path.isfile(nj_tree) and os.path.isfile(ml_tree):
                print("Calculating statistics for:", base_name)

                num_sequences, alignment_length, no_gap_length_avg = PhyloStat.get_alignment_stats(alignments_file)
                avg_extant_to_root_nj = PhyloStat.get_mean_extant_to_root(nj_tree)
                avg_extant_to_root_ml = PhyloStat.get_mean_extant_to_root(ml_tree)

                data.append([base_name, num_sequences,
                             alignment_length, no_gap_length_avg,
                             avg_extant_to_root_nj, avg_extant_to_root_ml])

        df = pd.DataFrame(data, columns=["pfam_id", "sequence_num",
                                         "alignment_len", "avg_sequence_len",
                                         "average_extant_to_root_NJ", "average_extant_to_root_ML"])
        df.to_csv(self.output_file, sep='\t', index=False)
        print("Statistics have been written to", self.output_file)


if __name__ == "__main__":
    cwd = os.getcwd()
    alignments_folder = os.path.join(cwd, "data/Pfam")
    nj_tree_folder = os.path.join(cwd, "Trees/Pfam/NJ")
    ml_tree_folder = os.path.join(cwd, "Trees/Pfam/ML")
    output_file = "Pfam_summary.tsv"

    phylo_stats = PhyloStat(alignments_folder, nj_tree_folder, ml_tree_folder, output_file)
    phylo_stats.calc_stats()
