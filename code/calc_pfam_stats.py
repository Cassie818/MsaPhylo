import os
import pandas as pd
from Bio import SeqIO
from ete3 import Tree
import glob


def get_alignment_stats(alignments_file):
    sequences = list(SeqIO.parse(alignments_file, "fasta"))
    sequence_lengths = [len(seq.seq) for seq in sequences]
    no_gap_lengths = [len(str(seq.seq).replace("-", "")) for seq in sequences]
    num_sequences = len(sequences)
    alignment_length = sequence_lengths[0] if sequence_lengths else 0
    no_gap_length_avg = sum(no_gap_lengths) / num_sequences if num_sequences else 0
    return num_sequences, alignment_length, no_gap_length_avg


def calculate_total_distance_to_root(node):
    distance = 0
    while node.up:
        distance += node.dist
        node = node.up
    return distance


def get_mean_extant_to_root(tree_file):
    tree = Tree(tree_file)
    tree_heights = [calculate_total_distance_to_root(leaf) for leaf in tree.get_leaves()]
    average_height = sum(tree_heights) / len(tree_heights) if tree_heights else 0
    return average_height


def calc_stats(alignments_folder, trees_folder1, trees_folder2, output_file):
    data = []

    for alignments_file in glob.glob(os.path.join(alignments_folder, "*.fasta")):
        base_name = os.path.splitext(os.path.basename(alignments_file))[0]
        trees_file1 = os.path.join(trees_folder1, base_name + ".tree")
        trees_file2 = os.path.join(trees_folder2, base_name + ".tree")

        if os.path.isfile(trees_file1):
            print("Calculating statistics for:", base_name)

            num_sequences, alignment_length, no_gap_length_avg = get_alignment_stats(alignments_file)
            avg_extant_to_root_nj = get_mean_extant_to_root(trees_file1)
            avg_extant_to_root_ml = get_mean_extant_to_root(trees_file2)

            data.append([base_name, num_sequences,
                         alignment_length, no_gap_length_avg,
                         avg_extant_to_root_nj, avg_extant_to_root_ml])

    df = pd.DataFrame(data, columns=["pfam_id", "sequence_num", "alignment_len", "avg_sequence_len",
                                     "average_extant_to_root_NJ", "average_extant_to_root_ML"])
    df.to_csv(output_file, sep='\t', index=False)


# Get the current working directory
cwd = os.getcwd()
# Set the alignments and trees folders as subdirectories of the current working directory
alignments_folder = os.path.join(cwd, "data/Pfam")
trees_folder1 = os.path.join(cwd, "Trees/Pfam/NJ")
trees_folder2 = os.path.join(cwd, "Trees/Pfam/ML")

# Specify the output file path
output_file = "pfam_summary.tsv"

# Call the function to calculate statistics for all alignment files and write them to the output file
calc_stats(alignments_folder, trees_folder1, trees_folder2, output_file)
print("Statistics have been written to", output_file)
