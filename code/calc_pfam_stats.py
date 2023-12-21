import os
import pandas as pd
from Bio import SeqIO
from ete3 import Tree

data_dict = {}
current_ac = None


def calc_stats(alignments_folder, trees_folder, output_file):
    # Create an empty DataFrame to store the statistics
    data = []

    # Iterate over alignment files
    for filename in os.listdir(alignments_folder):

        alignments_file = os.path.join(alignments_folder, filename)
        trees_file = os.path.join(trees_folder, filename.split(".")[0] + ".tree")

        if os.path.isfile(alignments_file) and os.path.isfile(trees_file):
            print("Calculating statistics for:", filename)

            # Calculate sequence length and number of sequences
            sequences = SeqIO.parse(alignments_file, "fasta")
            sequence_lengths = []
            no_gap_lengths = []
            num_sequences = 0

            for sequence in sequences:
                sequence_lengths.append(len(sequence.seq))
                no_gap_lengths.append(len(str(sequence.seq).replace(".", "")))
                num_sequences += 1

            # Calculate alignment length
            alignment_length = sequence_lengths[0]  # Length of the first sequence

            no_gap_lengths_average = sum(no_gap_lengths) / num_sequences
            # Calculate average height of the tree
            tree = Tree(trees_file)
            tree_heights = [leaf.dist for leaf in tree.get_leaves()]
            average_height = sum(tree_heights) / len(tree_heights)

            # Add the statistics to the DataFrame
            data.append([filename.split(".")[0], num_sequences, alignment_length,
                         no_gap_lengths_average, average_height])

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=["pfam_id", "sequence_num", "alignment_len", "avg_sequence_len",
                                     "average_tree_height"])

    # Write the DataFrame to a TSV file
    df.to_csv(output_file, sep='\t', index=False)


# Get the current working directory
cwd = os.getcwd()

# Set the alignments and trees folders as subdirectories of the current working directory
alignments_folder = os.path.join(cwd, "alignments")
trees_folder = os.path.join(cwd, "trees")
# pfam_file_path = os.path.join(cwd, "unprocessed_pfam_data/example.pfam")
pfam_file_path = os.path.join(cwd, "unprocessed_pfam_data/Pfam-A.seed")

# Specify the output file path
output_file = "pfam_summary.tsv"

# Call the function to calculate statistics for all alignment files and write them to the output file
calc_stats(alignments_folder, trees_folder, output_file)

print("Statistics have been written to", output_file)
