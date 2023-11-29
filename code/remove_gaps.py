from Bio import AlignIO
import os


def remove_large_gaps(alignment, threshold=0.2):
    """Remove columns with gaps > threshold."""
    num_sequences = len(alignment)
    keep_columns = []

    for col_num in range(alignment.get_alignment_length()):
        col = alignment[:, col_num]
        gap_count = col.count('-')
        if gap_count / num_sequences <= threshold:
            keep_columns.append(col_num)

    trimmed_alignment = alignment[:, keep_columns[0]:keep_columns[0] + 1]

    for col_num in keep_columns[1:]:
        trimmed_alignment += alignment[:, col_num:col_num + 1]

    return trimmed_alignment


def process_files(input_directory, output_directory, file_extension="fasta"):
    """Process all files in the given directory with the specified file extension."""
    for filename in os.listdir(input_directory):
        if filename.endswith("." + file_extension):
            file_path = os.path.join(input_directory, filename)
            alignment = AlignIO.read(file_path, file_extension)
            trimmed = remove_large_gaps(alignment)

            output_file_path = os.path.join(output_directory, os.path.splitext(filename)[0] + "_trimmed.fasta")
            AlignIO.write(trimmed, output_file_path, file_extension)


# Set your input and output directories here
input_directory = "./data/PFAM"
output_directory = "./data/PFAM/trimmed"

# Process the files
process_files(input_directory, output_directory)
