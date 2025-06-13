from Bio import SeqIO
import re
import os
import copy
import pandas as pd

msa_records = list(SeqIO.parse('./data/AVGFP.fasta', 'fasta'))

mutations = pd.read_csv('epistasis_out/second_order_mutations.csv', header=None)
mutations.columns = ['seq', 'log_fitness', 'num_mut', 'mutant']


def apply_single_mutation_on_seq(seq,
                                 mutation):
    """
    Apply a single mutation (e.g., 'A110G') to a sequence.
    Preserve case (uppercase/lowercase) of the original amino acid.
    """
    match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
    if not match:
        raise ValueError(f"Invalid mutation format: {mutation}")

    original_aa, position, mutated_aa = match.groups()
    position = int(position) - 1  # Convert to 0-based indexing

    seq_list = list(seq)
    ref_aa = seq_list[position]

    # Match regardless of case, but preserve original case in replacement
    if ref_aa.upper() == original_aa:
        seq_list[position] = mutated_aa.lower() if ref_aa.islower() else mutated_aa
    else:
        print(f"Warning: Position {position + 1} has {ref_aa}, not {original_aa}. No mutation applied.")

    return ''.join(seq_list)


def apply_multiple_mutations_on_seq(seq,
                                    mutation_list):
    """
    Apply a second-order mutations to a sequence.
    """
    mutated_seq = seq
    for mut in mutation_list:
        mutated_seq = apply_single_mutation_on_seq(mutated_seq, mut)
    return mutated_seq


def mutate_first_sequence_in_msa(msa_records,
                                 mutation_list):
    """
    Apply mutations only to the first sequence in the MSA (wild-type sequence),
    keep all other sequences unchanged.
    """
    new_records = []
    for i, record in enumerate(msa_records):
        new_record = copy.deepcopy(record)
        if i == 0:
            new_record.seq = apply_multiple_mutations_on_seq(str(record.seq), mutation_list)
        new_records.append(new_record)
    return new_records


output_dir = "mutated_MSAs"
os.makedirs(output_dir, exist_ok=True)

for idx, row in mutations.iterrows():
    mutation_string = row['mutant']

    if pd.isna(mutation_string) or not re.match(r"^[A-Z]\d+[A-Z](-[A-Z]\d+[A-Z])?$", mutation_string):
        print(f"Skipping invalid mutation entry: {mutation_string}")
        continue

    # e.g., ['A110G', 'K158R']
    mutation_list = mutation_string.split('-')

    # Generate MSA with the first mutation
    mutated_msa_1 = mutate_first_sequence_in_msa(msa_records, [mutation_list[0]])
    output_file_1 = os.path.join(output_dir, f"mutated_MSA_{mutation_list[0]}.fasta")
    SeqIO.write(mutated_msa_1, output_file_1, 'fasta')

    # Generate MSA with the second mutation
    if len(mutation_list) > 1:
        mutated_msa_2 = mutate_first_sequence_in_msa(msa_records, [mutation_list[1]])
        output_file_2 = os.path.join(output_dir, f"mutated_MSA_{mutation_list[1]}.fasta")
        SeqIO.write(mutated_msa_2, output_file_2, 'fasta')

    # Generate MSA with both mutations
    mutated_msa_double = mutate_first_sequence_in_msa(msa_records, mutation_list)
    output_file_double = os.path.join(output_dir, f"mutated_MSA_{mutation_string.replace('-', '_')}.fasta")
    SeqIO.write(mutated_msa_double, output_file_double, 'fasta')

    print(f"Generated MSAs for mutation: {mutation_string}")
