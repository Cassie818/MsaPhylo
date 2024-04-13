import csv
import random
from Bio import SeqIO
from code.params import MSA_PATH


class ShuffleMsa:

    def __init__(self, protein_domain):
        self.protein_domain = protein_domain
        self.msa_file = f'{MSA_PATH}{self.protein_domain}.fasta'

    @staticmethod
    def _shuffle_list(data_list):
        if len(set(data_list)) == 1:
            return data_list
        random.shuffle(data_list)
        return data_list

    def shuffle_covariance(self):
        output_file = f'{MSA_PATH}{self.protein_domain}_shuffle_covariance.fasta'
        records = list(SeqIO.parse(self.msa_file, "fasta"))
        seq_length = len(records[0].seq)
        shuffled_records = []

        for i in range(seq_length):
            column = ''.join(record.seq[i] for record in records)
            shuffled_column = self._shuffle_list(list(column))
            for j, record in enumerate(records):
                if j >= len(shuffled_records):
                    shuffled_records.append(record)
                shuffled_records[j].seq = shuffled_records[j].seq[:i] + shuffled_column[j] + shuffled_records[j].seq[
                                                                                             i + 1:]

        SeqIO.write(shuffled_records, output_file, "fasta")
        print(f'Generated shuffle covariance data of {self.protein_domain}!')

    def _read_fasta(self):
        with open(self.msa_file, 'r') as f:
            content = f.readlines()

        sequences = []
        current_sequence = ''
        for line in content:
            line = line.rstrip()
            if line.startswith('>'):
                if current_sequence:
                    sequences.append(current_sequence)
                sequences.append(line)
                current_sequence = ''
            else:
                current_sequence += line
        sequences.append(current_sequence)

        return sequences

    def shuffle_columns(self):
        output_seq_file = f'{MSA_PATH}{self.protein_domain}_shuffle_columns.fasta'
        output_order_file = f'{MSA_PATH}{self.protein_domain}_shuffle_columns_order.txt'
        sequences = self._read_fasta()
        sequence_length = len(sequences[1])

        shuffled_order = [random.sample(range(sequence_length), sequence_length)]
        shuffled_sequences = []

        for sequence in sequences:
            if not sequence.startswith('>'):
                shuffled_sequence = ''.join(sequence[i] for i in shuffled_order[0])
                shuffled_sequences.append(shuffled_sequence)
            else:
                shuffled_sequences.append(sequence)

        with open(output_seq_file, 'w') as f:
            f.write('\n'.join(shuffled_sequences))

        with open(output_order_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(shuffled_order)

        print(f'Generated Shuffle columns data of {self.protein_domain}!')

    def shuffle_rows(self):
        output_seq_file = f'{MSA_PATH}{self.protein_domain}_shuffle_rows.fasta'
        output_order_file = f'{MSA_PATH}{self.protein_domain}_shuffle_rows_order.txt'
        sequences = self._read_fasta()
        sequence_length = len(sequences[1])

        shuffled_sequences = []
        shuffled_order_list = []

        for sequence in sequences:
            if not sequence.startswith('>'):
                shuffled_order = random.sample(range(sequence_length), sequence_length)
                shuffled_order_list.append(shuffled_order)
                shuffled_sequence = ''.join(sequence[i] for i in shuffled_order)
                shuffled_sequences.append(shuffled_sequence)

            else:
                shuffled_sequences.append(sequence)

        with open(output_seq_file, 'w') as file:
            file.write('\n'.join(shuffled_sequences))

        with open(output_order_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(shuffled_order_list)

        print(f'Generated Shuffle rows data of {self.protein_domain}!')

    def keep_fasta_column(self):
        with open(self.msa_file, 'r') as f:
            content = f.readlines()

        sequences = []
        current_sequence = ''
        for line in content:
            line = line.rstrip()
            if line.startswith('>'):
                if current_sequence:
                    sequences.append(current_sequence)
                sequences.append(line)
                current_sequence = ''
            else:
                current_sequence += line
        sequences.append(current_sequence)

        sequence_length = len(sequences[1])

        all_indices = list(range(sequence_length))

        for pos in range(sequence_length):
            keep_column = f'{MSA_PATH}{self.protein_domain}_keep_pos{pos}.fasta'
            remove_sequences = []

            for sequence in sequences:
                if sequence.startswith('>'):
                    remove_sequences.append(sequence)
                else:
                    remove_sequence = ''.join([sequence[pos]])
                    remove_sequences.append(remove_sequence)

            with open(keep_column, 'w') as f:
                f.write('\n'.join(remove_sequences))


if __name__ == '__main__':
    msa_type_list = ['default', 'sc', 'scovar', 'sr']
    with open('./data/Pfam/protein_domain.txt', 'r') as file:
        lines = file.readlines()
    protein_domain_list = [line.strip() for line in lines]

    for protein_domain in protein_domain_list:
        shuffled = ShuffleMsa(protein_domain)
        shuffled.shuffle_columns()
        shuffled.shuffle_covariance()
        shuffled.shuffle_rows()
