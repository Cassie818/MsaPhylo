import csv
import random
from Bio import SeqIO

EMB_PATH = './Embeddings/Pfam/'
ATTN_PATH = './Attentions/Pfam/'
MSA_PATH = './data/Pfam/'


class ChangeAA:

    def __init__(self, protein_family):
        self.protein_family = protein_family
        self.msa_file = f'{MSA_PATH}{self.protein_family}_seed_hmmalign_no_inserts.fasta'

    @staticmethod
    def _shuffle_list(data_list):
        if len(set(data_list)) == 1:
            return data_list
        random.shuffle(data_list)
        return data_list

    def mix_fasta_column(self):
        output_file = f'{MSA_PATH}{self.protein_family}_mix_column.fasta'
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
        print('Generated Mix columns data!')

    def _read_fasta(self):
        with open(self.msa_file, 'r') as file:
            lines = file.readlines()

        sequences = []
        current_sequence = ''
        for line in lines:
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

    def shuffle_fasta_all(self):
        output_seq_file = f'{MSA_PATH}{self.protein_family}_shuffle_all.fasta'
        output_order_file = f'{MSA_PATH}{self.protein_family}_shuffle_all_order.txt'
        sequences = self._read_fasta()
        sequence_length = len(sequences[1])

        shuffled_sequences = []
        shuffled_order = []

        for sequence in sequences:
            if not sequence.startswith('>'):
                shuffled_indices = random.sample(range(sequence_length), sequence_length)
                shuffled_order.append(shuffled_indices)
                shuffled_sequence = ''.join(sequence[i] for i in shuffled_indices)
                shuffled_sequences.append(shuffled_sequence)
            else:
                shuffled_sequences.append(sequence)

        with open(output_seq_file, 'w') as file:
            file.write('\n'.join(shuffled_sequences))

        with open(output_order_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(shuffled_order)

        print('Generated Shuffle all data!')

    def shuffle_fasta_column(self):
        output_seq_file = f'{MSA_PATH}{self.protein_family}_shuffle_column.fasta'
        output_order_file = f'{MSA_PATH}{self.protein_family}_shuffle_column_order.txt'
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

        with open(output_seq_file, 'w') as file:
            file.write('\n'.join(shuffled_sequences))

        with open(output_order_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(shuffled_order)

        print('Generated Shuffle columns data!')

    def keep_fasta_column(self):
        with open(self.msa_file, 'r') as file:
            lines = file.readlines()

        sequences = []
        current_sequence = ''
        for line in lines:
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
            keep_column = f'{MSA_PATH}{self.protein_family}_keep_column{pos}.fasta'
            remove_sequences = []

            for sequence in sequences:
                if sequence.startswith('>'):
                    remove_sequences.append(sequence)
                else:
                    remove_sequence = ''.join([sequence[pos]])
                    remove_sequences.append(remove_sequence)

            with open(keep_column, 'w') as file:
                file.write('\n'.join(remove_sequences))
