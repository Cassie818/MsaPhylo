import random
import numpy as np
import numpy.random as nrand
from ete3 import Tree


class MSASimulator:
    def __init__(self, num_leaf_nodes=100, len_protein=100, gap_rate=0.05):
        self.num_leaf_nodes = num_leaf_nodes
        self.len_protein = len_protein
        self.gap_rate = gap_rate
        self.tree = None
        self.amino_acids = []
        self.R = None
        self.PI = None
        self.Q = None

    def generate_random_tree(self):
        self.tree = Tree()
        random.seed(0)
        self.tree.populate(self.num_leaf_nodes,
                           names_library=range(self.num_leaf_nodes),
                           random_branches=True,
                           branch_range=(0, 2))
        idx_nodes = self.num_leaf_nodes
        for node in self.tree.traverse('preorder'):
            if node.name == "":
                node.name = str(idx_nodes)
                idx_nodes += 1
        self.tree.write(outfile="random_tree.newick", format=1)

    def load_replacement_matrix(self, file_path):
        with open(file_path, 'r') as file_handle:
            self.amino_acids = [aa for aa in next(file_handle).split()]
            self.R = np.genfromtxt(file_handle, skip_header=1, usecols=range(20))
            self.PI = np.genfromtxt(file_handle, skip_header=1, usecols=(20,))

        self.Q = np.array(self.R) * self.PI
        np.fill_diagonal(self.Q, -np.sum(self.Q - np.diag(np.diag(self.Q)), axis=1))

    def simulate_sequences(self):
        self.tree = Tree("random_tree.newick", format=1)
        self.tree.name = str(len(self.tree))
        nrand.seed(0)

        # Initialize the root sequence
        root_seq = nrand.choice(self.amino_acids, size=self.len_protein, replace=True, p=self.PI / np.sum(self.PI))
        self.tree.add_feature('seq', root_seq)

        # Simulate the sequences for each node
        for node in self.tree.traverse('preorder'):
            if node.is_root():
                continue
            anc_node = node.up
            seq = np.copy(anc_node.seq)
            dist = node.dist

            while dist > 0:
                rates = np.diag(self.Q)[np.searchsorted(self.amino_acids, seq)]
                tot_rate = -np.sum(rates)
                wait_time = nrand.exponential(scale=1 / tot_rate)

                if wait_time > dist:
                    break

                idx_prob = rates / tot_rate
                idx = nrand.choice(range(self.len_protein), p=idx_prob)
                aa_type_prob = self.Q[:, np.searchsorted(self.amino_acids, seq[idx])] / -self.Q[
                    np.searchsorted(self.amino_acids, seq[idx]), np.searchsorted(self.amino_acids, seq[idx])]
                aa_type_prob[np.searchsorted(self.amino_acids, seq[idx])] = 0
                aa_mutant = nrand.choice(self.amino_acids, p=aa_type_prob)
                seq[idx] = aa_mutant
                dist -= wait_time
            node.add_feature('seq', seq)

    def introduce_gaps(self, sequence):
        new_sequence = list(sequence)
        num_gaps = nrand.poisson(self.gap_rate * self.len_protein)
        for _ in range(num_gaps):
            gap_position = nrand.randint(self.len_protein)
            if nrand.rand() < 0.5:
                new_sequence.insert(gap_position, '-')
            else:
                new_sequence[gap_position] = '-'
        return ''.join(new_sequence)

    def create_msa_with_gaps(self):
        for node in self.tree.traverse('preorder'):
            with_gaps = self.introduce_gaps(node.seq)
            node.add_feature('seq_with_gaps', with_gaps)

    def save_to_fasta(self, filename="simulated_msa_with_gaps.fasta"):
        with open(filename, 'w') as file_handle:
            for node in self.tree.traverse('preorder'):
                sequence_with_gaps = node.seq_with_gaps
                file_handle.write(f">{node.name}\n{sequence_with_gaps}\n")
        print(f"Simulated MSA with gaps has been saved to '{filename}'.")


simulator = MSASimulator()
simulator.generate_random_tree()
simulator.load_replacement_matrix("lg_LG.PAML.txt")
simulator.simulate_sequences()
simulator.create_msa_with_gaps()
simulator.save_to_fasta()
