import numpy as np
from ete3 import Tree
import numpy.random as nrand


class PhylogeneticSimulator:
    def __init__(self, num_leaf_nodes=100, len_protein=100):
        self.num_leaf_nodes = num_leaf_nodes
        self.len_protein = len_protein
        self.tree = None
        self.LG_matrix = None
        self.amino_acids = ['A', 'R', 'N', 'D', 'C',
                            'Q', 'E', 'G', 'H', 'I',
                            'L', 'K', 'M', 'F', 'P',
                            'S', 'T', 'W', 'Y', 'V']
        self.aa2idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}

    def generate_tree(self):
        self.tree = Tree()
        self.tree.populate(self.num_leaf_nodes,
                           names_library=range(self.num_leaf_nodes),
                           random_branches=True,
                           branch_range=(0, 0.5))
        idx_nodes = self.num_leaf_nodes
        for node in self.tree.traverse('preorder'):
            if node.name == "":
                node.name = str(idx_nodes)
                idx_nodes += 1

    def read_LG_matrix(self, filename):
        R_dict = {}
        Q_dict = {}
        PI_dict = {}
        with open(filename, 'r') as file_handle:
            line_num = 0
            for line in file_handle:
                line = line.strip()
                fields = line.split()
                if line_num < 20:
                    assert (len(fields) == line_num)
                    if len(fields) != 0:
                        for i in range(line_num):
                            R_dict[(self.amino_acids[line_num], self.amino_acids[i])] = fields[i]
                else:
                    if len(fields) != 0:
                        for i in range(len(fields)):
                            PI_dict[self.amino_acids[i]] = fields[i]
                line_num += 1

        R = np.zeros((20, 20))
        PI = np.zeros((20, 1))
        for i in range(len(self.amino_acids)):
            for j in range(len(self.amino_acids)):
                if i > j:
                    R[i, j] = float(R_dict[(self.amino_acids[i], self.amino_acids[j])])
                    R[j, i] = R[i, j]
            PI[i] = PI_dict[self.amino_acids[i]]

        Q = np.zeros((20, 20))
        for i in range(20):
            for j in range(20):
                if i != j:
                    Q[i, j] = PI[j] * R[i, j]
            Q[i, i] = - np.sum(Q[i, :])

        for i in range(20):
            for j in range(20):
                Q_dict[(self.amino_acids[i], self.amino_acids[j])] = Q[i, j]

        self.LG_matrix = {'R': R, 'Q': Q, 'PI': PI,
                          'R_dict': R_dict, 'Q_dict': Q_dict,
                          'PI_dict': PI_dict, 'amino_acids': self.amino_acids}

    def simulate_sequences(self):
        if not self.tree:
            raise ValueError("Tree not generated.")
        if not self.LG_matrix:
            raise ValueError("LG matrix not loaded.")

        Q_dict = self.LG_matrix['Q_dict']
        R_dict = self.LG_matrix['R_dict']
        PI_dict = self.LG_matrix['PI_dict']
        amino_acids = self.LG_matrix['amino_acids']
        aa2idx = self.aa2idx
        len_protein = self.len_protein
        R = self.LG_matrix['R']
        Q = self.LG_matrix['Q']
        PI = self.LG_matrix['PI']
        root_seq = nrand.choice(self.amino_acids,
                                size=self.len_protein,
                                replace=True,
                                p=PI.reshape(-1) / np.sum(PI))
        self.tree.add_feature('seq', root_seq)

        for node in self.tree.traverse('preorder'):
            if node.is_root():
                continue
            anc_node = node.up

            seq = np.copy(anc_node.seq)
            dist = node.dist

            while True:
                tot_rate = -np.sum([Q_dict[(aa, aa)] for aa in seq])
                wait_time = nrand.exponential(scale=1 / tot_rate)

                if wait_time > dist: break

                idx_prob = np.array([-Q_dict[(aa, aa)] for aa in seq]) / tot_rate
                idx = nrand.choice(range(len_protein), p=idx_prob)

                aa_idx = aa2idx[seq[idx]]
                aa_type_prob = Q[aa_idx, :] / (-Q[aa_idx, aa_idx])
                aa_type_prob[aa_idx] = 0

                aa_mutant = nrand.choice(amino_acids, p=aa_type_prob)
                seq[idx] = aa_mutant
                dist -= wait_time
            node.add_feature('seq', seq)

    def save_sequences_to_file(self, filename):
        if not self.tree:
            raise ValueError("Tree not generated or sequences not simulated.")

        with open(filename, 'w') as file_handle:
            for node in self.tree.traverse('preorder'):
                header = ">{}".format(node.name)
                sequence = "".join(node.seq).upper()
                file_handle.write(header + "\n")
                file_handle.write(sequence + "\n")


def main():
    # Create an instance of the PhylogeneticSimulator
    simulator = PhylogeneticSimulator(num_leaf_nodes=100, len_protein=100)
    # Generate a random phylogenetic tree
    simulator.generate_tree()
    print("Phylogenetic tree generated.")
    # Load the LG replacement matrix from a file
    lg_matrix_file = "lg_LG.PAML.txt"
    simulator.read_LG_matrix(lg_matrix_file)
    print("LG matrix loaded.")
    # Simulate sequences on the phylogenetic tree
    simulator.simulate_sequences()
    print("Sequences simulated.")
    # Optionally, you can save the simulated sequences to a file
    simulator.save_sequences_to_file("s1.fasta")
    print("Sequences saved to 's1.fasta'.")


if __name__ == '__main__':
    main()
