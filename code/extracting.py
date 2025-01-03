import torch
import string
from typing import List, Tuple
from esm import pretrained
from Bio import SeqIO
from code.params import MSA_PATH, EMB_PATH, ATTN_PATH, MSA_TYPE_MAP, EMB_TYPE_MAP, ATTN_TYPE_MAP


class Extractor:
    """Class for extracting embeddings and column attention heads."""
    def __init__(
            self,
            prot_family: str,
            msa_typ: str
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "esm_msa1b_t12_100M_UR50S"
        self.encoding_dim, self.encoding_layer, self.max_seq_len, self.max_seq_depth = 768, 12, 1024, 1024
        self.prot_family = prot_family
        self.msa_type = msa_typ if msa_typ in MSA_TYPE_MAP else "default"
        self.msa_fasta_file = f'{MSA_PATH}{prot_family}{MSA_TYPE_MAP[self.msa_type]}'

    @staticmethod
    def remove_insertions(sequence: str) -> str:
        """
        Removes any insertions into the sequence. Needed to load aligned sequences in an MSA.
        Utilities from https://github.com/facebookresearch/esm
        """
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        translation = str.maketrans(deletekeys)
        return sequence.translate(translation)

    def read_msa(self) -> List[Tuple[str, str]]:
        """ Reads MSA file. """
        return [(record.description, Extractor.remove_insertions(str(record.seq)))
                for record in SeqIO.parse(self.msa_fasta_file, "fasta")]

    def get_embedding(self):
        """ Extracts embeddings """
        model, alphabet = pretrained.load_model_and_alphabet(self.model_name)
        batch_converter = alphabet.get_batch_converter()

        emb = f'{EMB_PATH}{self.prot_family}{EMB_TYPE_MAP[self.msa_type]}{self.model_name}.pt'
        plm_embedding = {}

        model.eval()
        msa_data = [self.read_msa()]
        msa_labels, msa_strs, msa_tokens = batch_converter(msa_data)
        seq_num = len(msa_labels[0])
        seq_len = len(msa_strs[0][0])
        if seq_len > self.max_seq_depth or seq_num > self.max_seq_len:
            raise ValueError("It exceeds the capacity of the MSA transformer!")

        with torch.no_grad():
            for layer in range(self.encoding_layer):
                out = model(msa_tokens, repr_layers=[layer], return_contacts=False)
                token_representations = out["representations"][layer].view(seq_num, -1, self.encoding_dim)
                # Remove the start token
                token_representations = token_representations[:, 1:, :]
                print(f"Finish extracting embeddings from layer {layer}.")
                plm_embedding[layer] = token_representations

        torch.save(plm_embedding, emb)
        print("Embeddings saved in output file:", emb)

    def get_col_attention(self):
        """ Extracts column attention """
        model, alphabet = pretrained.load_model_and_alphabet(self.model_name)
        batch_converter = alphabet.get_batch_converter()

        attn = f'{ATTN_PATH}{self.prot_family}{ATTN_TYPE_MAP[self.msa_type]}{self.model_name}.pt'

        model.eval()
        msa_data = [self.read_msa()]
        msa_labels, msa_strs, msa_tokens = batch_converter(msa_data)

        with torch.no_grad():
            results = model(msa_tokens, repr_layers=[12], need_head_weights=True)

        torch.save(results, attn)
        print("Column attention saved in output file:", attn)


if __name__ == '__main__':
    msa_type_list = ['default', 'sc', 'scovar', 'sr']
    with open('./data/Pfam/protein_domain.txt', 'r') as file:
        lines = file.readlines()
    protein_domain_list = [line.strip() for line in lines]

    for protein_family in protein_domain_list:
        for msa_type in msa_type_list:
            ex = Extractor(protein_family, msa_type)
            ex.get_embedding()
            ex.get_col_attention()
