import torch
import string
from esm import pretrained
from Bio import SeqIO

EMB_PATH = './Embeddings/'
ATTN_PATH = './Attentions/'
MSA_PATH = './Msa/'
TREE_PATH = './Trees/'

MSA_TYPE_MAP = {
    "default": "_seed_hmmalign_no_inserts.fasta",
    "sc": "_shuffle_column.fasta",
    "sa": "_shuffle_all.fasta",
    "mc": "_mix_column.fasta"
}

EMB_TYPE_MAP = {
    "default": "_emb_no_shuffle_",
    "sc": "_emb_shuffle_column_",
    "sa": "_emb_shuffle_all_",
    "mc": "_emb_mix_column_"
}

ATTN_TYPE_MAP = {
    "default": "_attn_no_shuffle_",
    "sc": "_attn_shuffle_column_",
    "sa": "_attn_shuffle_all_",
    "mc": "_attn_mix_column_"
}


def remove_insertions(sequence):
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)


class Extractor:

    def __init__(self, protein_family, msa_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "esm_msa1b_t12_100M_UR50S"
        self.encoding_dim, self.encoding_layer, self.max_seq_length = 768, 12, 1022
        self.protein_family = protein_family
        self.msa_type = msa_type if msa_type in MSA_TYPE_MAP else "default"
        self.msa_fasta_file = f'{MSA_PATH}{protein_family}{MSA_TYPE_MAP[self.msa_type]}'

    def read_msa(self):
        return [(record.description, remove_insertions(str(record.seq)))
                for record in SeqIO.parse(self.msa_fasta_file, "fasta")]

    def get_embedding(self):
        model, alphabet = pretrained.load_model_and_alphabet(self.model_name)
        batch_converter = alphabet.get_batch_converter()

        emb = f'{EMB_PATH}{self.protein_family}{EMB_TYPE_MAP[self.msa_type]}{self.model_name}.pt'
        plm_embedding = {}

        model.eval()
        msa_data = [self.read_msa()]
        msa_labels, msa_strs, msa_tokens = batch_converter(msa_data)
        seq_num = len(msa_labels[0])

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
        model, alphabet = pretrained.load_model_and_alphabet(self.model_name)
        batch_converter = alphabet.get_batch_converter()

        attn = f'{ATTN_PATH}{self.protein_family}{ATTN_TYPE_MAP[self.msa_type]}{self.model_name}.pt'

        model.eval()
        msa_data = [self.read_msa()]
        msa_labels, msa_strs, msa_tokens = batch_converter(msa_data)

        with torch.no_grad():
            results = model(msa_tokens, repr_layers=[12], need_head_weights=True)

        torch.save(results, attn)
        print("Column attention saved in output file:", attn)


if __name__ == '__main__':
    protein_family_list = ['PF00004']
    msa_type_list = ['no']
    for protein_family in protein_family_list:
        for msa_type in msa_type_list:
            ext = Extractor(protein_family, msa_type)
            ext.get_embedding()
            ext.get_col_attention()