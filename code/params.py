"""Constants used in this work"""

MSA_PATH = './data/Pfam/'
EMB_PATH = './Embeddings/Pfam/'
ATTN_PATH = './Attentions/Pfam/'
TREE_PATH = './Trees/'

MSA_TYPE_MAP = {
    "default": ".fasta",
    "sc": "_shuffle_columns.fasta",
    "scovar": "_shuffle_covariance.fasta"
}

EMB_TYPE_MAP = {
    "default": "_emb_",
    "sc": "_emb_shuffle_columns_",
    "scovar": "_emb_shuffle_covariance_"
}

ATTN_TYPE_MAP = {
    "default": "_attn_",
    "sc": "_attn_shuffle_columns_",
    "scovar": "_attn_shuffle_covariance_"
}

LAYER = 12
HEAD = 12
