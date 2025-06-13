import torch
import esm

import numpy as np
import pandas as pd
import re

from typing import List, Tuple
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import pearsonr


def process_msa(filepath: str) -> List[Tuple[str, str]]:
    results = []
    with open(filepath, 'r') as file:
        name = None
        seq_lines = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    full_seq = ''.join(seq_lines).replace('.', '-').upper()
                    results.append((name, full_seq))
                name = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if name is not None:
            full_seq = ''.join(seq_lines).replace('.', '-').upper()
            results.append((name, full_seq))

    return results


def get_mutation_embedding(
        msa_path,
        mutation_sites,  # 1-based aligned index
        device='cuda'
):
    processed_alignment = process_msa(msa_path)
    assert len(processed_alignment) > 1, "Expected MSA with >1 sequence"

    msa_transformer, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    _, _, tokens = batch_converter([processed_alignment])
    tokens = tokens.to(device)

    with torch.no_grad():
        results = msa_transformer(tokens, repr_layers=[12])
        # shape: (1, N, L, 768)
        representations = results["representations"][12][:, :, 1:, :]  # remove start token

    wild_type_repr = representations[0, 0]  # shape: (L, 768)

    embeddings = [wild_type_repr[site - 1].cpu() for site in mutation_sites]

    concatenated = np.concatenate(embeddings, axis=0)  # shape: (768 * N,)

    return concatenated


def pred_epistasis(emb,
                   y,
                   random_state=68):
    """
    predict epistasis from ridge model
    """
    # classify double variants
    threshold = np.std(y)
    true_labels = np.where(np.abs(y) > threshold, 1, 0)

    model = Ridge(alpha=1)

    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(model, emb, y, cv=cv)

    pearson_corr, p1 = pearsonr(y, y_pred)

    print(f"Pearson Correlation: {pearson_corr}, p-value: {p1}")

    threshold2 = np.std(y_pred)
    pred_labels = np.where(np.abs(y_pred) > threshold2, 1, 0)

    return true_labels, pred_labels


if __name__ == '__main__':
    epistasis_file = 'AVGFP_experiment_epistasis.csv'
    df = pd.read_csv(epistasis_file)
    df_sampled = df.sample(n=1000, random_state=38)

    edouble_list = []
    wt_list = []
    y_list = []

    double_mutants = df_sampled[df_sampled['mutant'].str.contains('-')]

    for idx, row in double_mutants.iterrows():
        double = row['mutant']
        mut1, mut2 = double.split('-')

        match1 = re.match(r"([A-Z])(\d+)([A-Z])", mut1)
        wt_aa1, pos1, mut_aa1 = match1.groups()
        pos1 = int(pos1)

        match2 = re.match(r"([A-Z])(\d+)([A-Z])", mut2)
        wt_aa2, pos2, mut_aa2 = match2.groups()
        pos2 = int(pos2)

        path_double = f'./mutated_MSAs/mutated_MSA_{mut1}_{mut2}.fasta'
        path_wt = f'./data/AVGFP.fasta'

        emb12 = get_mutation_embedding(path_double, [pos1, pos2])
        wt = get_mutation_embedding(path_wt, [pos1, pos2])
        fitness12 = row['epistasis_deviation']

        edouble_list.append(emb12)
        wt_list.append(wt)
        y_list.append(fitness12)

    wild = np.stack(wt_list)
    double = np.stack(edouble_list)
    y = np.array(y_list)

    emb = np.concatenate([wild, double], axis=1)
    true_labels, pred_labels = pred_epistasis(emb, y)
