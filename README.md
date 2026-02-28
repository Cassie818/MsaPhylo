# Learning the Language of Phylogeny with MSA Transformer

## Introduction

1. MSA Transformer takes a multiple sequence alignment as an input to reveal conservation patterns, and is trained
   with Masked Language Modeling objectives to capture epistasis  [1]. Previous research showed that combinations of MSA
   Transformer's column attention heads correlate with the Hamming
   distance between input sequences, suggesting their application in tracing evolutionary lineages of proteins [2]. <br>
2. We further found that its embedding tree can be used for phylogenetic reconstruction, based on the hypothesis that
   MSA Transformer primarily relies on column-wise conservation information to infer phylogeny. We anticipate it to not
   replace but complement classical phylogenetic inference, to recover the evolutionary history
   of protein families.
3. Unlike traditional phylogenetic trees, embedding tree is assumed to capture epistasis effects and is more sensitive to gaps.

<img src="https://github.com/Cassie818/MsaPhylo/blob/main/Figures/fig1.png" width=800> <br>

## Install packages
Install MsaPhylo
```
git clone https://github.com/Cassie818/MsaPhylo.git
cd MsaPhylo
```

Install pytorch: https://pytorch.org/

```
pip install fair-esm --quiet
pip install transformers --quiet
pip install pysam --quiet
pip install Bio
pip install ete3
```

## Usages

Check out guidance information ```python MsaPhylo.py -h ```

```bash
python MsaPhylo.py
    --i <The_MSA_FILE> \
    --name <NAME_OF_OUTPUT_FILE> \
    --o <OUTPUT_DIRECTORY> \
    --l <LAYER_OF_THE_MSA_TRANSFORMER>
```

Examples:

``` bash
python MsaPhylo.py
        --i "./data/Pfam/PF00066.fasta" \
        --name 'PF00066' \
        --o "/results/trees/" \
        --l 2
```

## Instructions

<ol>

<li> INPUT MSA FILE

```
>Seq1
-SVNINELDLDLIRPGMKLIIIGRPGSGKSVIIKSLIASKRYIPAAIVISGSEEANHFYKTIFPSCFIYNKFNISIIEKI
HKRQITAKNILGTSWLLLIIDDCMDDSKLFCEKTVMDLFKNGRHWNILVVVASQYVMDLKPVIRATIDGVFLLREPNMTY
KEKMWLNFASIIP-KKEFFILMEKITQDHTALYIDNTIINAHWSDCVKYYKASLNIDELFGCEEYKAYCV----
>Seq2
-SIEIKELDLNYVRPGMKIIVIGRPGSGKSTLIKSLIASKRHIPAAVVISGSEEANHFYKNLFPECFVYNKFNLSLIDRI
HKRQITAKNLLDMSWLLLIIDDCMDDSKLFCDKMVMDLFKNGRHWNILVIVASQYVMDLKPVIRSTLDGVFLLREPNMSY
KEKMWLNFASIIP-KKYFFDLMEEITQDHTALYIDNTAINSHWSDCVKYYKATINVDEPFGCEEYKSYII----
>Seq3
----------TELRPGMKLIVLGKPQRGKSVLIKSIIAAKRHIPAAVVISGSEEANHFYSKLLPNCFVYNKFDADIITRV
KQRQLALKNVDPHSWLMLIFDDCMDNAKMFNHEAVMDLFKNGRHWNVLVIIASQYIMDLNASLRCCIDGIFLFTETSQTC
VDKIYKQFGGNIP-KQTFHTLMEKVTQDHTCLYIDNTTTRQKWEDMVRYYKAPLDADVGFGFKDY---------
>Seq4
----------MSSLPDKSTVLFGESGTGKSTIIDDILFQIKPVGQIIVFCPTDRNNKAYSGRVPLPCIHDKITDEVLRDI
WSRQSALTQVYKNPRLVIIFDDCSSQLNLKKNKVIQDIFYQGRHVFITTLIAIQTDKVLDPEIKKNAFVSIFTEETCASS
------YFERKSNDLDKEAKNRARNASKHQKLAWVRDEKR------FYKLMATKHDDFRFGNPIIWNYCEQIQ-
```

</li>

<li> Theoretically, it can handle up to 1,024 protein sequences with an average alignment length of 1,024, but the actual capacity depends on memory requirements.</li>
<li> To construct the embedding tree, you can specify any layer from 1 to 12. It is recommended to use early layers, ranging from 2 to 5.</li>
<li> The current implementation of the embedding tree lacks support for bootstrapping values; therefore, the original MSA is utilized instead. This functionality is scheduled for enhancement in future updates. </li>
</ol>

## Citation

If you are using the MSA Transformer for phylogenetic reconstruction, please consider citing:
```
@article{chen2026learning,
  title={Learning the language of phylogeny with MSA transformer},
  author={Chen, Ruyi and Foley, Gabriel and Bod{\'e}n, Mikael},
  journal={Cell Systems},
  volume={17},
  number={1},
  year={2026},
  publisher={Elsevier}
}
```

## Contact

Feel free to contact me (<a href="ruyi.chen@uq.edu.au">ruyi.chen@uq.edu.au</a>) if you have any questions about
phylogenetic reconstruction🌟

## References

[1] Rao, Roshan M., et al. "MSA transformer." International Conference on Machine Learning. PMLR, 2021. <br>
[2] Lupo, Umberto, Damiano Sgarbossa, and Anne-Florence Bitbol. "Protein language models trained on multiple sequence
alignments learn phylogenetic relationships." Nature Communications 13.1 (2022): 6298.

