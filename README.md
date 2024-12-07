# Learning the Language of Phylogeny with the MSA Transformer

## Introduction

<img src="https://github.com/Cassie818/MsaPhylo/blob/main/Figures/fig1.png" width=800>  
The embedding tree can be used for phylogenetic research based on the hypothesis that the MSA Transformer primarily relies on column-wise conservation information to infer phylogeny


## Install packages

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

## Contact

Feel free to contact me (<a href="ruyi.chen@uq.edu.au">ruyi.chen@uq.edu.au</a>) if you have any questions about
phylogenetic reconstruction.

