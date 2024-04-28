# Deciphering the Black Box: Mastering the MSA Transformer for Phylogenetic Tree Reconstruction

## Introduction

![](https://github.com/Cassie818/MsaPhylo/blob/main/Figures/fig1.png)

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

```
python MsaPhylo.py
    -i <The MSA FILE> \
    -name <NAME OF OUTPUT FILE> \
    -o <OUTPUT DIRECTORY> \
    -l <LAYER>
```

Examples:

```
python MsaPhylo.py
        -i "/Users/cassie/Desktop/MsaPhylo/data/Pfam/PF00066.fasta" \
        -name 'PF00066' \
        -o "/Users/cassie/Desktop/" \
        -l 2
```

a) INPUT MSA FILE

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

(1) The maximum sequence numbers that the MSA Transformer can accepted is 1024. However, according different
computational
environments, the numbers varies.

## Citation

If you are using the MSA Transformer for Phylogenetic Reconstruction, please cite our paper: