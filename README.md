# Deciphering the Black Box: Mastering the MSA Transformer for Phylogenetic Tree Reconstruction

## Introduction

<img src="https://github.com/Cassie818/MsaPhylo/blob/main/Figures/fig1.png" width=600>


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
    -i <The MSA FILE> \
    -name <NAME OF OUTPUT FILE> \
    -o <OUTPUT DIRECTORY> \
    -l <LAYER OF THE MSA TRANSFORMER>
```

Examples:

``` bash
python MsaPhylo.py
        -i "./data/Pfam/PF00066.fasta" \
        -name 'PF00066' \
        -o "/Users/cassie/Desktop/" \
        -l 2
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
<li> To construct the phylogenetic tree, you can specify any layer from 1 to 12. It is recommended to use 2 or 3 layers for optimal results.</li>
</ol>

## Citation

If you are using the MSA Transformer for phylogenetic reconstruction, please consider citing:

## Contact

Feel free to contact me if you have any questions about phylogenetic reconstruction.

