# Deciphering the Black Box: Mastering the MSA Transformer for Phylogenetic Tree Reconstruction

## Install packages

Install pytorch: https://pytorch.org/

```python
!pip
install
fair - esm - -quiet
!pip
install
transformers - -quiet
!pip
install
pysam - -quiet
!pip
install
Bio
!pip
install
ete3
!pip
install
dendropy
```

## Usage

```python
python
MsaPhylo.py
- input < INPUT
MSA
FILE >
-name < NAME
OF
OUTPUT
FILE >
-output < PATH
OF
OUTPUT
FILE >
-layer < LAYER >
```

Examples:

```python
python
MsaPhylo.py
- input
"/Users/cassie/Desktop/MsaPhylo/data/Pfam/PF00066.fasta"
- name
'PF00066'
- output
"/Users/cassie/Desktop/"
- layer
2
```

a) INPUT MSA FILE

```
>S1
QPKCP---QEEYCRDRF-SNSVCDEVCMREECEFDGGDCF
>S2
WSRCA---NATFCEASF-QNGKCDELCNTPFCLYDGNDCD
>S3
WHACA---NQ-TCRTVF-ADGVCDPSCNTAACVFDGDDCV
>S4
WSKCH---RPSYCWSRF-SNGKCDEECNNRNCLYDGKDCS
>S5
FDACQ---NASYCASVF-NNGVCDWQCNSYDCNFDGLDCR
>S6
WANCT---NP-MCWRVF-NNSQCDEACNNEDCLYDNFDCR
>S7
WARCA---DP-RCWRVF-NNSQCDESCNNADCLYDNFDCK
>S8
WGRCP---HP-ECWRLF-NNSQCDERCNTAECLYDNFDCK
```

The maximum sequence numbers that the MSA Transformer can accepted is 1024. However, according different computational
environments, the numbers varies.

## Citation

If you are using the MSA Transformer for Phylogenetic Reconstruction, please cite our paper: