import argparse


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='Building the phylogenetic trees using the MSA Transformer.')
    parser.add_argument('--input', required=True, help='Input FASTA file path')
    parser.add_argument('--output', required=True, help='Output path to save the phylogenetic trees')
    parser.add_argument('--layer', required=False, help='Specify the layer of the MSA Transformer')


if __name__ == '__main__':
    main()
