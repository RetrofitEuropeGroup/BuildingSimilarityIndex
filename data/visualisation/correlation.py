import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute distribution visualisation for all columns in a large csv file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='Path to the input CSV file (feature space)')
    parser.add_argument('-o', '--output-path', type=str, default='./output',
                        help='Output path for results')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help='Verbose')

    return parser.parse_args()


def main():
    args = parse_arguments()

    df = pd.read_csv(args.input)
    df.drop(columns=['id','woonfunctie','bijeenkomstfunctie','celfunctie','gezondheidszorgfunctie','industriefunctie','kantoorfunctie','logiesfunctie','onderwijsfunctie','sportfunctie','winkelfunctie'],
            inplace=True)
    corr_matrix = df.corr()
    corr_matrix.to_csv('./output/corrm_new.csv')

    # plt.scatter(df['circularity_2d'].values, df['hemisphericality_3d'].values)
    # plt.savefig('./output/aaaa.png')
    # plt.close()


if __name__ == '__main__':
    main()