import argparse
import os
import sys
from math import inf

import csv
import matplotlib as plt
import numpy as np
import pandas as pd
from typing import List


class Distribution:

    def __init__(self):
        self.max = -inf
        self.min = inf
        self.buckets = {}
        self._integer = False

    def get_extrema(self, df, feature_name):
        self.max = max(self.max, df[feature_name].max())
        self.min = min(self.min, df[feature_name].min())

    def _make_buckets(self, df, feature_name, number_of_buckets: int = 50) -> None:
        sample = df[feature_name].sample(10)
        if all(int(val) == val for val in sample):
            self._integer = True
        else:
            self.buckets = {round(bucket, 3): 0 for bucket in np.arange(self.min, self.max, (self.max-self.min)/number_of_buckets)}

    def sort_in_buckets(self, df, feature_name, number_of_buckets: int = 50) -> None:
        clean_df = df.dropna(subset=[feature_name], inplace=False)
        
        if not self.buckets:
            self._make_buckets(clean_df, feature_name, number_of_buckets)
        
        if self._integer:
            value_counts = clean_df[feature_name].value_counts().to_dict()
            for key in value_counts.keys():
                self.buckets[key] = self.buckets.get(key, 0) + value_counts[key]
            # for value in clean_df[feature_name]:
            #     self.buckets[int(value)] = self.buckets.get(int(value), 0) + 1
        else:
            sorted_buckets = sorted(self.buckets.keys())
            values = clean_df[feature_name].to_numpy()

            indices = np.searchsorted(sorted_buckets, values, side='right') - 1

            for key,count in zip(*np.unique(indices, return_counts=True)):
                self.buckets[sorted_buckets[key]] += count

    def calculate_statistics(self):
        pass

    def return_statistics(self):
        
        
        
        return (
            self.distribution,
            self.max,
            self.min,
            self.average,
            self.median,
            self.modus
        ) 


def calculate_distribution(
    input: str, 
    columns_to_include: List[str] = None,
    columns_to_exclude: List[str] = None,
    chunksize = 1000000,
    verbose = False
):
    if columns_to_include:
        feature_names = columns_to_include
    else:
        with open(input, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            feature_names = next(csv_reader)
            if columns_to_exclude:
                feature_names = [feature for feature in feature_names if feature not in columns_to_exclude]

    if verbose:
        print("Calculating metrics of features: " + ", ".join(feature_names) + "\n")

    for feature_name in feature_names:
        if verbose:
            print(f"Calculating metric of {feature_name}...")

        distribution = Distribution()

        with pd.read_csv(input, usecols=[feature_name], chunksize=chunksize) as reader:
            for chunk in reader:
                distribution.get_extrema(chunk, feature_name)

        with pd.read_csv(input, usecols=[feature_name], chunksize=chunksize) as reader:
            for chunk in reader:
                distribution.sort_in_buckets(chunk, feature_name)

            # print(distribution.buckets)
            # print(chunk.loc[chunk[feature_name] > max(distribution.buckets.keys()), feature_name])

    return distribution.return_statistics()


def render_histogram(
    X: List[float],
    height: List[float],
    output_path: str,
    feature_name: str,
    # dpi: int = 300,
    # figsize: Tuple[float, float] = (30, 25, "cm")
    # color, names of axis
    verbose = False
):
    fig, ax = plt.bar()

    plt.savefig()
    plt.close(fig)

    if verbose:
        print(f"Visualisation of feature {feature_name} to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute distribution visualisation for all columns in a large csv file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='Path to the input CSV file (feature space)')
    parser.add_argument('-o', '--output-folder', type=str, default='./output',
                        help='Output folder for results')
    parser.add_argument('-ci', '--columns-include', type=str, default=None,
                        help='CSV columns/features to be included. sep=","')
    parser.add_argument('-cx', '--columns-exclude', type=str, default=None,
                        help='CSV columns/features to be excluded. sep=","')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help='Verbose')

    return parser.parse_args()


def validate_input(input_path: str) -> None:
    """Validate that inputs exist and are accessible."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if os.path.getsize(input_path) == 0:
        raise ValueError("Input file is empty")


def main():
    args = parse_arguments()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)
    if args.columns_include:
        args.columns_include = [column for column in args.columns_include.split(',')]
    if args.columns_exclude:
        args.columns_exclude = [column for column in args.columns_exclude.split(',')]
    

    try:
        validate_input(args.input)
        calculate_distribution(args.input, 
                               columns_to_include=args.columns_include,
                               columns_to_exclude=args.columns_exclude,
                               verbose=args.verbose)

        if args.verbose:
            # Print results
            pass

    except FileNotFoundError:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()