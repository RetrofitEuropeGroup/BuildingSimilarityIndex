import argparse
import os
import sys
from math import inf

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List


class Distribution:

    def __init__(self, log_bucket: bool = False):
        self.max = -inf
        self.min = inf
        self.buckets = {}
        self._log_bucket = log_bucket
        self._integer = False
        self.average_parts = [[],[]]

    def get_extrema(self, df, feature_name):
        self.max = max(self.max, df[feature_name].max())
        self.min = min(self.min, df[feature_name].min())

    def _make_buckets(self, df, feature_name, number_of_buckets: int = 100) -> None:
        sample = df[feature_name].sample(10)
        if all(int(val) == val for val in sample):
            self._integer = True
        elif self._log_bucket:
            self.buckets = {round(bucket, 3): 0 for bucket in np.logspace(np.log10(self.min), np.log10(self.max), number_of_buckets)}
        else:
            self.buckets = {round(bucket, 3): 0 for bucket in np.arange(self.min, self.max, (self.max-self.min)/number_of_buckets)}

    def sort_in_buckets(self, df, feature_name, number_of_buckets: int = 100) -> None:
        clean_df = df.dropna(subset=[feature_name], inplace=False)
        self._add_to_average_parts(clean_df, feature_name)
        
        if not self.buckets:
            self._make_buckets(clean_df, feature_name, number_of_buckets)
        
        if self._integer:
            value_counts = clean_df[feature_name].value_counts().to_dict()
            for key in value_counts.keys():
                self.buckets[key] = self.buckets.get(key, 0) + value_counts[key]
        else:
            sorted_buckets = sorted(self.buckets.keys())
            values = clean_df[feature_name].to_numpy()

            indices = np.searchsorted(sorted_buckets, values, side='right') - 1

            for key,count in zip(*np.unique(indices, return_counts=True)):
                self.buckets[sorted_buckets[key]] += count

    def _add_to_average_parts(self, df, feature_name):
        self.average_parts[0].append(df.shape[0])
        self.average_parts[1].append(df.loc[:, feature_name].mean())

    def _calculate_median(self):
        n = sum(self.buckets.values())
        targets = [n // 2] if n % 2 else [n // 2 - 1, n // 2]
        cumulative = 0
        found = []

        for key, value in sorted(self.buckets.items()):
            prev_cumulative = cumulative
            cumulative += value
            for t in targets:
                if prev_cumulative <= t < cumulative:
                    found.append(key)
            if len(found) == len(targets):
                break

        return sum(found) / len(found)

    def _calculate_statistics(self):
        self.average = round(sum([amt * avg for amt, avg in zip(*self.average_parts)]) / sum(self.average_parts[0]), 3)

        self.median = self._calculate_median()

        self.modus = max(self.buckets, key=self.buckets.get)

    def return_statistics(self):
        self._calculate_statistics()
        
        return {
            'buckets': self.buckets,
            'max': self.max,
            'min': self.min,
            'mean': self.average,
            'median': self.median,
            'modus': self.modus,
            'integer': self._integer,
        }


def calculate_distribution(
    input: str,
    feature_name: str,
    visual_config: dict = {},
    chunksize = 1000000,
    verbose = False
):
    if verbose:
        print(f"\nCalculating metric of {feature_name}...")

    distribution = Distribution(log_bucket=visual_config.get('log_x', False))

    with pd.read_csv(input, usecols=[feature_name], chunksize=chunksize) as reader:
        for chunk in reader:
            distribution.get_extrema(chunk, feature_name)

    with pd.read_csv(input, usecols=[feature_name], chunksize=chunksize) as reader:
        for chunk in reader:
            distribution.sort_in_buckets(chunk, feature_name)

    return distribution.return_statistics()


def list_features(
    input: str, 
    columns_to_include: List[str] = None,
    columns_to_exclude: List[str] = None,
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

    return feature_names


def render_histogram(
    X: List[float],
    height: List[float],
    output_dir: str,
    feature_name: str,
    # dpi: int = 300,
    # figsize: Tuple[float, float] = (30, 25, "cm")
    # color, names of axis,
    config: dict = {},
    verbose = False
):
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    if config.get('log_x', False):
        ax.set_xscale('log')
    if config.get('log_y', False):
            ax.set_yscale('log')

    if not config.get('integer', True):
        bin_widths = np.diff(X)
        bin_widths = np.append(bin_widths, config.get('max', X[-1]+0.8) - X[-1])
        ax.bar(X, height, width=bin_widths, align='edge')
    else:
        ax.bar(X, height)

    output_path = os.path.join(output_dir, f"{feature_name}.png")
    plt.savefig(output_path)
    plt.close()

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
        with open('./data/visualisation/visualisation_config.json', 'r') as jfile:
            visual_config = json.load(jfile)

        feature_list = list_features(input=args.input, 
                                     columns_to_include=args.columns_include,
                                     columns_to_exclude=args.columns_exclude)
        for feature in feature_list:
            results = calculate_distribution(args.input, 
                                            feature_name=feature,
                                            visual_config=visual_config.get(feature, {}),
                                            verbose=args.verbose)
            
            if args.verbose:
                print("\n" + "=" * 15 + f"{feature}" + "=" * 15)
                print(f"Maximum: {results['max']}")
                print(f"Minimum: {results['min']}")
                print(f"Mean:    {results['mean']}")
                print(f"Median:  {results['median']}")
                print(f"Modus:   {results['modus']}")
                # print(f"Buckets: {results['buckets']}")

            labels = sorted(list(results['buckets'].keys()))
            height = [results['buckets'].get(key) for key in labels]

            vconf = visual_config.get(feature, {}) | results

            render_histogram(
                X=labels,
                height=height,
                output_dir=args.output_folder,
                feature_name=feature,
                config=vconf,
                verbose=args.verbose
            )

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