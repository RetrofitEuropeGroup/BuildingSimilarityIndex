import os
import sys
from pathlib import Path

# add the path of the own directory to the system path so that the modules can be imported
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)

from merge_cityjson import MergeCityJSON
from metrics.cityStats import calculate_metrics
from turning_functions.main import process_to_features

class processing():
    def __init__(self, output_path: str, bag_data_folder: str = None, cityjson_path: str = None):
        # TODO: add documentation about the variables
        if bag_data_folder is None and cityjson_path is None:
            raise ValueError("Either input_folder or cityjson_path should be provided")
        elif bag_data_folder is not None and cityjson_path is not None:
            raise ValueError("It is not possible to use both bag_data_folder and cityjson_path")

        self.bag_data_folder = bag_data_folder
        self.cityjson_path = cityjson_path
        self.validate_output_path(output_path)


    def validate_output_path(self, file_path: str):
        if file_path.endswith('.csv') == False:
            raise ValueError("""The output_path can only be a .csv file""")
        else:
            parent_dir = os.path.dirname(file_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)

            self.output_path = file_path        

    # main functions
    def merge_files(self):
        """ Merges the files in the input folder to a single file"""
        merge_folder = Path(self.bag_data_folder.replace('bag_data', 'bag_data_merged')) # create a new folder, if needed, for the merged files
        merger = MergeCityJSON(self.bag_data_folder, output_folder=merge_folder)
        merger.run()

        self.cityjson_path = merger.file_path

    def create_feature_space(self):
        """ Create a pd dataframe with the feature space. First step in the process is to calculate the 2d / 3d metrics on the merged cityjson which outputs a geo df, this is then used to execute the turning function"""
        # determine the number of processors to use and where the cityStats.py file is located, max function is used to prevent using all processors while at the same time ensuring that at least 1 processor is used
        n_processors = max(os.cpu_count() - 1, 1)

        # run the citystats script in which the metrics are calculated
        feature_space_metrics = calculate_metrics(input=self.cityjson_path, jobs=n_processors)

        # execute the turning function
        feature_space_tf = process_to_features(feature_space_metrics)

        # merge the results in one data frame & return the df
        feature_space_full = feature_space_metrics.merge(feature_space_tf, on='id')
        return feature_space_full

    def run(self):
        # merge if a folder has been provided
        if self.bag_data_folder is not None:
            self.merge_files()

        # calculate the 2d / 3d metrics and turning funciton features
        feature_space_full = self.create_feature_space()
        feature_space_full.to_csv(self.output_path)

if __name__ == '__main__':
    p = processing(output_path="data/feature_space/test_fast.csv", bag_data_folder="data/bag_data")
    p.run()
    
    import pandas as pd
    df = pd.read_csv(p.output_path)
    print(df.head())