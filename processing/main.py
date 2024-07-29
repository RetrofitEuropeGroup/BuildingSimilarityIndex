import os
import sys
from pathlib import Path

# add the path of the own directory to the system path so that the modules can be imported
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)

from merge_cityjson import MergeCityJSON
from metrics.cityStats import calculate_metrics
from turning_functions.main import perform_turning_function

class processing():
    def __init__(self, output_file: str, bag_data_folder: str = None, cityjson_file: str = None):
        """
        Initializes an instance of the class that is used to process the data. Processing the data consists of 
        two steps: merging the files in the input folder to a single file (1) and creating a feature space from the merged file.
        The feature space is a pandas dataframe that contains the 2D and 3D metrics and the turning function features, other
        variables could be added as well. It saves the feature space to a csv file in the output_file and saves it to self.feature_space

        Args:
            output_file (str): The path to the output file where the processed data (feature space) will be saved.
            bag_data_folder (str, optional): The path to the folder containing cityjson with a single building, source is from the BAG. Defaults to None.
            cityjson_file (str, optional): The path to the CityJSON file with a single or multiple buildings. Defaults to None.
        """

        self._bag_data_folder = bag_data_folder
        self._cityjson_file = cityjson_file
        self._set_output_file(output_file)

        if bag_data_folder is None and cityjson_file is None:
            raise ValueError("Either input_folder or cityjson_file should be provided")
        elif bag_data_folder is not None and cityjson_file is not None:
            raise ValueError("It is not possible to use both bag_data_folder and cityjson_file")

        

    def _set_output_file(self, file_path: str):
        if file_path.endswith('.csv') == False:
            raise ValueError("The output_file can only be a .csv file")
        else:
            parent_dir = os.path.dirname(file_path)
            if not os.path.exists(parent_dir) and parent_dir != '': # if the parent directory does not exist, create it
                os.makedirs(parent_dir)

            self.output_file = file_path        

    # main functions
    def _merge_files(self):
        """ Merges the files in the input folder to a single file"""
        merge_folder = Path(self._bag_data_folder.replace('bag_data', 'bag_data_merged')) # create a new folder, if needed, for the merged files
        merger = MergeCityJSON(self._bag_data_folder, output_folder=merge_folder)
        merger.run()

        self._cityjson_file = merger.file_path

    def _create_feature_space(self):
        """ Create a pd dataframe with the feature space. First step in the process is to calculate the 2d / 3d metrics on the merged cityjson which outputs a geo df, this is then used to execute the turning function"""
        # determine the number of processors to use and where the cityStats.py file is located, max function is used to prevent using all processors while at the same time ensuring that at least 1 processor is used
        n_processors = max(os.cpu_count() - 1, 1)

        # run the citystats script in which the metrics are calculated
        feature_space_metrics = calculate_metrics(input=self._cityjson_file, jobs=n_processors)

        # execute the turning function
        feature_space_tf = perform_turning_function(feature_space_metrics)

        # merge the results in one data frame & return the df
        feature_space_merged = feature_space_metrics.merge(feature_space_tf, on='id')
        feature_space_merged.drop(columns=['geometry'], inplace=True) # drop the geometry column as it is a shapely object and cannot be saved to a csv file

        return feature_space_merged

    def run(self):
        # merge if a folder has been provided
        if self._bag_data_folder is not None:
            self._merge_files()

        # calculate the 2d / 3d metrics and turning funciton features
        # TODO: create a single progress bar
        self.feature_space = self._create_feature_space()
        self.feature_space.to_csv(self.output_file)

if __name__ == '__main__':
    p = processing(output_file="data/feature_space/test_fast.csv", bag_data_folder="data/bag_data")
    p.run()
    
    import pandas as pd
    df = pd.read_csv(p.output_file)
    print(df.head())