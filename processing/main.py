import os
import sys
from pathlib import Path
import pandas as pd

# add the path of the own directory to the system path so that the modules can be imported
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)

from merge_cityjson import MergeCityJSON
from metrics.cityStats import calculate_metrics
from turning_functions.main import perform_turning_function

class processing():
    def __init__(self, feature_space_file: str, bag_data_folder: str = None, categorical_columns: list = None, verbose: bool = False):
        """
        Initializes an instance of the class that is used to process the data. Processing the data consists of 
        two steps: merging the files in the input folder to a single file (1) and creating a feature space from the merged file.
        The feature space is a pandas dataframe that contains the 2D and 3D metrics and the turning function features, other
        variables could be added as well. It saves the feature space to a csv file in the feature_space_file and saves it to self.feature_space

        Args:
            feature_space_file (str): The path to the output file where the processed data (=feature space) will be saved.
            bag_data_folder (str, optional): The path to the folder containing multiple cityjson files with a single building, source is from the BAG. Defaults to None.
            categorical_columns (list, optional): A list of column names that are categorical, these columns will get distance 1/num_categories if they are not equal and 0 if they are. Defaults to None.
            verbose (bool, optional): If True, the progress will be printed. Defaults to False.
        """

        self._bag_data_folder = bag_data_folder
        self._categorical_columns = categorical_columns
        self._set_feature_space_file(feature_space_file)
        self._verbose = verbose

    def _set_feature_space_file(self, file_path: str):
        if file_path.endswith('.csv') == False:
            raise ValueError("The feature_space_file can only be a .csv file")
        else:
            parent_dir = os.path.dirname(file_path)
            if not os.path.exists(parent_dir) and parent_dir != '': # if the parent directory does not exist, create it
                os.makedirs(parent_dir)

            self.feature_space_file = file_path        

    # main functions
    def _merge_files(self):
        """ Merges the files in the input folder to a single file"""
        merge_folder = Path(self._bag_data_folder + '_merged') # create a new folder, if needed, for the merged files
        merger = MergeCityJSON(self._bag_data_folder, output_folder=merge_folder)
        merger.run()

        self._merged_file = merger.file_path

    def _create_feature_space(self):
        """ Create a pd dataframe with the feature space. First step in the process is to calculate the 2d / 3d metrics on the merged cityjson which outputs a geo df, this is then used to execute the turning function"""
        # determine the number of processors to use and where the cityStats.py file is located, max function is used to prevent using all processors while at the same time ensuring that at least 1 processor is used
        n_processors = max(os.cpu_count() - 1, 1)

        # run the citystats script in which the metrics are calculated
        feature_space_metrics = calculate_metrics(input=self._merged_file, jobs=n_processors, verbose=self._verbose)

        # execute the turning function
        feature_space_tf = perform_turning_function(feature_space_metrics)

        # merge the results in one data frame & return the df
        feature_space_merged = feature_space_metrics.merge(feature_space_tf, on='id')
        feature_space_merged.drop(columns=['geometry'], inplace=True) # drop the geometry column as it is a shapely object and cannot be saved to a csv file

        return feature_space_merged

    def run(self):
        # merge if a folder has been provided
        if len(os.listdir(self._bag_data_folder)) == 0:
            raise Exception("The bag_data_folder is empty, please provide a folder with cityjson files")
        if self._bag_data_folder is not None:
            self._merge_files()

        # calculate the 2d / 3d metrics and turning function features
        # TODO: create a single progress bar
        self.feature_space = self._create_feature_space()
        if self._categorical_columns is not None:
            self.feature_space = pd.get_dummies(self.feature_space, columns=self._categorical_columns, dtype=int)
        
        self.feature_space.to_csv(self.feature_space_file)
