import sys
import os
import dotenv

module_dir = os.path.dirname(os.path.realpath(__file__)) # add the path of the module so it can be found
sys.path.append(module_dir)

dotenv.load_dotenv() # load the api key for the bag for the .env file

from collection import collection
from processing import processing
from similarity_calculation import similarity

class BuildingSimilarity():
    """ Combines the collection, processing and similarity classes to create a pipeline to calculate the similarity between buildings.
    Args:
        feature_space_file (str): The path to the output file (must be a .csv) where the processed data (=feature space) will be saved.
        bag_data_folder (str, optional): The path to the folder containing cityjson with a single building, source is from the BAG. Defaults to None.
        categorical_columns (list, optional): A list of column names that are categorical, these columns will get distance 1/num_categories if they are not equal and 0 if they are. Defaults to None.
        column_weights (dict, optional): A dictionary specifying the weights for the distance calculation of each column. Defaults to None.
        columns (list, optional): A list of column names to consider for the distance calculation, if used all columns bear the same weight. Defaults to None.
        verbose (bool, optional): If True, the progress will be printed. Defaults to False.
    """

    def __init__(self, 
                feature_space_file: str="data/feature_space/feature_space.csv",
                bag_data_folder: str = 'data/bag_data',
                all_ids: list = None,
                categorical_columns: list = None,
                column_weights: dict = None,
                columns: list = None,
                neighborhood_id: str = None,
                verbose: bool = False):
        self.collection = collection(bag_data_folder, all_ids, neighborhood_id, verbose)
        self.processing = processing(feature_space_file, bag_data_folder, self.collection.formatted_ids, categorical_columns, verbose)
        self.similarity = similarity(feature_space_file, column_weights, columns)

