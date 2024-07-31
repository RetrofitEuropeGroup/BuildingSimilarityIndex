from collection import collection
from processing import processing
from similarity_calculation import similarity

class BuildingSimilarity():
    """ Combines the collection, processing and similarity classes to create a pipeline to calculate the similarity between buildings.
    Args:
        feature_space_file (str): The path to the output file (must be a .csv) where the processed data (=feature space) will be saved.
        bag_data_folder (str, optional): The path to the folder containing cityjson with a single building, source is from the BAG. Defaults to None.
        cityjson_file (str, optional): The path to the CityJSON file with a single or multiple buildings. Defaults to None.
        categorical_columns (list, optional): A list of column names that are categorical, these columns will get distance 1/num_categories if they are not equal and 0 if they are. Defaults to None.
        column_weights (dict, optional): A dictionary specifying the weights for the distance calculation of each column. Defaults to None.
        columns (list, optional): A list of column names to consider for the distance calculation, if used all columns bear the same weight. Defaults to None.
    """

    def __init__(self, 
                feature_space_file: str="data/feature_space/feature_space.csv",
                bag_data_folder: str = None,
                cityjson_file: str = None,
                categorical_columns: list = None,
                column_weights: dict = None,
                columns: list = None):
        self.collection = collection()
        self.processing = processing(feature_space_file, bag_data_folder, cityjson_file, categorical_columns)
        self.similarity = similarity(feature_space_file, columns, column_weights)
