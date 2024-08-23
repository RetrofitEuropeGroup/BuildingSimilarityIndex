import sys
import os

module_dir = os.path.dirname(os.path.realpath(__file__)) # add the path of the module so it can be found
sys.path.append(module_dir)

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
                verbose: bool = False):
        formatted_ids = self.format_ids(all_ids)
        self.collection = collection(bag_data_folder, formatted_ids, verbose)
        self.processing = processing(feature_space_file, bag_data_folder, formatted_ids, categorical_columns, verbose)
        self.similarity = similarity(feature_space_file, column_weights, columns)

    def format_ids(self, all_ids):
        formatted_ids = []
        for id in all_ids:        
            if id.startswith("NL.IMBAG.Pand."): # remove the prefix if it is there
                id = id[14:]
            if '-' in id: # remove the suffix if it is there
                id = id.split('-')[0]
            formatted_ids.append(id)
        return formatted_ids

if __name__ == "__main__":
    import geopandas as gpd
    import json

    with open('analysis/column_weights.json', 'r') as f:
        column_weights = json.load(f)
    
    if 1:
        df = gpd.read_file(r'C:\Users\TimoScheidel\OneDrive - HAN\Future Factory\FF_BuildingSimilarityIndex\analysis\subset20k.gpkg')
        all_ids = df['id'].tolist()
        bs = BuildingSimilarity(bag_data_folder='data/20K', all_ids=all_ids[:10], verbose=True, column_weights=column_weights, feature_space_file='data/feature_space/fs_20k_test.csv')

        bs.collection.collect_id_list(force_new=False)
        bs.processing.run()
    else:
        all_ids = ["0153100000203775", "0153100000277229", "0772100000262212",
        "0153100000213600", "0327100000255061", "0327100000258432",
        "0327100000252015", "0327100000264673", "0307100000377568",
        "0307100000326243", "0307100000337962", "0402100001519973"]
        bs = BuildingSimilarity(bag_data_folder='data/voorbeelden', verbose=True, column_weights=column_weights, feature_space_file='data/feature_space/fs_voorbeelden.csv')
        bs.collection.collect_id_list(all_ids, force_new=True)
        bs.processing.run()
    # bs.similarity.distance_matrix_reference(reference_feature_space='data/feature_space/fs_voorbeelden.csv', dist_matrix_path='data/distance_matrix/dm_reference3.csv')