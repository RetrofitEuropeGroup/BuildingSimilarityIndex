import os
import sys
from pathlib import Path

# add the path of the own directory to the system path so that the modules can be imported
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)

from merge_cityjson import MergeCityJSON
from metrics.cityStats import process_cityjson

class processing():
    def __init__(self, gpkg_path: str, bag_data_folder: str = None, cityjson_path: str = None):
        if bag_data_folder is None and cityjson_path is None:
            raise ValueError("Either input_folder or cityjson_path should be provided")
        elif bag_data_folder is not None and cityjson_path is not None:
            raise ValueError("It is not possible to use both bag_data_folder and cityjson_path")

        self.bag_data_folder = bag_data_folder
        self.cityjson_path = cityjson_path
        self.validate_gpkg_path(gpkg_path)


    def validate_gpkg_path(self, file_path: str):
        if isinstance(file_path, str) == False:
            raise ValueError("The gpkg_path (which is the output file) should be a string")
        elif file_path.endswith('.gpkg') == False:
            raise ValueError("""The gpkg_path (which is the output file) is the path to where the geopackage
                              data frame will be saved. Extension should be .gpkg""")
        else:
            self.gpkg_path = file_path
        parent_dir = os.path.dirname(self.gpkg_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

    # main functions
    def merge_files(self):
        """ Merges the files in the input folder to a single file"""
        merge_folder = Path(self.bag_data_folder.replace('bag_data', 'bag_data_merged')) # create a new folder, if needed, for the merged files
        merger = MergeCityJSON(self.bag_data_folder, output_folder=merge_folder)
        merger.run()

        self.cityjson_path = merger.file_path

    def initiate_gpkg(self):
        """ Initiates the .gpkg path with the 2d and 3d metrics"""
        # determine the number of processors to use and where the cityStats.py file is located
        n_processors = max(os.cpu_count() - 1, 1)
        if os.getcwd().endswith('processing'):
            city_stats_location =  "metrics/cityStats.py"
        else:
            city_stats_location =  "processing/metrics/cityStats.py"

        # run the citystats script
        process_cityjson(self.cityjson_path, self.gpkg_path, jobs=n_processors)

    def run(self):
        # merge if not a single file has been provided
        if self.bag_data_folder is not None:
            self.merge_files()

        # calculate the 2d / 3d metrics and save them to a gpkg
        self.initiate_gpkg()

        # TODO: add the results from the turning function, you can load the geometry from the gpkg.
        # the filepath of the gpkg is stored in self.gpkg_path


if __name__ == '__main__':
    p = processing(gpkg_path="data/gpkg/testcase_overvecht.gpkg", bag_data_folder="data/bag_data")
    # p = processing(cityjson_path="analysis/voorbeeldwoningen.city.json", gpkg_path="collection/output/output.gpkg")
    p.run()
    import geopandas as gpd
    gdf = gpd.read_file(p.gpkg_path)
    print(gdf.columns)