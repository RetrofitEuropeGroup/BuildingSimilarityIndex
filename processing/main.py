import os
import sys
from pathlib import Path

# if we run the script from the processing folder, we need to add the parent folder to the path
# otherwise it won't search for processing.merge_cityjson in the root folder of the repository
# this is useful for testing
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from processing.merge_cityjson import MergeCityJSON

class processing():
    def __init__(self, input_folder: str = None, input_file: str = None, gpkg_path: str = None):
        if input_folder is None and input_file is None:
            raise ValueError("Either input_folder or input_file should be provided")
        elif input_folder is not None and input_file is not None:
            raise ValueError("It is not possible to use both input_folder and input_file")
        
        self.input_folder = input_folder
        self.input_file = input_file
        self._set_gpkg_path(gpkg_path)
        

    def _set_gpkg_path(self, gpkg_path):
        if gpkg_path is None and self.input_file is not None:
            self.gpkg_path = self.input_file.replace('.city.json', '.gpkg')
        elif gpkg_path is None and self.input_folder is not None:
            # TODO: change this to a more dynamic name
            # TODO: we should not assume that the output folder is called output and that the input folder is called input
            self.gpkg_path = f"{self.input_folder.replace('input', 'output')}/output.gpkg" 
        elif isinstance(gpkg_path, str) == False:
            raise ValueError("The gpkg_path (output file) should be a string")
        elif gpkg_path.endswith('.gpkg') == False:
            raise ValueError("The gpkg_path (output file) is the path to where the geopackage data frame will be saved. Extension should be .gpkg")
        else:
            self.gpkg_path = gpkg_path

    # main functions
    def merge_files(self):
        """ Merges the files in the input folder to a single file"""
        merger = MergeCityJSON(self.input_folder)
        merger.run()
        self.input_file = merger.file_path

    def initiate_gpkg(self):
        """ Initiates the gpkg file with the 2d and 3d metrics"""
        # determine the number of processors to use and where the cityStats.py file is located
        n_processors = os.cpu_count() - 2
        if os.getcwd().endswith('processing'):
            city_stats_location =  "metrics/cityStats.py"
        else:
            city_stats_location =  "processing/metrics/cityStats.py"

        # run the citystats script
        command = f'python {city_stats_location} {self.input_file} -j {n_processors} -o {self.gpkg_path}'
        os.system(command)

    def run(self):
        # merge if not a single file has been provided
        if self.input_folder is not None:
            self.merge_files()

        # calculate the 2d / 3d metrics and add them to the gpkg
        self.initiate_gpkg()

        # TODO: add the results from the turning function, you can load the geometry from the gpkg.
        # the filepath of the gpkg is stored in self.gpkg_path


if __name__ == '__main__':
    p = processing("collection/input")
    p.run()
    import geopandas as gpd
    gdf = gpd.read_file(p.gpkg_path)
    print(gdf.columns)