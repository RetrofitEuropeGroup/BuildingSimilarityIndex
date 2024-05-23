import json
import os

try:
    from processing.merge_cityjson import MergeCityJSON
except:
    from merge_cityjson import MergeCityJSON


class processing():
    def __init__(self, input_folder: str = None, input_file: str = None):
        if input_folder is not None and input_file is not None:
            raise ValueError("It is not possible to use both input_folder and input_file")
        
        self.input_folder = input_folder
        self.input_file = input_file

    def merge_files(self):
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
        self.gpkg_path = self.input_file.replace('.city.json', '.gpkg')
        command = f'python {city_stats_location} "{self.input_file}" -j {n_processors} -o "{self.gpkg_path}"'
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
    
    # p = processing(input_file=r"C:\Users\TimoScheidel\OneDrive - HAN\Future Factory\data\output/merged (3).city.json")
    p = processing("collection/input")
    p.run()