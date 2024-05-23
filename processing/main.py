import json
import os


from processing.merge_cityjson import MergeCityJSON


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

    def add_metrics(self):
        pass

    def main(self):
        # merge if not a single file has been provided
        if self.input_folder is not None:
            self.merge_files()

        # self.initate_gpkg()

        # calculate the 2d / 3d metrics and add them to the gpkg
        # self.add_metrics()

        # TODO: add the results from the turning function

        # save the gpkg
        # self.save()

if __name__ == '__main__':
    
    # p = processing(input_file=r"C:\Users\TimoScheidel\OneDrive - HAN\Future Factory\data\output/merged (3).city.json")
    p = processing("collection/input", convert=False)
    p.main()
    p.input_file