import geopandas as gpkg

from merge_cityjson import MergeCityJSON


class processing():
    def __init__(self, input_folder: str = None, input_file: str = None, convert:bool = False):
        if input_folder is not None and input_file is not None:
            raise ValueError("It is not possible to use both input_folder and input_file")
        
        self.convert = convert
        self.input_folder = input_folder
        self.input_file = input_file
        self.merge = True

    def convert_files(self):
        # TODO: implement
        # convert the .json files to .city.jsonl files
        # return the path to the folder containing the .city.jsonl files
        pass

    def get_input(self):
        merger = MergeCityJSON(self.input_folder)
        merger.run()
        self.input_file = merger.file_path

    def add_metrics(self):
        pass

    def main(self):
        if self.convert:
            self.convert_files()

        # merge if needed
        if self.input_folder is not None:
            self.get_input()

        # self.initate_gpkg()

        # calculate the 2d / 3d metrics and add them to the gpkg
        self.add_metrics()

        # TODO: add the results from the turning function

        # save the gpkg
        # self.save()

if __name__ == '__main__':
    
    p = processing(input_file=r"output/merged (3).city.json")
    p.main()