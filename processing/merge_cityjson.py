import os
import json

from cjio.cityjson import CityJSON


class MergeCityJSON:
    """Class to merge multiple CityJSON files into a single CityJSON file. 
    Note that the input_folder must only contain files that are to be merged. 
    The merged file will be saved a seperate output folder.
    Note that there are several format that are very similar to CityJSON, 
    such as CityJSONL, CityGML, and GeoJSON. This repo only handles CityJSON files."""

    def __init__(self, input_folder: str, output_folder: str = None):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def merge_objects(self):
        if len(self.all_objects) > 1:
            # merge the other objects into the first object, otherwise return the first object
            self.all_objects[0].merge(self.all_objects[1:])
        else:
            print("WARNING: Only one object found, no need to merge.")
        self.merged_obj = self.all_objects[0].j # is the merged object in json format

    def load_objects(self):
        all_objects = []
        for file in os.listdir(self.input_folder):
            if 'merged' in file:
                print('Merged file found, should not be in the input folder. Skipping...')
                continue
            elif file.endswith('city.json') or file.endswith('city.jsonl'):
                cm = CityJSON(open(f"{self.input_folder}/{file}", 'r'))
                all_objects.append(cm)
        self.all_objects = all_objects

    def prepare_output_folder(self):
        # set the output folder based on the input folder if not specified
        if self.output_folder is None:
            self.output_folder = self.input_folder.replace('input', 'output')
        
        # make sure the output folder exists
        if os.path.exists(self.output_folder) == False:
            os.mkdir(self.output_folder)

    def create_output_name(self):
        self.prepare_output_folder()

        # create a name for the merged file
        file_name = 'merged_1.city.json'
        while file_name in os.listdir(self.output_folder):
            version_num = int(file_name.split('_')[1].split('.')[0])
            file_name = f"merged ({version_num + 1}).city.json"
        self.file_name = file_name

    def save(self):
        with open(f"{self.output_folder}/{self.file_name}", 'w') as f:
            json.dump(self.merged_obj, f)

# def rename_files(input_folder):
#     # rename files to the correct format
#     for file in os.listdir(input_folder):
#         if not file.endswith('city.json') and file.endswith('.json'):
#             os.rename('input/' + file, 'input/' + file[:-5] + '.city.json')


    def run(self):
        # load & merge
        self.load_objects()
        self.merge_objects()
        print(len(self.merged_obj))
        print(len(self.all_objects))
        
        # save the merged object
        self.create_output_name()
        self.save()


if __name__ == '__main__':
    merger = MergeCityJSON(r'input_folder', 'test_output')
    merger.run()
    