import os
import json

from cjio.cityjson import CityJSON

class MergeCityJSON:
    """Class to merge multiple CityJSON files into a single CityJSON file. 
    Note that the input_folder must only contain files that are to be merged. 
    The merged file will be saved a seperate output folder.
    Note that there are several format that are very similar to CityJSON, 
    such as CityJSONL, CityGML, and GeoJSON. This repo only handles CityJSON files."""

    def __init__(self, input_folder: str, output_folder: str = None, formatted_ids: list = None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.formatted_ids = formatted_ids
        self.file_name = 'merged.city.json'

    def merge_objects(self):
        if len(self.all_objects) == 0:
            raise Exception("No objects found to merge. Please check the input folder and if applicable, the formatted_ids list.")
        elif len(self.all_objects) == 1:
            print("WARNING: Only one object found, no need to merge.") # the first obj is just used as the "merged" obj
        else:
            # merge the other objects into the first object, otherwise return the first object
            self.all_objects[0].merge(self.all_objects[1:])
        self.merged_obj = self.all_objects[0].j # is the merged object in json format

    def load_objects(self):
        self.all_objects = []
        for file in os.listdir(self.input_folder):
            if 'merged' in file:
                continue
            elif file.endswith('city.json') and (self.formatted_ids is None or file.split('.')[0] in self.formatted_ids):
                with open(f"{self.input_folder}/{file}", 'r') as f:
                    cm = CityJSON(f)
                self.all_objects.append(cm)

    def prepare_output_folder(self):
        # set the output folder based on the input folder if not specified
        if self.output_folder is None:
            self.output_folder = self.input_folder.replace('input', 'output')
        
        # make sure the output folder exists
        if os.path.exists(self.output_folder) == False:
            os.mkdir(self.output_folder)

    def save(self):
        self.file_path = f"{self.output_folder}/{self.file_name}"
        with open(self.file_path, 'w') as f:
            json.dump(self.merged_obj, f)

    def run(self):
        self.prepare_output_folder()

        # load & merge
        self.load_objects()
        self.merge_objects()

        # save the merged object
        self.save()
