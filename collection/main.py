import sys
import json
import os
import asyncio

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from collection.async_requests import request_url_list

class collection():
    """
    A class that assists in the collection of the data from the 3D-BAG api. Provide a list to the collect_id_list function 
    and it request it from the 3D-BAG, then it saves the data in cityjson format to the bag_data folder.
    """

    def __init__(self):
        self._bag_data_folder = 'data/bag_data'

    def _convert_to_cityjson(self, data: dict):
        """Converts the data (that is collected with the API request) to CityJSON format."""
        cityjson = data['metadata']
        cityjsonfeature = data['feature']
        cityjson['CityObjects'] = cityjsonfeature['CityObjects']
        cityjson['vertices'] = cityjsonfeature['vertices']
        return cityjson

    def _save(self, cityjson: dict, id: str):
        """Saves the CityJSON data to a file."""
        with open(f'{self._bag_data_folder}/{id}.city.json', 'w') as file:
            json.dump(cityjson, file)

    def _check_bag_data_folder(self):
        """Checks if the bag_data folder exists and is a folder, if not it will be created. If it exists and is not empty, a warning is printed."""
        if not os.path.exists(self._bag_data_folder):
            os.makedirs(self._bag_data_folder)
        elif not os.path.isdir(self._bag_data_folder):
            raise ValueError("The bag_data_folder should be a directory, not a file.")
        elif len(os.listdir(self._bag_data_folder)) > 1:
            print("WARNING: the bag_data folder is not empty, the data will be overwritten if it already exists")
        elif len(os.listdir(self._bag_data_folder)) == 1 and os.path.exists(f"{self.bag_data_folder}/.gitkeep") == False:
            print("WARNING: the bag_data folder is not empty, the data will be overwritten if it already exists")
        # else: the folder is empty, so no warning is needed

    def collect_id_list(self, all_ids: list):
        """Requests and save the data in cityjson format for all the ids in the list."""

        self._check_bag_data_folder()

        # Remove the prefix from the ids so they are consistent. also check if the file already exists to avoid unnecessary requests
        request_ids = []
        for id in all_ids:
            if id.startswith("NL.IMBAG.Pand."):
                id = id[14:]

            if os.path.exists(f"{self._bag_data_folder}/{id}.city.json"):
                print(f"File {id}.city.json already exists, skipping")
            else:
                request_ids.append(id)

        # make the urls for the async requests
        all_urls = [f"https://api.3dbag.nl/collections/pand/items/NL.IMBAG.Pand.{id}" for id in request_ids]
        result = asyncio.run(request_url_list(all_urls))

        # convert to the right format and save the data, this is not async because the data is already fetched
        for i, data in enumerate(result):
            cityjson = self._convert_to_cityjson(data)
            self._save(cityjson, all_ids[i])

if __name__ == "__main__":
    import geopandas as gpd
    df = gpd.read_file("testcase_overvecht/BSI_overvecht.gpkg", layer='pand')
    all_ids = df['identificatie'].tolist()

    start_id = 0
    step = 100
    for i in range(start_id, len(all_ids), step):
        c = collection()
        c.collect_id_list(all_ids[i:i+step])
        print(f'Processed building {i} to {i+step} out of total {len(all_ids)} buildings')