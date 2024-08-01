import sys
import json
import os
import asyncio
import requests

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from collection.async_requests import request_url_list

class collection():
    """
    A class that assists in the collection of the data from the 3D-BAG api. Provide a list to the collect_id_list function 
    and it request it from the 3D-BAG, then it saves the data in cityjson format to the bag_data folder.
    """

    def __init__(self, bag_data_folder: str, verbose: bool = False):
        self._bag_data_folder = bag_data_folder
        self._verbose = verbose # TODO: integrate this in the request function

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
        # else: the folder is there already

    def collect_id_list(self, all_ids: list):
        # TODO: check if exists with different id formats
        """Requests and save the data in cityjson format for all the ids in the list."""

        self._check_bag_data_folder()

        # Remove the prefix from the ids so they are consistent. also check if the file already exists to avoid unnecessary requests
        request_ids = []
        existing_files = 0
        for id in all_ids:
            if id.startswith("NL.IMBAG.Pand."):
                id = id[14:]

            if os.path.exists(f"{self._bag_data_folder}/{id}.city.json"):
                existing_files += 1
            else:
                request_ids.append(id)

        # check how many requests are needed
        if existing_files == len(all_ids):
            print("All the requested data is already saved on the machine, no new requests are needed.")
            return
        elif existing_files > 0:
            print(f"{existing_files} out of {len(all_ids)} files already exist, so {len(all_ids) - existing_files} more request(s) are needed.")
        
        ## make the urls for the async requests
        all_urls = [f"https://api.3dbag.nl/collections/pand/items/NL.IMBAG.Pand.{id}" for id in request_ids]
        if asyncio.get_event_loop().is_running():
            print("Asyncio is already running, so the requests will be done synchronously. Note that this is not the most efficient way.")
            for url in all_urls:
                try:
                    r = requests.get(url)
                    cityjson = self._convert_to_cityjson(r.json())
                    self._save(cityjson, request_ids[all_urls.index(url)])
                except:
                    error_count += 1
            if self._verbose:
                print(f"{error_count} errors occurred while requesting the data.")
        else:
            result = asyncio.run(request_url_list(all_urls))

            # convert to the right format and save the data, this is not async because the data is already fetched
            for i, data in enumerate(result):
                cityjson = self._convert_to_cityjson(data)
                self._save(cityjson, all_ids[i])
            if self._verbose:
                print(f"{len(all_urls)-len(result)} errors occurred while requesting the data.")
        