import sys
import json
import os
import asyncio
import requests
import dotenv

dotenv.load_dotenv()

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

    def _convert_to_cityjson(self, data: dict, bag_attributes):
        """Converts the data (that is collected with the API request) to CityJSON format."""
        cityjson = data['metadata']
        cityjsonfeature = data['feature']
        cityjson['CityObjects'] = cityjsonfeature['CityObjects']
        cityjson['vertices'] = cityjsonfeature['vertices']

        if bag_attributes is not None:
            bag_id = list(cityjson['CityObjects'].keys())[0] 

            cityjson['CityObjects'][bag_id]['attributes']['totaal_oppervlakte'] = bag_attributes.get("totaal_oppervlakte")
            cityjson['CityObjects'][bag_id]['attributes']['aantal_verblijfsobjecten'] = bag_attributes.get("aantal_verblijfsobjecten")
            cityjson['CityObjects'][bag_id]['attributes']['gebruiksdoelen'] = bag_attributes.get("gebruiksdoelen")
        return cityjson

    def _make_url(self, id: str):
        return f"https://api.3dbag.nl/collections/pand/items/NL.IMBAG.Pand.{id}"

    def convert_id(self, id: str):
        if id.startswith("NL.IMBAG.Pand."): # remove the prefix if it is there
            id = id[14:]
        if '-' in id: # remove the suffix if it is there
            id = id.split('-')[0]
        return id

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
    
    def _process_bag_result(self, result: dict):
        all_atributes = []
        for bag_obj in result:
            # get the adressen and their info
            adressen = bag_obj.get('_embedded', {}).get('adressen')
            if adressen is None:
                print(f"Could not find adressen in the response json: {bag_obj}")
                return []
            
            # loop over the adresses to get the relevant information
            total_oppervlakte = 0
            gebruiksdoelen = []
            for adres in adressen:
                gebruiksdoelen_single_obj = adres.get('gebruiksdoelen', [])
                gebruiksdoelen.extend(gebruiksdoelen_single_obj)

                oppervlakte = adres.get('oppervlakte')
                if oppervlakte is None:
                    print(f'Surface cannot be found for adres: {adres}')
                else:
                    total_oppervlakte += oppervlakte
            if gebruiksdoelen == []:
                print(f'Could not identify the purpose of use for bag object: {bag_obj}')
            attributes = {'totaal_oppervlakte': total_oppervlakte, 'aantal_verblijfsobjecten': len(adressen), 'gebruiksdoelen': gebruiksdoelen} 
            all_atributes.append(attributes)
        return attributes

    def _get_bag_attributes(self, all_ids: list):
        # construct the header & parameters
        key = os.environ.get('BAG_API_KEY')
        headers = {'X-Api-Key':key, 'Accept-Crs': 'EPSG:28992'}
        all_params = [{'pandIdentificatie':id} for id in all_ids]

        # url is the same for every request
        url = 'https://api.bag.kadaster.nl/lvbag/individuelebevragingen/v2/adressenuitgebreid'
        all_urls = [url] * len(all_ids)

        # request the data & process the result
        result = asyncio.run(request_url_list(all_urls, headers=headers, params=all_params))
        extracted_bag_data = self._process_bag_result(result)
        return extracted_bag_data

    def collect_id_list(self, all_ids: list):
        """Requests and save the data in cityjson format for all the ids in the list."""

        self._check_bag_data_folder()

        # Remove the prefix from the ids so they are consistent. also check if the file already exists to avoid unnecessary requests
        request_ids = []
        existing_files = 0
        for id in all_ids:
            id = self.convert_id(id)

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

        # if asyncio is running, do the requests synchronously
        if asyncio.get_event_loop().is_running():
            error_count = 0
            if self._verbose:
                print("Asyncio is already running, so the requests will be done synchronously. Note that this is not the most efficient way.")
            for id in request_ids: # TODO: add a progress bar
                try:
                    r = requests.get(self._make_url(id))
                    cityjson = self._convert_to_cityjson(r.json())
                    self._save(cityjson, id)
                except:
                    error_count += 1
            if self._verbose:
                print(f"{error_count} errors occurred while requesting the data.")
        # if asyncio is not running, use the async function
        else:
            all_urls = [self._make_url(id) for id in request_ids]
            result = asyncio.run(request_url_list(all_urls))
            bag_attributes = self._get_bag_attributes(all_ids)

            # convert to the right format and save the data, this is not async because the data is already fetched
            for i, data in enumerate(result):
                cityjson = self._convert_to_cityjson(data, bag_attributes)
                self._save(cityjson, request_ids[i]) # use request_ids to get the right id without the pre- and suffix
            if self._verbose:
                print(f"{len(all_urls)-len(result)} errors occurred while requesting the data.")
        