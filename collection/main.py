import sys
import json
import os
import asyncio
import requests
import dotenv
from tqdm import tqdm

dotenv.load_dotenv() # load the api key for the bag for the .env file

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

    def _convert_to_cityjson(self, data: dict, bag_attributes: dict, id):
        """Converts the data (that is collected with the API request) to CityJSON format."""
        cityjson = data['metadata']
        cityjsonfeature = data['feature']
        cityjson['CityObjects'] = cityjsonfeature['CityObjects']
        cityjson['vertices'] = cityjsonfeature['vertices']

        bag_id = f"NL.IMBAG.Pand.{id}"
        cityjson['CityObjects'][bag_id]['attributes']['totaal_oppervlakte'] = bag_attributes[id].get("totaal_oppervlakte")
        cityjson['CityObjects'][bag_id]['attributes']['aantal_verblijfsobjecten'] = bag_attributes[id].get("aantal_verblijfsobjecten")
        cityjson['CityObjects'][bag_id]['attributes']['gebruiksdoelen'] = bag_attributes[id].get("gebruiksdoelen")
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
    
    def _process_bag_result(self, result: dict, id):
        # get the adressen and their info
        adressen = result.get('_embedded', {}).get('adressen')
        if adressen is None: #TODO: check why some are missing
            print(f"Could not find adressen in the response json: {id}")
            return {}
        
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
            print(f"Could not identify the purpose of use for bag object: {id}")
        attributes = {'totaal_oppervlakte': total_oppervlakte, 'aantal_verblijfsobjecten': len(adressen), 'gebruiksdoelen': gebruiksdoelen} 
        return attributes

    def _get_bag_attributes(self, request_ids: list):
        #TODO: make it optional to use the bag
        
        url = 'https://api.bag.kadaster.nl/lvbag/individuelebevragingen/v2/adressenuitgebreid'
        key = os.environ.get('BAG_API_KEY') #TODO: make sure this is an option
        if key is None:
            print("WARNING: the BAG_API_KEY is None. Make sure you include BAG_API_KEY in the .env file in the root folder")
        headers = {'X-Api-Key':key, 'Accept-Crs': 'EPSG:28992'}
        

        all_attributes = {}
        for id in tqdm(request_ids, desc='Getting the BAG data'):
            try:
                params = {'pandIdentificatie':id}
                r = requests.get(url, params=params, headers=headers).json()
                attributes = self._process_bag_result(r, id)
                all_attributes[id] = attributes
            except Exception as e:
                print(f"Error while getting the information for id {id} from {url}. \n Error message: {e}")
                all_attributes[id] = {}
        return all_attributes

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
        if asyncio.get_event_loop().is_running(): #TODO: this doesn't work if you run the class twice
            error_count = 0
            if self._verbose:
                print("Asyncio is already running, so the requests will be done synchronously. Note that this is not the most efficient way.")
            for id in request_ids: # TODO: add a progress bar
                try:
                    r = requests.get(self._make_url(id))
                    cityjson = self._convert_to_cityjson(r.json(), bag_attributes, id)
                    self._save(cityjson, id)
                except:
                    error_count += 1
            if self._verbose:
                print(f"{error_count} errors occurred while requesting the data.")
        # if asyncio is not running, use the async function
        else:
            all_urls = [self._make_url(id) for id in request_ids]
            bag_attributes = self._get_bag_attributes(request_ids)
            result = asyncio.run(request_url_list(all_urls))

            # convert to the right format and save the data, this is not async because the data is already fetched
            for id, data in zip(request_ids, result):
                if data is None:
                    continue
                cityjson = self._convert_to_cityjson(data, bag_attributes, id)
                self._save(cityjson, id) # use request_ids to get the right id without the pre- and suffix
            if self._verbose: #TODO: doesn't work anymore as we save all result
                print(f"{len(all_urls)-len(result)} errors occurred while requesting the data.")
        