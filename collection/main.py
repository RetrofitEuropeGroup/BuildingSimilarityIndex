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
from collection.WFS_roof import roofmetrics

class collection():
    """
    A class that assists in the collection of the data from the 3D-BAG api. Provide a list to the collect_id_list function 
    and it request it from the 3D-BAG, then it saves the data in cityjson format to the bag_data folder.
    """

    def __init__(self, bag_data_folder: str, verbose: bool = False):
        self._bag_data_folder = bag_data_folder
        self._verbose = verbose # TODO: integrate this in the request function

    def _convert_to_cityjson(self, data: dict, bag_attributes: dict, roof_attributes: dict, id: str):
        """Converts the data (that is collected with the API request) to CityJSON format."""
        cityjson = data['metadata']
        cityjsonfeature = data['feature']
        cityjson['CityObjects'] = cityjsonfeature['CityObjects']
        cityjson['vertices'] = cityjsonfeature['vertices']

        bag_id = f"NL.IMBAG.Pand.{id}"
        cityjson['CityObjects'][bag_id]['attributes']['totaal_oppervlakte'] = bag_attributes[id].get("totaal_oppervlakte")
        cityjson['CityObjects'][bag_id]['attributes']['aantal_verblijfsobjecten'] = bag_attributes[id].get("aantal_verblijfsobjecten")
        cityjson['CityObjects'][bag_id]['attributes']['gebruiksdoelen'] = bag_attributes[id].get("gebruiksdoelen")
        cityjson['CityObjects'][bag_id]['attributes']['main_roof_parts'] = roof_attributes[id].get("main_roof_parts")
        cityjson['CityObjects'][bag_id]['attributes']['main_roof_area'] = roof_attributes[id].get("main_roof_area")
        cityjson['CityObjects'][bag_id]['attributes']['other_roof_parts'] = roof_attributes[id].get("other_roof_parts")
        cityjson['CityObjects'][bag_id]['attributes']['other_roof_area'] = roof_attributes[id].get("other_roof_area")
        cityjson['CityObjects'][bag_id]['attributes']['part_ratio'] = roof_attributes[id].get("part_ratio")
        cityjson['CityObjects'][bag_id]['attributes']['area_ratio'] = roof_attributes[id].get("area_ratio")
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
        if adressen is None:
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
            raise ValueError("WARNING: the BAG_API_KEY is None. Make sure you include BAG_API_KEY in the .env file in the root folder")
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

    def _get_bbox(self, id: str, data: dict):
        """Get the bounding box of a building from the 3D-BAG API."""
        if "NL.IMBAG.Pand." + id not in data['feature']['CityObjects']:
            print(f"WARNING: Could not find the id {id} in the data.")
            return None #TODO: remove after a while
        translate = data['metadata']['transform']['translate']
        centroid_x, centroid_y = translate[0], translate[1]
        #TODO: check if a smaller bbox is faster
        return f"{centroid_x-100},{centroid_y-100},{centroid_x+100},{centroid_y+100}"

    def _get_roof_attributes(self, request_ids: list):
        roof_attributes = {} #TODO: we might want to request asynchronously
        for id, data in tqdm(zip(request_ids, self.result), desc='Getting the roof data', total=len(request_ids)):
            bbox = self._get_bbox(id, data)
            roof_attributes[id] = roofmetrics(id, bbox)
        return roof_attributes

    def _set_request_ids(self, all_ids, force_new=False):
        self.request_ids = []
        existing_files = 0
        for id in all_ids:
            id = self.convert_id(id)

            if force_new == False and os.path.exists(f"{self._bag_data_folder}/{id}.city.json"):
                existing_files += 1
            else:
                self.request_ids.append(id)

        # check how many requests are needed
        if existing_files == len(all_ids):
            print("All the requested data is already saved on the machine, no new requests are needed.")
        elif existing_files > 0:
            print(f"{existing_files} out of {len(all_ids)} files already exist, so {len(all_ids) - existing_files} more request(s) are needed.")

    async def async_request_data(self):
        all_urls = [self._make_url(id) for id in self.request_ids]
        bag_attributes = self._get_bag_attributes(self.request_ids)
        self.result = await request_url_list(all_urls)
        roof_attributes = self._get_roof_attributes(self.request_ids)

        # convert to the right format and save the data, this is not async because the data is already fetched
        for id, data in zip(self.request_ids, self.result): # TODO: check if id corresponds to the id in the data
            if data is None:
                continue
            cityjson = self._convert_to_cityjson(data, bag_attributes, roof_attributes, id)
            self._save(cityjson, id) # use request_ids to get the right id without the pre- and suffix
        if self._verbose: #TODO: doesn't work anymore as we save all results
            print(f"{len(all_urls)-len(self.result)} errors occurred while requesting the data.")

    def _request_data(self):
        #TODO: implement this function
        error_count = 0
        if self._verbose:
            print("Asyncio is already running, so the requests will be done synchronously. Note that this is not the most efficient way.")
        for id in self.request_ids: # TODO: add a progress bar
            try:
                r = requests.get(self._make_url(id))
                cityjson = self._convert_to_cityjson(r.json(), bag_attributes, id) #TODO: make it work properly with bag_attributes
                self._save(cityjson, id)
            except:
                error_count += 1
        if self._verbose:
            print(f"{error_count} errors occurred while requesting the data.")

    def collect_id_list(self, all_ids: list, force_new=False):
        """Requests and save the data in cityjson format for all the ids in the list."""

        self._check_bag_data_folder()

        # Remove the prefix from the ids so they are consistent. also check if the file already exists to avoid unnecessary requests
        self._set_request_ids(all_ids, force_new)

        # if asyncio is running, do the requests synchronously
        if len(self.request_ids) and asyncio.get_event_loop().is_running(): #TODO: this doesn't work if you run the class twice
            raise NotImplementedError("The async_request_data function cannot be called while asyncio is running.")
        # if asyncio is not running, use the async function
        elif len(self.request_ids):
            asyncio.run(self.async_request_data())
        # else: there is no data to request
        