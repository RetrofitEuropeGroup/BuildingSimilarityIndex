import json
import os
import asyncio
from tqdm import tqdm
import aiohttp
import random
import time

from collection.WFS_roof import roofmetrics
from collection.hoodcollector import hoodcollector


class collection():
    """
    A class that assists in the collection of the data from the 3D-BAG api. Provide a list to the collect_id_list function 
    and it request it from the 3D-BAG, then it saves the data in cityjson format to the bag_data folder.
    """

    def __init__(self, bag_data_folder: str, all_ids: list = None, neighborhood_id: str = None,  verbose: bool = False):
        self._bag_data_folder = bag_data_folder
        self._check_bag_data_folder() # creates the dir if needed, also check if it is not a file
        self._verbose = verbose
        self.key = self._get_key()
        self.formatted_ids = self.format_ids(all_ids, neighborhood_id)

    def _get_key(self):
        key = os.environ.get('BAG_API_KEY')
        if key is None:
            raise ValueError("WARNING: the BAG_API_KEY is None. Make sure you include BAG_API_KEY in the .env file in the root folder")
        return key
    
    def format_ids(self, all_ids=None, neighborhood_id=None):
        if neighborhood_id is not None:
            all_ids = asyncio.run(hoodcollector(neighborhood_id, verbose=self._verbose))
        elif all_ids is None:
            raise ValueError("Either all_ids or neighborhood_id should be provided to the BuildingSimilarity class.")

        formatted_ids = []
        for id in all_ids:        
            if id.startswith("NL.IMBAG.Pand."): # remove the prefix if it is there
                id = id[14:]
            if '-' in id: # remove the suffix if it is there
                id = id.split('-')[0]
            formatted_ids.append(id)
        return formatted_ids

    def _convert_to_cityjson(self, data: dict, bag_attributes: dict, roof_attributes: dict, id: str):
        """Converts the data (that is collected with the API request) to CityJSON format."""
        # create the cityjson object
        cityjson = data['metadata']
        cityjson['CityObjects'] = data['feature']['CityObjects']
        cityjson['vertices'] = data['feature']['vertices']

        # add the attributes to the cityjson object
        all_variables = ['totaal_oppervlakte', 'aantal_verblijfsobjecten', 'gebruiksdoelen', 'main_roof_parts', 'main_roof_area', 'other_roof_parts', 'other_roof_area', 'part_ratio', 'area_ratio']
        attributes = {**bag_attributes, **roof_attributes}
        for var in all_variables:
            cityjson['CityObjects'][f"NL.IMBAG.Pand.{id}"]['attributes'][var] = attributes.get(var)
        return cityjson

    def _make_url(self, id: str):
        return f"https://api.3dbag.nl/collections/pand/items/NL.IMBAG.Pand.{id}"

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
            return {}
        
        # loop over the adresses to get the relevant information
        total_oppervlakte = 0
        gebruiksdoelen = []
        for adres in adressen:
            gebruiksdoelen_single_obj = adres.get('gebruiksdoelen', [])
            gebruiksdoelen.extend(gebruiksdoelen_single_obj)

            oppervlakte = adres.get('oppervlakte')
            if oppervlakte is None:
                raise ValueError(f'Surface cannot be found for building {id}')
            else:
                total_oppervlakte += oppervlakte
        if gebruiksdoelen == []:
            raise ValueError(f"Could not identify the use purpose for building: {id}")
        attributes = {'totaal_oppervlakte': total_oppervlakte, 'aantal_verblijfsobjecten': len(adressen), 'gebruiksdoelen': gebruiksdoelen} 
        return attributes

    async def _get_additional_bag_attributes(self, id: str, session, retries=0):
        await asyncio.sleep(min(random.normalvariate(0.5), 0.1)) # to avoid the rate limit, this does not affect speed as the 3d_bag call takes way longer anyway
        
        headers = {'X-Api-Key':self.key, 'Accept-Crs': 'EPSG:28992'}
        url = 'https://api.bag.kadaster.nl/lvbag/individuelebevragingen/v2/adressenuitgebreid'

        params = {'pandIdentificatie':id}
        r = await session.get(url, params=params, headers=headers)
        if r.status == 429 and retries < 1:
            await asyncio.sleep(2)
            return await self._get_additional_bag_attributes(id, session, retries+1)
        r.raise_for_status()
        bag_result = await r.json()
        return self._process_bag_result(bag_result, id)

    async def _get_3d_bag(self, id: str, session, retries=0):
        url = self._make_url(id)
        r = await session.get(url)
        if (r.status == 429 or r.status == 502) and retries < 1:
            await asyncio.sleep(2) # to avoid the rate limit
            return await self._get_3d_bag(id, session, retries+1)
        r.raise_for_status()
        r = await r.json()
        return r

    def _get_bbox(self, data: dict):
        """Get the bounding box of a building from the 3D-BAG API."""
        translate = data['metadata']['transform']['translate']
        centroid_x, centroid_y = translate[0], translate[1]
        return f"{centroid_x-10},{centroid_y-10},{centroid_x+10},{centroid_y+10}"

    async def _get_roof_attributes(self, id: str, data: dict, session):
        bbox = self._get_bbox(data)
        return await roofmetrics(id, bbox, session=session)

    def _set_request_ids(self, force_new=False):
        """Sets the request_ids attribute to a list of ids that are not already saved on the machine."""
        self.request_ids = []
        existing_files = 0
        for id in self.formatted_ids:

            if force_new == False and os.path.exists(f"{self._bag_data_folder}/{id}.city.json"):
                existing_files += 1
            else:
                self.request_ids.append(id)

        # check how many requests are needed
        if existing_files == len(self.formatted_ids):
            print("All the requested data is already saved on the machine, no new requests are needed.")
        elif existing_files > 0:
            print(f"{existing_files} out of {len(self.formatted_ids)} files already exist, so {len(self.formatted_ids) - existing_files} more request(s) are needed.")

    async def _async_collect_building(self, id):
        start_time = time.time()
        async with self.semaphore: # used to limit the amount of requests at the same time
            try:
                async with aiohttp.ClientSession() as session:
                    tasks = [self._get_3d_bag(id, session), self._get_additional_bag_attributes(id, session)]
                    bag3d_data, bag_attributes = await asyncio.gather(*tasks)
                    roof_attributes = await self._get_roof_attributes(id, bag3d_data, session)
                if roof_attributes is None:
                    raise Exception('Roof attributes are not available')                    
                cityjson = self._convert_to_cityjson(bag3d_data, bag_attributes, roof_attributes, id)
                self._save(cityjson, id) # use request_ids to get the right id without the pre- and suffix
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                if self._verbose:
                    print(f"  Request for {id} failed: {e}")
                self.errors += 1
            self.bar.update(1)
        if time.time() - start_time < 1:
            print(f"Sleeping for {1 - (time.time() - start_time)} seconds")
            await asyncio.sleep(1 - (time.time() - start_time))

    async def async_request_data(self):
        self.errors = 0
        self.bar = tqdm(total=len(self.request_ids), desc="Requesting data", position=0, leave=True)

        self.semaphore = asyncio.Semaphore(50)
        tasks = [self._async_collect_building(id) for id in self.request_ids]
        await asyncio.gather(*tasks)
        self.bar.close()

        print(f"{self.errors} error(s) occurred while requesting the data, collected {len(self.request_ids) - self.errors} building(s).")

    def collect_id_list(self, force_new=False):
        """Requests and save the data in cityjson format for all the ids in the list."""
        # Remove the prefix from the ids so they are consistent. also check if the file already exists to avoid unnecessary requests
        self._set_request_ids(force_new)

        if len(self.request_ids):
            asyncio.run(self.async_request_data())
        # else: there is no data to request
