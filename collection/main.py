import requests
import json
import os


class collection():
    def __init__(self, input_folder: str = "./collection/input"):
        self.input_folder = input_folder

    def request_id(self, id: str):
        """Uses the 3dbag API to request data for a specific id and writes it to a jsonl file."""
        url = f"https://api.3dbag.nl/collections/pand/items/{id}"

        # Request the data
        response = requests.get(url)
        data = response.json()
        return data

    def convert_to_cityjson(self, data: dict):
        """Converts the data from the request to CityJSON format."""
        cityjson = data['metadata']
        cityjsonfeature = data['feature']
        cityjson['CityObjects'] = cityjsonfeature['CityObjects']
        cityjson['vertices'] = cityjsonfeature['vertices']
        return cityjson

    def save(self, cityjson: dict, id: str):
        """Saves the CityJSON data to a file."""
        with open(f'{self.input_folder}/{id}.city.json', 'w') as file:
            json.dump(cityjson, file)

    def check_input_folder(self):
        """Checks if the input folder exists and is a folder, if not it will be created. If it exists and is not empty, a warning is printed."""
        if not os.path.exists(self.input_folder):
            os.makedirs(self.input_folder)
        elif not os.path.isdir(self.input_folder):
            raise ValueError("The input_folder should be a directory, not a file.")
        elif len(os.listdir(self.input_folder)) > 1:
            print("WARNING: the input folder is not empty, the data will be overwritten if it already exists")
        elif len(os.listdir(self.input_folder)) == 1 and os.path.exists(f"{self.input_folder}/.gitkeep") == False:
            print("WARNING: the input folder is not empty, the data will be overwritten if it already exists")
        # else: the folder is empty, so no warning is needed

    def request_list(self, all_ids: list):
        """Requests and save the data in cityjson format for all the ids in the list."""

        self.check_input_folder()

        for id in all_ids:
            try:
                if id.startswith('NL.IMBAG.Pand.') == False:
                    id = f'NL.IMBAG.Pand.{id}'

                data = self.request_id(id)
                cityjson = self.convert_to_cityjson(data)
                self.save(cityjson, id)

            except Exception as e:
                print(f'error with id {id}:\n {e}')
                continue

if __name__ == "__main__":
    import timeit

    all_ids = ["NL.IMBAG.Pand.0202100000238878", "NL.IMBAG.Pand.0202100000206918"]
    c = collection()
    print(timeit.timeit(lambda: c.request_list(all_ids), number=1))