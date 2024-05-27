import requests
import json

# TODO: make a class
# TODO: empty input folder before running

class collection():
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
        with open(f'./collection/input/{id}.city.json', 'w') as file:
            json.dump(cityjson, file)

    def request_list(self, all_ids: list):
        # TODO: make this asynchronous

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