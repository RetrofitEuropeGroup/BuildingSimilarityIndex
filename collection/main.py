import requests
import json


def request_id(id: str):
    """Uses the 3dbag API to request data for a specific id and writes it to a jsonl file."""
    url = f"https://api.3dbag.nl/collections/pand/items/{id}"

    # Request the data
    response = requests.get(url)
    data = response.json()
    return data
    
def convert_to_cityjson(data: dict):
    """Converts the data from the request to CityJSON format."""
    cityjson = data['metadata']
    cityjsonfeature = data['feature']
    cityjson['CityObjects'] = cityjsonfeature['CityObjects']
    cityjson['vertices'] = cityjsonfeature['vertices']
    return cityjson

def save(cityjson: dict, id: str):
    """Saves the CityJSON data to a file."""
    with open(f'./collection/input/{id}.city.json', 'w') as file:
        json.dump(cityjson, file)

def main(all_ids: list):
    # TODO: make this asynchronous
    for id in all_ids:
        try:
            if id.startswith('NL.IMBAG.Pand.') == False:
                id = f'NL.IMBAG.Pand.{id}'
            
            data = request_id(id)
            cityjson = convert_to_cityjson(data)
            save(cityjson, id)

        except Exception as e:
            print(f'error with id {id}:\n {e}')
            continue

if __name__ == "__main__":
    import timeit

    all_ids = ["NL.IMBAG.Pand.0202100000238878", "NL.IMBAG.Pand.0202100000206918"]
    print(timeit.timeit(lambda: main(all_ids), number=1))