import urllib.request
import json


def request_id(id: str):
    """Uses the 3dbag API to request data for a specific id and writes it to a jsonl file."""
    url = f"https://api.3dbag.nl/collections/pand/items/{id}"

    # Request the data
    with urllib.request.urlopen(url) as response:
        j = json.loads(response.read().decode('utf-8'))

        # write the data to a jsonl file
        with open(f"collection/data/{id}.city.jsonl", "w") as my_file:
            my_file.write(json.dumps(j["metadata"]) + "\n")
            if "feature" in j:
                my_file.write(json.dumps(j["feature"]) + "\n")
            if "features" in j:
                for f in j["features"]:
                    my_file.write(json.dumps(f) + "\n")

def get_data(all_ids: list):

    # TODO: make this asynchronous
    for id in all_ids:
        if id.startswith('NL.IMBAG.Pand.') == False:
            id = f'NL.IMBAG.Pand.{id}'
        request_id(id)

if __name__ == "__main__":
    import timeit

    all_ids = ["NL.IMBAG.Pand.0202100000238878", "NL.IMBAG.Pand.0202100000206918"]
    print(timeit.timeit(lambda: get_data(all_ids), number=1))