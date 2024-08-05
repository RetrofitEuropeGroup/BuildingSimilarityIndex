import os
import requests

url = 'https://api.bag.kadaster.nl/lvbag/individuelebevragingen/v2/adressenuitgebreid'
key = os.environ.get('BAG_API_KEY')

response = requests.get(url, params={'pandIdentificatie':'0153100000203775'}, headers={'X-Api-Key':key, 'Accept-Crs': 'EPSG:28992'})
print(response.content.decode())

