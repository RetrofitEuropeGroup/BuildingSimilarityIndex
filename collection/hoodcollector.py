import aiohttp
import pandas as pd
import geopandas as gpd
import asyncio
from shapely.geometry import mapping


async def get_neighborhood(hoodid, session, verbose=False):
    # set the parameters for the request
    hoodparams = dict(
        SERVICE="WFS",
        VERSION="2.0.0",
        request="GetFeature",
        typeNames="wijkenbuurten:buurten",
        srsname="urn:ogc:def:crs:EPSG::28992",
        FILTER=f'<?xml version="1.0" encoding="UTF-8"?><fes:Filter xmlns:fes="http://www.opengis.net/fes/2.0" '
               f'xmlns:gml="http://www.opengis.net/gml/3.2"><fes:PropertyIsEqualTo><fes:ValueReference>buurtcode</fes'
               f':ValueReference><fes:Literal>{hoodid}</fes:Literal></fes:PropertyIsEqualTo></fes:Filter>',
        outputFormat="geojson",
    )

    # make the request to retrieve the neighbourhood data (geometry, name, etc.)
    r = await session.get("https://service.pdok.nl/cbs/wijkenbuurten/2023/wfs/v1_0", params=hoodparams)
    r.raise_for_status()
    r = await r.text()

    neighborhood = gpd.read_file(r)
    return neighborhood

def create_coor_str(geometry):
    coordinates = mapping(geometry.boundary.geoms[0])['coordinates']
    coordinatestring = ''
    for i in coordinates:
        coordinatestring += f'{i[0]}+{i[1]},'
    coordinatestring = coordinatestring[:-1]
    return coordinatestring

async def hoodcollector(hoodid=None, session=None, verbose=False):
    """
    Function to get all BAG ids of buildings in a neighbourhood
    :param hoodid: CBS id of the neighbourhood, see https://www.pdok.nl/ogc-webservices/-/article/cbs-wijken-en-buurten
    :return: list of BAG ids
    """
    # Create a session if none is provided
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True
    else:
        close_session = False

    # Check if the hoodid is given and valid, if not use a default value
    if not hoodid or not isinstance(hoodid, str) or not hoodid.startswith("BU"):
        hoodid = 'BU03560803'
        print(f'No (valid) hood id provided, using default value {hoodid}: Hoven-West te Nieuwegein')
        verbose = True

    neighborhood = await get_neighborhood(hoodid, session, verbose)
    
    # convert geometry to string with the coordinates of the boundary of the neighbourhood
    geometry = neighborhood['geometry'].values[0]
    coordinatestring = create_coor_str(geometry)

    # set the parameters for the request loop
    count = 1000  # Number of records to fetch per request
    start_index = 0  # Starting point for fetching records
    ids = []

    while True: # loop until we don't get any more buildings
        url = (f"https://service.pdok.nl/lv/bag/wfs/v2_0?request=GetFeature&service=WFS&version=2.0.0&typeName=bag:pand"
               f"&SRSNAME=EPSG:28992&count={count}&startindex={start_index}&sortby=bag:identificatie&outputFormat"
               f"=geojson&Filter=%3CFilter%3E%3CIntersects%3E%3CPropertyName%3EGeometry"
               f"%3C/PropertyName%3E%3Cgml:Polygon%3E%3Cgml:outerBoundaryIs%3E%3Cgml:LinearRing%3E%3Cgml:coordinates%3E"
               f"{coordinatestring}%3C/gml:coordinates%3E+%3C/gml:LinearRing%3E%3C/gml:outerBoundaryIs%3E%3C/gml:Polygon"
               f"%3E%3C/Intersects%3E%3C/Filter%3E")

        # Perform the request
        r = await session.get(url)
        r.raise_for_status()
        building_data = await r.text()

        new_buildings = gpd.read_file(building_data)
        if len(new_buildings) > 0:
            ids.extend(new_buildings['identificatie'].tolist())
            start_index += count
        else:
            break
    
    # finish the function by: printing the number of buildings found, closing the session if needed and returning the ids
    if verbose:
        print(f'Found {len(ids)} buildings in the neighbourhood with name "{neighborhood["buurtnaam"].values[0]}" and id "{hoodid}"')
    
    if close_session: # close the session if it was created in this function
        await session.close()
    
    return ids

if __name__ == "__main__":
    print(asyncio.run(hoodids('BU03560803', verbose=True)))
