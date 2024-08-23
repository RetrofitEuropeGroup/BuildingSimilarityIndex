import aiohttp
import pandas as pd
import geopandas as gpd
import asyncio
from shapely.geometry import mapping


async def hoodids(hoodid=None, verbose=False, session=None):
    """
    Function to get all BAG ids of buildings in a neighbourhood
    :param hoodid: CBS id of the neighbourhood, see https://www.pdok.nl/ogc-webservices/-/article/cbs-wijken-en-buurten
    :return: list of BAG ids
    """

    if not hoodid or not isinstance(hoodid, str) or not hoodid.startswith("BU"):
        hoodid = 'BU03560803'
        print(f'No hood id provided, using default value {hoodid}: Hoven-West te Nieuwegein')
        verbose = True

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

    url = "https://service.pdok.nl/cbs/wijkenbuurten/2023/wfs/v1_0"
    if session is None:
        async with aiohttp.ClientSession() as session:
            r = await session.get(url, params=hoodparams)
            r.raise_for_status()
            r = await r.text()

    else:
        r = await session.get(url, params=hoodparams)
        r.raise_for_status()
        r = await r.text()

    neighborhood = gpd.read_file(r)
    geometry = neighborhood['geometry'].values[0]

    # convert geometry to string
    coordinates = mapping(geometry.boundary.geoms[0])['coordinates']
    coordinatestring = ''
    for i in coordinates:
        coordinatestring += f'{i[0]}+{i[1]},'
    coordinatestring = coordinatestring[:-1]

    if verbose:
        print(f'Found neighbourhood with id {hoodid} and name {neighborhood["buurtnaam"].values[0]}')

    count = 1000  # Number of records to fetch per request
    start_index = 0  # Starting point for fetching records
    buildings = gpd.GeoDataFrame()

    while True:
        url = (f"https://service.pdok.nl/lv/bag/wfs/v2_0?request=GetFeature&service=WFS&version=2.0.0&typeName=bag:pand"
               f"&SRSNAME=EPSG:28992&count=1000&startindex={start_index}&sortby=bag:identificatie&outputFormat"
               f"=geojson&Filter=%3CFilter%3E%3CIntersects%3E%3CPropertyName%3EGeometry"
               f"%3C/PropertyName%3E%3Cgml:Polygon%3E%3Cgml:outerBoundaryIs%3E%3Cgml:LinearRing%3E%3Cgml:coordinates%3E"
               f"{coordinatestring}%3C/gml:coordinates%3E+%3C/gml:LinearRing%3E%3C/gml:outerBoundaryIs%3E%3C/gml:Polygon"
               f"%3E%3C/Intersects%3E%3C/Filter%3E")

        if verbose:
            print(f'Fetching buildings in the neighbourhood with id {hoodid} starting from index {start_index} \n {url}')

        # Perform the request
        session = None
        if session is None:
            async with aiohttp.ClientSession() as session:
                r = await session.get(url)
                r.raise_for_status()
                r = await r.text()
                new_buildings = gpd.read_file(r)
                if len(new_buildings) == 0:
                    break
                else:
                    buildings = pd.concat([buildings, new_buildings], ignore_index=True)
                    start_index += 1000

    if verbose:
        print(f'Found {len(buildings)} buildings in the neighbourhood with id {hoodid}')

    return buildings['identificatie'].tolist()


print(asyncio.run(hoodids('BU03440512', verbose=True)))
