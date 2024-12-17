import aiohttp
import re
import geopandas as gpd
import asyncio
from shapely.geometry import mapping


async def get_neighborhood(hoodid, session):
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
        coordinatestring += f'{i[0]} {i[1]},'
    coordinatestring = coordinatestring[:-1]
    return coordinatestring

def create_xml_query(neighborhood):
    geometry = neighborhood['geometry'].values[0]
    coordinatestring = create_coor_str(geometry)

    query = f"""<?xml version="1.0" encoding="utf-8"?>
    <GetFeature xmlns="http://www.opengis.net/wfs/2.0" xmlns:gml="http://www.opengis.net/gml/3.2" service="WFS" version="2.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schemas.opengis.net/wfs/2.0/wfs.xsd http://schemas.opengis.net/wfs/2.0.0/WFS-transaction.xsd">
        <Query typeNames="pand" xmlns:bag="http://bag.geonovum.nl">
            <fes:Filter xmlns:fes="http://www.opengis.net/fes/2.0">
                <fes:Intersects>
                    <fes:ValueReference>geometrie</fes:ValueReference>
                    <gml:Polygon gml:id="filter" srsName="urn:ogc:def:crs:EPSG::28992">
                        <gml:exterior>
                            <gml:LinearRing>
                                <gml:posList srsDimension="2">{coordinatestring}</gml:posList>
                            </gml:LinearRing>
                        </gml:exterior>
                    </gml:Polygon>
                </fes:Intersects>
            </fes:Filter>
        </Query>
    </GetFeature>"""
    return query


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
        print(f'No (valid) hood id provided: {hoodid}, using default value BU03560803: Hoven-West te Nieuwegein')
        hoodid = 'BU03560803'
        verbose = True

    neighborhood = await get_neighborhood(hoodid, session)
    if len(neighborhood['geometry'].values) == 0:
        print(f'No neighbourhood found with id "{hoodid}"')
        if close_session:
            await session.close()
        return []

    data = create_xml_query(neighborhood)
    url = "https://service.pdok.nl/lv/bag/wfs/v2_0"
    headers = {"Content-Type": "application/xml"}

    r = await session.post(url, headers=headers, data=data)
    r.raise_for_status()
    building_data = await r.text()
    ids = re.findall("(?<=<bag:identificatie>)\d{16}(?=<\/bag:identificatie>)", building_data)
    
    # finish the function by: printing the number of buildings found, closing the session if needed and returning the ids
    if verbose:
        print(f'Found {len(ids)} buildings in the neighbourhood with name "{neighborhood["buurtnaam"].values[0]}" and id "{hoodid}"')
    
    if close_session: # close the session if it was created in this function
        await session.close()
    
    return ids

if __name__ == "__main__":
    asyncio.run(hoodcollector('BU03560803'))