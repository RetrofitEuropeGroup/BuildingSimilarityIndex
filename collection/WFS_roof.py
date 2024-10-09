import geopandas as gpd
import geojson
import aiohttp

async def roofmetrics(bagid=None, bbox=None, verbose=False, session=None):
    """
    Function to get roof metrics from 3dbag API
    :param bagid: BAG id of the building
    :param bbox: Bounding box of the building in the form of 'minx,miny,maxx,maxy'
    :return: dictionary with roof metrics
    """
    if isinstance(bagid, str) and bagid.startswith("NL.IMBAG.Pand.") == False:
        bagid = "NL.IMBAG.Pand." + bagid
    if not bagid or not bbox:
        bagid = 'NL.IMBAG.Pand.0321100000002220'
        bbox = "138467.57,449429.21,138521.21,449474.03"
        print('No BAG id or bounding box provided, using default values')
        verbose = True
    
    params=dict(
            service="WFS",
            version="2.0.0",
            request="GetFeature",
            typeName="BAG3D:lod22",
            outputFormat="json",
            BBOX=bbox,
        )
    
    url = "https://data.3dbag.nl/api/BAG3D/wfs"
    if session is None:
        async with aiohttp.ClientSession() as session:
            r = await session.get(url, params=params)
            r.raise_for_status()
            r = await r.text()
    else:
        r = await session.get(url, params=params)
        r.raise_for_status()
        r = await r.text()
    

    # Create GeoDataFrame from geojson and set coordinate reference system
    data = gpd.GeoDataFrame.from_features(geojson.loads(r), crs="EPSG:7415")
    data = data[data['identificatie'] == bagid]

    # calculate max height of the roofparts
    maxheight = data['b3_h_max'].max()
    # filter roofparts that are within 1m of the max height
    data['main_roof'] = data['b3_h_max'] > maxheight - 1
    # for all currently false roofparts, make true if geometry has area larger than 50m2
    data['main_roof'] = data['main_roof'] | (data['geometry'].area > 50)

    # TODO: horizontal roof angle not available in 3dbag data, so we cannot use 3d roof area yet. Using a proxy based on b3_opp_grond, b3_opp_dak_schuin and b3_opp_dak_plat
    # data['3d_area'] = data['geometry'].area / math.cos(data['b3_hellingshoek'])

    # calculate metrics:
    mr_no_parts = data['main_roof'].sum()
    mr_area = data[data['main_roof']]['geometry'].area.sum()

    other_no_parts = len(data) - mr_no_parts
    other_area = data[~data['main_roof']]['geometry'].area.sum()

    if mr_no_parts + other_no_parts == 0:
        part_ratio = None
    else:
        part_ratio = mr_no_parts / (mr_no_parts + other_no_parts)
    if mr_area + other_area == 0:
        area_ratio = None
    else:
        area_ratio = mr_area / (mr_area + other_area)

    if verbose:
        print(f'Building: {bagid}')
        print(f'Main roof: {mr_no_parts} parts with a total area of {mr_area:.2f} m2')
        print(f'Other roof: {other_no_parts} parts with a total area of {other_area:.2f} m2')
        print(f'Part ratio: {part_ratio:.2f}')
        print(f'Area ratio: {area_ratio:.2f}')

    return dict(
        main_roof_parts=int(mr_no_parts), # convert to int for json serialisation
        main_roof_area=mr_area,
        other_roof_parts=int(other_no_parts),
        other_roof_area=other_area,
        part_ratio=part_ratio,
        area_ratio=area_ratio
    )