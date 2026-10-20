import json
import geopandas as gpd
import geojson
import aiohttp
import requests



def _roofmetrics(bagid=None, bbox=None, verbose=False):
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
    
    # params=dict(
    #         service="WFS",
    #         version="2.0.0",
    #         request="GetFeature",
    #         typeName="BAG3D:lod22",
    #         outputFormat="json",
    #         BBOX=bbox,
    #     )
    
    params = dict(
    service="WFS",
    version="2.0.0",
    request="GetFeature", 
    # typeName="BAG3D:lod22",
    typeName="BAG3D:tiles",
    outputFormat="json",
    BBOX="204085.226,325278.304,204209.867,325402.946",  # Rounded + CRS
    )

    url = "https://data.3dbag.nl/api/BAG3D/wfs"
    r = requests.get(url, params=params)
    # r.raise_for_status() #TODO: Maybe?
    r = r.content
    

    # Create GeoDataFrame from geojson and set coordinate reference system
    json_content = geojson.loads(r)
    if json_content.get('bbox') is None:
        return None
    data = gpd.GeoDataFrame.from_features(json_content, crs="EPSG:7415")
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



def convert_to_cityjson(data: dict, bag_attributes: dict, roof_attributes: dict, id: str):
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


def get_bbox(data: dict, decimals=5):
    """Get the bounding box of a building from the 3D-BAG API."""
    translate = data['transform']['translate']
    centroid_x, centroid_y = translate[0], translate[1]
    # perimeter = 10
    perimeter = translate[2]
    return f"""{round(centroid_x-perimeter, decimals)},
               {round(centroid_y-perimeter, decimals)},
               {round(centroid_x+perimeter, decimals)},
               {round(centroid_y+perimeter, decimals)}"""

def get_roof_attributes(id: str, data: dict):
    bbox = get_bbox(data)
    return _roofmetrics(id, bbox)

def make_cityjson(data_3dbag, data_bag, pand_id):
    roof_attributes = get_roof_attributes(pand_id, data_3dbag)
    cityjson = convert_to_cityjson(data_3dbag, data_bag, roof_attributes)
    return cityjson


def convert_polygon_to_cityjson():

    gdf = gpd.read_file(r'/home/nisfernos/Downloads/9-328-624.gpkg')

    gdf_polys = gdf[gdf.geometry.geom_type == "MultiPolygon"]
    print(gdf_polys)
    geom_type, coords, offsets = shapely.to_ragged_array(gdf_polys.geometry, include_z=True)
    print(geom_type)
    print(coords)
    print(offsets)


    cityjson = CITYJSON_TEMPLATE.copy()
    cityjson["vertices"] = coords.tolist()

    rings, polys, geoms = offsets

    for i, (idx, feature) in enumerate(gdf_polys.iterrows()):
        cityobject_id = str(idx)

        boundaries = []
        for poly in range(geoms[i], geoms[i+1] if (i + 1) < len(geoms) else len(geoms) - 1):
            ring_list = []
            for ring in range(polys[poly], polys[poly + 1] if (poly + 1) < len(polys) else len(rings) - 1):
                ring_list.append(
                    [j for j in range(rings[ring], rings[ring + 1] if (ring + 1) < len(rings) else len(coords) - 1)][::-1]
                )
            boundaries.append(ring_list)

        attributes = {
            k: str(feature[k])
            for k in feature.keys()
            if k != "geometry"
        }

        cityjson["CityObjects"][cityobject_id] = {
            "type": "Building",
            "attributes": attributes,
            "geometry": [
                {
                    "type": "MultiSurface",
                    "boundaries": boundaries
                }
            ]
        }


# if __name__ == "__main__":
#     pass
