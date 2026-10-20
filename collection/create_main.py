from fiona import listlayers
import flatgeobuf as fgb
import geopandas as gpd
import gzip
import json
import os
import requests
import sqlite3
import time

from collection.utility import make_cityjson, get_roof_attributes
from collection.WFS_roof import roofmetrics



def info_gpkg(filepath, _layer='pand'):
    gdf = gpd.read_file(filepath, layer=_layer, rows=10)
    print(f"Head: {gdf.head(10)}")
    print(f"Layers: {listlayers(filepath)}")
    print(f"Kolommen: {gdf.columns}")
    print(f"Lengte: {len(gdf)}")


def load2gdf_slices(in_fc, use_fields=None, chunk_size=1000):
    # https://stackoverflow.com/questions/69184154/geopandas-fast-read-slice-of-geopackage-format
    i = 0
    out_gdf = gpd.GeoDataFrame()

    while True:
        chunk = slice(i, i + chunk_size, 1)
        gdf_iter = gpd.read_file(in_fc, rows=chunk, columns=use_fields)

        if gdf_iter.shape[0] == 0:
            break

    return out_gdf


def get_tiles(data_folder, verbose=False):
    # Download file https://data.3dbag.nl/v20250903/tile_index.fgb 
    fgb_file = os.path.join(data_folder, "tile_index.fgb")
    if not os.path.exists(fgb_file):
        if verbose:
            print(os.path.join(data_folder, "tile_index.fgb"))
        fgb_response = requests.get("https://data.3dbag.nl/v20250903/tile_index.fgb")
        with open(fgb_file, mode="wb") as f:
            f.write(fgb_response.content)
    with open(fgb_file, "rb") as f:
        tiles = fgb.load(f)
        return(tiles)


def download_3dbag_cj(data_folder, tilefile, verbose=False):
    tiles_folder = os.path.join(data_folder, "tiles")
    if not os.path.exists(tiles_folder):
        os.mkdir(tiles_folder)

    tile_cjz_response = requests.get(f"{tilefile}")
    # Decompress response and load it into a zip file
    tile_cj = json.loads(gzip.decompress(tile_cjz_response.content))

    return tile_cj


def query_bag(cursor_bag, pand_id:str, columns):
    """Queries the BAG for specific attributes of a specific building (pand).

    Args:
        cursor_bag (sqlite.cursor): SQLite cursor
        pand_id (_type_): BAG identification number. (Without "NL.IMBAG.PAND.")
        columns (list): The list of columns for which values to return. 

    Returns:
        tuple: Requested features
    """
    if not columns:
        raise Exception("'columns' argument must contain a value.")
    column_query = f"{columns[0]}" + "".join(f", {c}" for c in columns[1:])
    query = f'''SELECT {column_query} 
                FROM pand 
                WHERE identificatie = "{pand_id}"
            '''
    cursor_bag.execute(query)
    return cursor_bag.fetchone()


def main_tilessqlite(data_folder, file_path_bag, verbose=False):
    conn_bag = connect_gpkg(file_path_bag)
    cursor_bag = conn_bag.cursor()

    try:
        tiles = get_tiles(data_folder)
        bag_columns = ["identificatie", "gebruiksdoel", "aantal_verblijfsobjecten", "oppervlakte_min", "oppervlakte_max"]

        for tile in tiles['features']:
            tile_3dbag_cj = download_3dbag_cj(data_folder, tile['properties']['cj_download'])

            for pand_id in tile_3dbag_cj['CityObjects'].keys():
                if '-' in pand_id:
                    continue
                bag_data = query_bag(cursor_bag, pand_id[14:], columns=bag_columns)
                if bag_data is None:
                    print(f"No pand with id '{pand_id}' in BAG.gpkg")
                    continue
                bag_data = {bag_columns[i]:bag_data[i] for i in range(len(bag_columns))}
                
                if bag_data['gebruiksdoel'] != "woonfunctie":
                    continue
                
                roofmetrics = get_roof_attributes(pand_id, tile_3dbag_cj)
                # if roofmetrics is not None:
                print(roofmetrics)
                break
                # cityjson = make_cityjson(tile_3dbag_cj, bag_data, pand_id)
                # print(cityjson)
                print(pand_id)
            # Combineer in .city.json import function
            # Citystats import funciton
            # Voeg toe aan feature space ez?
            # Delete file?
            if verbose:
                tile_name = tile['properties']['cj_download'].split("/")[-1].split(".")[0]
                print(f"Completed tile: {tile_name}")
                break
        return
    
    finally:
        if conn_bag:
            cursor_bag.close()
            conn_bag.close()


def connect_gpkg(file_path):
    conn = sqlite3.connect(file_path)
    conn.enable_load_extension(True)

    try:
        conn.load_extension('mod_spatialite')
        cursor = conn.cursor()
        cursor.execute("SELECT InitSpatialMetadata(1)")
        # cursor.execute("SELECT EnableGPKGMode(1)")
        # cursor.execute('''SELECT CastAutomagic()''') ?
        cursor.execute('''SELECT EnableGpkgAmphibiousMode();''') # Potentially deprecated
        cursor.close()

    except sqlite3.Error as e:
        print(f"SpatiaLite extension warning: {e}")
        print(f"Continuing will return in geometry being returned as BLOBS")

    return conn


def query_building_gpkg(cursor, id):
    query = f'''
            SELECT *, AsText(geom)
            FROM pand
            WHERE identificatie = "{id}";
            '''
    cursor.execute(query)

    return cursor.fetchone() # WHERE statement so fetchone


def main_itsqlite(file_path_bag, file_path_3dbag, verbose=False):
    conn_bag = connect_gpkg(file_path_bag)
    cursor_bag = conn_bag.cursor()

    conn_3dbag = connect_gpkg(file_path_3dbag)
    cursor_3dbag = conn_3dbag.cursor()

    try:
        ### if geom is available in lod22_3d layer
        # cursor_3dbag.execute('''SELECT *, AsText(geom) FROM lod22_3d WHERE identificatie = "NL.IMBAG.Pand.0307100000308306"''')
        # cb = cursor_3dbag.fetchone()
        # print([x[0] for x in cursor_3dbag.description])
        # print(cb)
        # return
        ###
        cursor_3dbag.execute('''SELECT *, AsText(geom) FROM pand''')
        content_3dbag = cursor_3dbag.fetchone()
        while content_3dbag:
            content_bag = query_building_gpkg(cursor_bag, content_3dbag[49].split(".")[-1])
            if not content_bag:
                print(content_3dbag[-11].split(".")[-1])
                continue

            if verbose:
                print(list(zip([x[0] for x in cursor_bag.description],
                                               content_bag)))
                print(list(zip([x[0] for x in cursor_3dbag.description],
                               content_3dbag)))
                
            content_3dbag = cursor_3dbag.fetchone()
            break #TODO: Remove
    
    finally:
        cursor_bag.close()
        conn_bag.close()

        cursor_3dbag.close()
        conn_3dbag.close()


if __name__=="__main__":
    gpkg_list = ["/home/nisfernos/data/bag_light.gpkg",
                "/home/nisfernos/data/3dbag_nl.gpkg",
                "/home/nisfernos/Downloads/9-328-624.gpkg"]
    data_folder = "/home/nisfernos/data/"

    t0 = time.time()

    # main_api() # maar dubbele gebouwen? mss niet
    main_tilessqlite(data_folder, gpkg_list[0], verbose=True) # Hoe geometry uit gpkg
    # main_itsqlite(gpkg_list[0], gpkg_list[1], verbose=True)

    t1 = time.time()

    print(f"Deze code duurt: {(t1-t0):.2f} seconden.")

"""
    print(load2gdf_slices(gpkg_list[1], ))

    import fiona

    No need to pass "layer='etc'" if there's only one layer
    with fiona.open(gpkg_list[0], layer='pand') as layer:
        for feature in layer:
            print(feature.geometry["coordinates"])
            print(feature.identificatie)
            break

    #### API call
        headers = {'X-Api-Key':'l7de97bcf7acd3447ab2caf915655422db', 'Accept-Crs': 'EPSG:28992'}
        url = 'https://api.bag.kadaster.nl/lvbag/individuelebevragingen/v2/adressenuitgebreid'
        params = {'pandIdentificatie':'1667100000001458'}    
        params = {'pandIdentificatie':'0003100000117485'}
    
        print(requests.get(url, params=params, headers=headers).json().get('_embedded', {}).get('adressen'))
    
        #### test gpkg head
        print(load2gdf_slices(gpkg_list[1], chunk_size=10))

[(1, b'GP\x00\x05\xf7\x1c\x00\x00\x00\x00\x00\x00p\xcb\x00A\x00\x00\x00\x00\xa0\xcb\x00A\x00\x00\x00\x00\xc0\x9c\x16A\x00\x00\x00\x00\xd8\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\xeb\x03\x00\x00\x01\x00\x00\x00#\x00\x00\x00\x00\x00\x00\x00\xa0\xcb\x00A\x00\x00\x00\x00\xcc\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0\x9f\xcb\x00A\x00\x00\x00@\xce\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x9e\xcb\x00A\x00\x00\x00`\xd0\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x9c\xcb\x00A\x00\x00\x00`\xd2\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x99\xcb\x00A\x00\x00\x00 \xd4\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x96\xcb\x00A\x00\x00\x00\xa0\xd5\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x92\xcb\x00A\x00\x00\x00\xc0\xd6\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0\x8e\xcb\x00A\x00\x00\x00\x80\xd7\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x8a\xcb\x00A\x00\x00\x00\x00\xd8\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x85\xcb\x00A\x00\x00\x00\x00\xd8\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x81\xcb\x00A\x00\x00\x00\x80\xd7\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@}\xcb\x00A\x00\x00\x00\xc0\xd6\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80y\xcb\x00A\x00\x00\x00\xa0\xd5\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@v\xcb\x00A\x00\x00\x00 \xd4\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0s\xcb\x00A\x00\x00\x00`\xd2\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0q\xcb\x00A\x00\x00\x00`\xd0\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`p\xcb\x00A\x00\x00\x00@\xce\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00p\xcb\x00A\x00\x00\x00\x00\xcc\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`p\xcb\x00A\x00\x00\x00\xc0\xc9\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0q\xcb\x00A\x00\x00\x00\xa0\xc7\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0s\xcb\x00A\x00\x00\x00\xc0\xc5\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@v\xcb\x00A\x00\x00\x00\xe0\xc3\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80y\xcb\x00A\x00\x00\x00`\xc2\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@}\xcb\x00A\x00\x00\x00@\xc1\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x81\xcb\x00A\x00\x00\x00\x80\xc0\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x85\xcb\x00A\x00\x00\x00\x00\xc0\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x8a\xcb\x00A\x00\x00\x00\x00\xc0\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0\x8e\xcb\x00A\x00\x00\x00\x80\xc0\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x92\xcb\x00A\x00\x00\x00@\xc1\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x96\xcb\x00A\x00\x00\x00`\xc2\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x99\xcb\x00A\x00\x00\x00\xe0\xc3\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x9c\xcb\x00A\x00\x00\x00\xc0\xc5\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x9e\xcb\x00A\x00\x00\x00\xa0\xc7\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0\x9f\xcb\x00A\x00\x00\x00\xc0\xc9\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0\xcb\x00A\x00\x00\x00\x00\xcc\x9c\x16A\x00\x00\x00\x00\x00\x00\x00\x00', 
  0.0, None, 'slanted', 'standard', 27.621999740600586, 128.3637237548828, 0, 0, 1, 0, 0, 1, 2, 0.0, 0.06930693238973618, 0.0, 0.0, 0.0, 0.0, 
  1786.6099853515625, 0.0, 11.84000015258789, 9.270000457763672, 0.0, 25.544553756713867, 56.76595687866211, 90.57425689697266, 'ahn5', 2024.0, 0, 
  'PREFERRED_AND_LATEST', 48.188045501708984, 48.188045501708984, 47.32743835449219, 25.0, '[]', '[]', '[]', 929.1686401367188, 929.1686401367188, 927.2742309570312, 
  '2018-12-13', '2018-12-13', '18.12136', None, None, 0, 'NL.IMBAG.Pand.1667100000012089', 2015.0, 'Pand in gebruik', None, None, None, None, 
  '2018-12-13T08:02:22Z', '2018-12-13T08:30:24Z', 1.0, 'POLYGON Z((137588 370483 0, 137587.953125 370483.5625 0, 137587.796875 370484.09375 0, '
  '137587.546875 370484.59375 0, 137587.21875 370485.03125 0, 137586.8125 370485.40625 0, 137586.34375 370485.6875 0, 137585.828125 370485.875 0, '
  '137585.28125 370486 0, 137584.71875 370486 0, 137584.171875 370485.875 0, 137583.65625 370485.6875 0, 137583.1875 370485.40625 0, 137582.78125 370485.03125 0, '
  '137582.453125 370484.59375 0, 137582.203125 370484.09375 0, 137582.046875 370483.5625 0, 137582 370483 0, 137582.046875 370482.4375 0, '
  '137582.203125 370481.90625 0, 137582.453125 370481.4375 0, 137582.78125 370480.96875 0, 137583.1875 370480.59375 0, 137583.65625 370480.3125 0, '
  '137584.171875 370480.125 0, 137584.71875 370480 0, 137585.28125 370480 0, 137585.828125 370480.125 0, 137586.34375 370480.3125 0, 137586.8125 370480.59375 0, '
  '137587.21875 370480.96875 0, 137587.546875 370481.4375 0, 137587.796875 370481.90625 0, 137587.953125 370482.4375 0, 137588 370483 0))')]
"""