import dotenv
import json
import os
import time

import flatgeobuf as fgb
import gzip
import requests
import sqlite3
from tqdm import tqdm

from collection.utility import get_roof_attributes


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


def download_3dbag_cj(data_folder, tilefile, error_3dbag=False, verbose=False):
    tiles_folder = os.path.join(data_folder, "tiles")
    if not os.path.exists(tiles_folder):
        os.mkdir(tiles_folder)

    tile_cjz_response = requests.get(f"{tilefile}")
    if verbose:
        print(tile_cjz_response.status_code)

    # Decompress response and load it into a zip file
    if not error_3dbag:
        tile_cj = json.loads(gzip.decompress(tile_cjz_response.content))
    else:
        tile_cj = json.loads(tile_cjz_response.content)

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


def main_tilessqlite(data_folder, file_path_bag, exists_ok=False, error_3dbag=False, verbose=False):
    if not os.path.exists(os.path.join(data_folder, "tiles")):
        os.mkdir(os.path.join(data_folder, "tiles"))

    conn_bag = connect_gpkg(file_path_bag)
    cursor_bag = conn_bag.cursor()

    try:
        tiles = get_tiles(data_folder)
        bag_columns = ["identificatie", "gebruiksdoel", "aantal_verblijfsobjecten", "oppervlakte_min", "oppervlakte_max"]

        for tile in tqdm(tiles['features'], desc="Tiles", mininterval=10):
            tilename = tile['properties']['cj_download'].split("/")[-1][:-3]
            # Als de JSON file al bestaat en exists_ok (False: vervang, True: sla over)
            if not exists_ok and os.path.exists(os.path.join(data_folder, "tiles", tilename)):
                if verbose:
                    print(f"CityJSON file '{tilename}' already exists.")
                continue

            tile_cj_url = tile['properties']['cj_download']
            if error_3dbag:
                tile_cj_url = tile['properties']['cj_download'].replace("https://data.3dbag.nl", "https://3dbag.fsn1.your-objectstorage.com")
            tile_3dbag_cj = download_3dbag_cj(data_folder, tile_cj_url, error_3dbag, verbose=verbose)

            for pand_id in tile_3dbag_cj['CityObjects'].keys():
                # If child (or buildingpart)
                if '-' in pand_id:
                    continue
                if tile_3dbag_cj['CityObjects'][pand_id]['attributes']['status'] != "Pand in gebruik":
                    continue    # TODO: rm from cj

                bag_data = query_bag(cursor_bag, pand_id[14:], columns=bag_columns)
                if bag_data is None:
                    print(f"No pand with id '{pand_id}' in BAG.gpkg")
                    continue
                bag_attributes = {bag_columns[i]:bag_data[i] for i in range(len(bag_columns)) if bag_columns[i] != 'identificatie'}
                
                if bag_attributes.get('gebruiksdoel') != "woonfunctie":
                    continue

                parent_cj = tile_3dbag_cj.get("CityObjects").get(pand_id)
                children = parent_cj.get("children")
                childs_cj = {child: tile_3dbag_cj.get("CityObjects").get(child) for child in children}
                roof_attributes = get_roof_attributes(bagid=pand_id, 
                                                  parent_cj=parent_cj, 
                                                  childs_cj=childs_cj, 
                                                  vertices=tile_3dbag_cj.get("vertices"), 
                                                  cj_scale=tile_3dbag_cj.get("transform").get("scale"),
                                                  verbose=verbose)
                attributes = {**bag_attributes, **roof_attributes}
                parent_cj['attributes'].update(attributes)

            # Save file in data_folder/tiles/{tilenumber}.city.json
            with open(os.path.join(data_folder, "tiles", tilename), 'w') as outfile:
                json.dump(tile_3dbag_cj, outfile)

            if verbose:
                tile_name = tile['properties']['cj_download'].split("/")[-1].split(".")[0]
                print(f"Completed tile: {tile_name}")
    
    finally:
        if conn_bag:
            cursor_bag.close()
            conn_bag.close()

    return


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
    
    finally:
        cursor_bag.close()
        conn_bag.close()

        cursor_3dbag.close()
        conn_3dbag.close()


if __name__=="__main__":
    dotenv.load_dotenv()
    gpkg_bag_light = os.environ.get('gpkg_bag_light')
    data_folder = os.environ.get('data_folder')

    # main_api() # maar dubbele gebouwen? mss niet
    main_tilessqlite(data_folder, gpkg_bag_light, exists_ok=False, error_3dbag=True, verbose=False)
    # main_itsqlite(gpkg_list[0], gpkg_list[1], verbose=True) # Hoe geometry uit gpkg
