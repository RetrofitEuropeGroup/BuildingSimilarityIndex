import dotenv
import json
import os

import flatgeobuf as fgb
import gzip
import requests
import sqlite3
from tqdm import tqdm

from collection.utility import get_roof_attributes


def get_tiles(data_folder: str, exists_ok=False, verbose=False):
    """Download tile index file if it not exists and load it in.

    Args:
        data_folder (str): Location where the file could be saved.
        exists_ok (bool, optional): Overwrite a possible already existing file. Defaults to False.
        verbose (bool, optional): Verbose. Defaults to False.

    Returns:
        FeatureCollection: ~dict~ of all tiles in the 3DBAG
    """
    # Download file https://data.3dbag.nl/v20250903/tile_index.fgb 
    fgb_file = os.path.join(data_folder, "tile_index.fgb")

    if not os.path.exists(fgb_file) or exists_ok:        
        fgb_response = requests.get("https://data.3dbag.nl/v20250903/tile_index.fgb")
        with open(fgb_file, mode="wb") as f:
            f.write(fgb_response.content)
        if verbose:
            print(f"Saved tile index to {os.path.join(data_folder, 'tile_index.fgb')}")
    
    with open(fgb_file, "rb") as f:
        tiles = fgb.load(f)
        
        return tiles


def download_3dbag_cj(data_folder: str, tile_cj_url: str, error_3dbag=False, verbose=False) -> dict:
    """Download the CityJSON file of a specific tile from the 3DBAG.

    Args:
        data_folder (str): Path to where the CityJSON files with additional information will be saved.
        tile_cj_url (str): url to the CityJSON file
        error_3dbag (bool, optional): Whether to use the backup link, if the 3DBAG is having issues. Defaults to False.
        verbose (bool, optional): Verbose. Defaults to False.

    Returns:
        dict: CityJSON of the specific tile.
    """
    tiles_folder = os.path.join(data_folder, "tiles")
    if not os.path.exists(tiles_folder):
        os.mkdir(tiles_folder)

    if error_3dbag:
        tile_cj_url = tile_cj_url.replace("https://data.3dbag.nl", "https://3dbag.fsn1.your-objectstorage.com")

    tile_cjz_response = requests.get(f"{tile_cj_url}")
    if verbose:
        print(tile_cjz_response.status_code)

    # Decompress response and load it into a zip file
    if not error_3dbag:
        tile_cj = json.loads(gzip.decompress(tile_cjz_response.content))
    else:
        tile_cj = json.loads(tile_cjz_response.content)

    return tile_cj


def batch_query_bag(cursor_bag, pand_ids: list, columns: list):
    """Queries the BAG for specific attributes of a specific building (pand).

    Args:
        cursor_bag (sqlite.cursor): SQLite cursor
        pand_ids (list): BAG identification number. (Without "NL.IMBAG.PAND.")
        columns (list): The list of columns for which values to return. 

    Returns:
        tuple: Requested features
    """
    if not columns or 'identificatie' not in columns:
        raise Exception("'columns' argument must contain at least one value and 'identificatie'.")

    column_query = ", ".join(columns)
    panden_query = "(" + ",".join(f"'{pid}'" for pid in pand_ids) + ")"
    
    query = f'''SELECT {column_query} 
                FROM pand 
                WHERE identificatie IN {panden_query};
            '''
    
    cursor_bag.execute(query)
    query_result = {}
    for row in cursor_bag.fetchall():
        query_result[row[columns.index('identificatie')]] = {columns[i]: row[i] for i in range(len(columns)) if columns[i] != 'identificatie'}

    return query_result


def connect_gpkg(file_path: str):
    """Connect to a Geopackage with SpatiaLite-enabled SQLite3.

    Args:
        file_path (str): File path to the geopackage

    Returns:
        Connection: Database (or geopackage) connection
    """
    conn = sqlite3.connect(file_path)
    conn.enable_load_extension(True)

    try:
        conn.load_extension('mod_spatialite')
        cursor = conn.cursor()
        cursor.execute('''SELECT InitSpatialMetadata(1)''')
        # cursor.execute("SELECT EnableGPKGMode(1)")
        # cursor.execute('''SELECT CastAutomagic()''') ?
        cursor.execute('''SELECT EnableGpkgAmphibiousMode();''') # Potentially deprecated
        cursor.close()

    except sqlite3.Error as e:
        print(f"SpatiaLite extension warning: {e}")
        print(f"Continuing will return in geometry being returned as BLOBS")

    return conn


def remove_building(tile_cj, pand_id) -> None:
    """Remove pand from CityJSON tile file.

    Args:
        tile_cj (dict): CityJSON tile file
        pand_id (str): Pand ID in form "NL.IMBAG.Pand.<number>
    """
    children = tile_cj['CityObjects'][pand_id].get('children')
    pand_ids = [pand_id] + children
    for id in pand_ids:
        del tile_cj['CityObjects'][id]


def main_tilessqlite(data_folder, file_path_bag, exists_ok=False, error_3dbag=False, verbose=False):
    os.makedirs(os.path.join(data_folder, "tiles"), exist_ok=True)

    conn_bag = connect_gpkg(file_path_bag)
    cursor_bag = conn_bag.cursor()

    try:
        tiles = get_tiles(data_folder)
        bag_columns = ["identificatie", "gebruiksdoel", "aantal_verblijfsobjecten", "oppervlakte_min", "oppervlakte_max"]

        for tile in tqdm(tiles['features'], desc="Tiles"):
            tilename = tile['properties']['cj_download'].split("/")[-1][:-3]
            # Als de JSON file al bestaat en exists_ok (False: vervang, True: sla over)
            if not exists_ok and os.path.exists(os.path.join(data_folder, "tiles", tilename)):
                if verbose:
                    print(f"CityJSON file '{tilename}' already exists.")
                continue

            tile_cj_url = tile['properties']['cj_download']
            tile_3dbag_cj = download_3dbag_cj(data_folder, tile_cj_url, error_3dbag, verbose=verbose)

            # All pand_id's without children/buildingparts (id's ending with "-<number>")
            pand_ids = [pid for pid in tile_3dbag_cj['CityObjects'].keys() if not '-' in pid]
            clean_ids = [pid[14:] for pid in pand_ids]

            bag_data = batch_query_bag(cursor_bag, clean_ids, columns=bag_columns)

            for pand_id in pand_ids:
                if tile_3dbag_cj['CityObjects'][pand_id]['attributes']['status'] != "Pand in gebruik":
                    remove_building(tile_3dbag_cj, pand_id)
                    continue

                if pand_id[14:] not in bag_data.keys() or bag_data[pand_id[14:]]['gebruiksdoel'] != "woonfunctie":
                    remove_building(tile_3dbag_cj, pand_id)
                    if verbose:
                        print(f"No pand with id '{pand_id}' in BAG.gpkg (probably a historic building).") # of bijvoorbeeld pand ten onrechte opgevoerd
                    continue
                bag_attributes = bag_data[pand_id[14:]]
 

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


# Add command line argurments
if __name__=="__main__":
    dotenv.load_dotenv()
    gpkg_bag_light = os.environ.get('gpkg_bag_light')
    data_folder = os.environ.get('data_folder')

    # main_api() # maar dubbele gebouwen? mss niet
    main_tilessqlite(
        data_folder, 
        gpkg_bag_light, 
        exists_ok=False, 
        error_3dbag=True, 
        verbose=False
    )
    # main_itsqlite(gpkg_list[0], gpkg_list[1], verbose=True) # Hoe geometry uit gpkg
