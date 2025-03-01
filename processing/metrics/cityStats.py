import json
import os
import math

import numpy as np
import pandas as pd
import geopandas
import pyvista as pv
import scipy.spatial as ss

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import processing.metrics.cityjson as cityjson
import processing.metrics.geometry as geometry
import processing.metrics.shape_index as si

def compute_stats(values, percentile = 90, percentage = 75):
    """
    Returns the stats (mean, median, max, min, range etc.) for a set of values.
    """
    hDic = {'Mean': np.mean(values), 'Median': np.median(values),
    'Max': max(values), 'Min': min(values), 'Range': (max(values) - min(values)),
    'Std': np.std(values)}
    m = max([values.count(a) for a in values])
    if percentile:
        hDic['Percentile'] = np.percentile(values, percentile)
    if percentage:
        hDic['Percentage'] = (percentage/100.0) * hDic['Range'] + hDic['Min']
    if m>1:
        hDic['ModeStatus'] = 'Y'
        modeCount = [x for x in values if values.count(x) == m][0]
        hDic['Mode'] = modeCount
    else:
        hDic['ModeStatus'] = 'N'
        hDic['Mode'] = np.mean(values)
    return hDic

def convexhull_volume(points):
    """Returns the volume of the convex hull"""
    try:
        return ss.ConvexHull(points).volume
    except Exception as e:
        print(f"Error: {e}")
        return 0

def get_errors(objid, cm):
    """Return false / true if it finds val3dity error in the cityjson"""
    obj = cm["CityObjects"][objid]

    if not "geometry" in obj or len(obj["geometry"]) == 0:
        return False

    parid = obj["parents"][0] # we can asume that there is a parent, as we only process buildingparts in egible function
    b3_val3dity_lod22 = cm['CityObjects'][parid]['attributes']['b3_val3dity_lod22']
    
    if b3_val3dity_lod22 == '[]':
        return False
    else:
        return True

def check_suitability(df, formatted_ids):
    ## filter based on some conditions
    # filter out the buldings with actual_volume lower than 40, no holes
    # and actual volume larger than convex hull volume
    perc_convex = sum(df['actual_volume'] > df['convex_hull_volume']) / len(formatted_ids)
    perc_vol40 = sum(df['actual_volume'] < 40) / len(formatted_ids)
    perc_holes = sum(df['hole_count'] > 0) / len(formatted_ids)
    clean = df[(df['actual_volume'] <= df['convex_hull_volume']) & (df['actual_volume'] >= 40) & (df['hole_count'] == 0)]
    return clean, perc_convex, perc_vol40, perc_holes

def check_metric_values(clean, df, formatted_ids):
    """filters out the buildings with index values out of the range (0-1.2)"""
    # start with determining which indices are 2d and 3d
    indices_2d = [col for col in df.columns if col.endswith("_2d")] + ["horizontal_elongation"]
    indices_3d = ([col for col in df.columns if col.endswith("_3d")] +
                ["horizontal_elongation"])

    # filter out the buildings with index values out of the range
    min_value = 0
    max_value = 1.2

    before = len(clean)
    for ind in indices_2d:
        clean = clean[(clean[ind] >= min_value) & (clean[ind] <= max_value)]
    for ind in indices_3d:
        clean = clean[(clean[ind] >= min_value) & (clean[ind] <= max_value)]
    outfiltered_metrics = (before - len(clean)) / len(formatted_ids)
    return clean, outfiltered_metrics   

def clean_df(df, formatted_ids, val3dity_errors, verbose=True):
    """Cleans the dataframe, it removes the buildings with errors,
    holes, and buildings with index values out of the normal range."""
    # check the suitability of the buildings based on 3 conditions
    
    
    perc_bag_errors = 1 - (len(df)+val3dity_errors) / len(formatted_ids)
    lost_perc = f"""\t{perc_bag_errors:.2%} \t has issues with the BAG data, meaning that they were not found in the BAG or had an error
        {val3dity_errors/len(formatted_ids):.2%} \t has issues with their 3d shape (val3dity error)\n"""
    
    clean, perc_convex, perc_vol40, perc_holes = check_suitability(df, formatted_ids)
    clean = clean.drop(columns=["hole_count", 'min_vertical_elongation', 'max_vertical_elongation'], axis=1)
    
    clean, outfiltered_metrics = check_metric_values(clean, df, formatted_ids)

    # let the user know how many buildings have been filtered out
    rows_lost_perc = 1 - len(clean) / len(formatted_ids)
    if verbose and rows_lost_perc == 0:
        print("INFO: All buildings are suitable for processing. None have been filtered out.")
    elif verbose:
        lost_perc += f"""\t{perc_convex:.2%} \t has a actual_volume that is larger than convex_hull_volume
        {perc_vol40:.2%} \t has an actual_volume smaller than 40
        {perc_holes:.2%} \t has a hole_count that is not zero
        {outfiltered_metrics:.2%} \t has metrics values out of the normal range"""
        header = f"INFO: {rows_lost_perc:.2%} of the buildings has been filtered out as it did not meet one of the requirements, note that buildings can be filtered out for multiple reasons"
        print(f"{header}\n{lost_perc}")
    return clean

def eligible(cm, id):
    """Returns True if the building is eligible for processing, otherwise returns False"""    
    # we only process buildingparts as they are 3D
    if cm["CityObjects"][id]['type'] != 'BuildingPart':
        return False
    
    # we cannot process multiple children as some children are really small
    parent_id = cm["CityObjects"][id]['parents'][0]
    if len(cm['CityObjects'][parent_id]['children']) > 1:
        return False

    # check if there are any val3dity errors with the building
    errors = get_errors(id, cm)
    return not errors

def get_parent_attributes(cm, obj):
    """Returns the attributes of the parent of the given object. The parent should be in the same city model."""
    building = cm["CityObjects"][obj]
    if "parents" in building.keys():
        parent_id = building["parents"][0]
        parent = cm["CityObjects"][parent_id]
        return parent['attributes']
    else:
        return None

class StatValuesBuilder:

    def __init__(self, values, indices_list) -> None:
        self.__values = values
        self.__indices_list = indices_list

    def compute_index(self, index_name):
        """Returns True if the given index is supposed to be computed"""

        return self.__indices_list is None or index_name in self.__indices_list

    def add_index(self, index_name, index_func):
        """Adds the given index value to the dict"""

        if self.compute_index(index_name):
            self.__values[index_name] = index_func()
        else:
            self.__values[index_name] = "NC"

def add_purpose_of_use(values, actual_use):
    possible_uses = ['woonfunctie', 'bijeenkomstfunctie', 'celfunctie', 'gezondheidszorgfunctie', 'industriefunctie', 
    'kantoorfunctie', 'logiesfunctie', 'onderwijsfunctie', 'sportfunctie', 'winkelfunctie',
    'overige gebruiksfunctie']
    uses_found = 0
    # add the possible uses to the dict to make it a seperate column
    for use in possible_uses:
        if actual_use is not None and use in actual_use: # if actual_use is None, just put 0 for every use
            uses_found += 1
            values[use] = 1
        else:
            values[use] = 0

    if actual_use is not None and uses_found != len(actual_use):
        print(f'WARNING: Found {uses_found} uses, but expected {len(actual_use)}. actual_use: {actual_use}')
    return values    

def get_n_val3dity_errors(cm):
    total = 0
    for building in cm["CityObjects"].values():
        if 'attributes' in building:
            if building['attributes']['b3_val3dity_lod22'] != '[]':
                total += 1
    return total

def process_building(building,
                     obj,
                     filter,
                     density_2d,
                     density_3d,
                     vertices,
                     custom_indices=None):

    if not filter is None and filter != obj:
        return obj, None

    # Skip if type is not Building or Building part
    if not building["type"] in ["BuildingPart"]:
        return obj, None

    # Skip if no geometry
    if not "geometry" in building or len(building["geometry"]) == 0:
        return obj, None
    if len(building["geometry"]) < 3:
        return obj, None
    geom = building["geometry"][2]

    mesh = cityjson.to_polydata(geom, vertices).clean()
    try:
        tri_mesh = cityjson.to_triangulated_polydata(geom, vertices).clean()
    except:
        print(f"{obj} geometry parsing crashed! Omitting...")
        return obj, {"type": building["type"]}

    tri_mesh, _ = geometry.move_to_origin(tri_mesh)

    fixed = tri_mesh

    points = cityjson.get_points(geom, vertices)
    ch_volume = convexhull_volume(points)

    if "semantics" in geom:
        roof_points = geometry.get_points_of_type(mesh, "RoofSurface")
        ground_points = geometry.get_points_of_type(mesh, "GroundSurface")
    else:
        roof_points = []
        ground_points = []

    if len(roof_points) == 0:
        height_stats = compute_stats([0])
    else:
        height_stats = compute_stats([v[2] for v in roof_points])

    if len(ground_points) > 0:
        shape = cityjson.to_shapely(geom, vertices)
    else:
        shape = cityjson.to_shapely(geom, vertices, ground_only=False)

    obb_2d = cityjson.to_shapely(geom, vertices, ground_only=False).minimum_rotated_rectangle

    # Compute OBB with shapely
    min_z = np.min(mesh.clean().points[:, 2])
    max_z = np.max(mesh.clean().points[:, 2])
    obb = geometry.extrude(obb_2d, min_z, max_z)

    # Get the dimensions of the 2D oriented bounding box
    S, L = si.get_box_dimensions(obb_2d)

    b3_opp_grond = building['attributes'].get("b3_opp_grond")
    if b3_opp_grond is not None:
        b3_hellingshoek_proxy = building['attributes'].get("b3_opp_dak_plat", 0) + building['attributes'].get("b3_opp_dak_schuin", 0) / b3_opp_grond
    else:
        print(f'WARNING: b3_opp_grond not found for building {obj}')
        b3_hellingshoek_proxy = None

    values = {
        "actual_volume": round(fixed.volume, 2),
        "convex_hull_volume": round(ch_volume, 2),
        "oorspronkelijkbouwjaar": building['attributes']['oorspronkelijkbouwjaar'],
        "b3_opp_buitenmuur": building['attributes']['b3_opp_buitenmuur'],
        "b3_opp_dak_plat": building['attributes']['b3_opp_dak_plat'],
        "b3_opp_dak_schuin": building['attributes']['b3_opp_dak_schuin'],
        "b3_opp_grond": building['attributes']['b3_opp_grond'],
        "b3_opp_scheidingsmuur": building['attributes']['b3_opp_scheidingsmuur'],
        "b3_bouwlagen": building['attributes']['b3_bouwlagen'],
        "b3_hellingshoek_proxy": round(b3_hellingshoek_proxy,2),
        "aantal_verblijfsobjecten": building['attributes'].get("aantal_verblijfsobjecten"),
        "totaal_oppervlakte": building['attributes'].get("totaal_oppervlakte"),
        "main_roof_parts": building['attributes'].get("main_roof_parts"),
        "main_roof_area": building['attributes'].get("main_roof_area"),
        "other_roof_parts": building['attributes'].get("other_roof_parts"),
        "other_roof_area": building['attributes'].get("other_roof_area"),
        "part_ratio": building['attributes'].get("part_ratio"),
        "area_ratio": building['attributes'].get("area_ratio"),
        "hole_count": tri_mesh.n_open_edges,
        "geometry": shape,
    }
    purposes = building['attributes'].get("gebruiksdoelen")
    if purposes is not None:
        purposes = np.unique(purposes)
    values = add_purpose_of_use(values, purposes)

    voxel = pv.voxelize(tri_mesh, density=density_3d, check_surface=False)
    grid = voxel.cell_centers().points

    builder = StatValuesBuilder(values, custom_indices)
    builder.add_index("circularity_2d", lambda: si.circularity(shape))
    builder.add_index("hemisphericality_3d", lambda: si.hemisphericality(fixed))
    builder.add_index("convexity_2d", lambda: shape.area / shape.convex_hull.area)
    builder.add_index("convexity_3d", lambda: fixed.volume / ch_volume)
    builder.add_index("fractality_2d", lambda: si.fractality_2d(shape))
    builder.add_index("fractality_3d", lambda: si.fractality_3d(fixed))
    builder.add_index("rectangularity_2d", lambda: shape.area / shape.minimum_rotated_rectangle.area)
    builder.add_index("rectangularity_3d", lambda: fixed.volume / obb.volume)
    builder.add_index("squareness_2d", lambda: si.squareness(shape))
    builder.add_index("cubeness_3d", lambda: si.cubeness(fixed))
    builder.add_index("horizontal_elongation", lambda: si.elongation(S, L))
    builder.add_index("min_vertical_elongation", lambda: si.elongation(L, height_stats["Max"]))
    builder.add_index("max_vertical_elongation", lambda: si.elongation(S, height_stats["Max"]))
    builder.add_index("form_factor_3D", lambda: shape.area / math.pow(fixed.volume, 2/3))
    builder.add_index("equivalent_rectangularity_index_2d", lambda: si.equivalent_rectangular_index(shape))
    builder.add_index("equivalent_prism_index_3d", lambda: si.equivalent_prism_index(fixed, obb))
    builder.add_index("proximity_index_2d", lambda: si.proximity_2d(shape, density=density_2d))
    builder.add_index("proximity_index_3d", lambda: si.proximity_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
    builder.add_index("exchange_index_2d", lambda: si.exchange_2d(shape))
    builder.add_index("spin_index_2d", lambda: si.spin_2d(shape, density=density_2d))
    builder.add_index("spin_index_3d", lambda: si.spin_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
    builder.add_index("perimeter_index_2d", lambda: si.perimeter_index(shape))
    builder.add_index("circumference_index_3d", lambda: si.circumference_index_3d(tri_mesh))
    builder.add_index("depth_index_2d", lambda: si.depth_2d(shape, density=density_2d))
    builder.add_index("depth_index_3d", lambda: si.depth_3d(tri_mesh, density=density_3d) if len(grid) > 2 else "NA")
    builder.add_index("girth_index_2d", lambda: si.girth_2d(shape))
    builder.add_index("girth_index_3d", lambda: si.girth_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
    builder.add_index("dispersion_index_2d", lambda: si.dispersion_2d(shape, density=density_2d))
    builder.add_index("dispersion_index_3d", lambda: si.dispersion_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
    builder.add_index("range_index_2d", lambda: si.range_2d(shape))
    builder.add_index("range_index_3d", lambda: si.range_3d(tri_mesh))
    builder.add_index("roughness_index_2d", lambda: si.roughness_index_2d(shape, density=density_2d))
    builder.add_index("roughness_index_3d", lambda: si.roughness_index_3d(tri_mesh, grid, density_2d) if len(grid) > 2 else "NA")
    return obj, values


# Assume semantic surfaces
def calculate_metrics(input,
                    formatted_ids,
                    output_path=None,
                    filter=None,
                    without_indices=False,
                    jobs=1,
                    density_2d=1.0,
                    density_3d=1.0,
                    verbose=False):
    """Uses a cityjson file to calculate the 2d / 3d metrics for the buildings in the cityjson file"""

    with open(input, "r") as f:
        cm = json.load(f)

    if "transform" in cm:
        s = cm["transform"]["scale"]
        t = cm["transform"]["translate"]
        verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
                for v in cm["vertices"]]
    else:
        verts = cm["vertices"]

    # mesh points
    vertices = np.array(verts)

    # Count the number of jobs
    total_jobs = 0
    for obj in cm["CityObjects"]:
        if eligible(cm, obj):
            total_jobs += 1

    num_cores = jobs
    if verbose:
        print(f'Using {num_cores} core(s) to process {total_jobs} building(s)')
    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        # add the jobs to the pool
        futures = []
        for obj in cm["CityObjects"]:
            if not eligible(cm, obj):
                continue

            building = cm["CityObjects"][obj]
            if 'attributes' not in building:
                building['attributes'] = get_parent_attributes(cm, obj) #TODO: check do we process multiple childs of the same parent? --> No we don't

            indices_list = [] if without_indices else None

            future = pool.submit(process_building,
                                building,
                                obj,
                                filter,
                                density_2d,
                                density_3d,
                                vertices,
                                indices_list)
            futures.append(future)

        # wait for the jobs to finish and add the results to the stats
        with tqdm(total=total_jobs, desc="Calculating the 2D & 3D metrics") as progress:
            stats = {}
            for future in as_completed(futures):
                # retrieve the result
                obj, vals = future.result()
                if not vals is None:
                    obj = obj.replace('NL.IMBAG.Pand.', '')
                    obj = obj.split('-')[0]
                    stats[obj] = vals
                progress.update(1) # update the progress bar

    df = pd.DataFrame.from_dict(stats, orient="index")
    df.index.name = "id"

    try:
        n_val3dity_errors = get_n_val3dity_errors(cm)
        clean = clean_df(df, formatted_ids, n_val3dity_errors, verbose)
    except Exception as e:
        print(f"ERROR: Problem with cleaning the dataframe, using the original. Error message: {e}")
        clean = df

    try:
        if output_path is not None and output_path.endswith(".csv"):
            clean.to_csv(output_path)
        elif output_path is not None and output_path.endswith('.gpkg'):
            gdf = geopandas.GeoDataFrame(clean, geometry="geometry")
            gdf.to_file(f"{output_path}", driver="GPKG")
        elif output_path is not None:
            raise ValueError("output_path should be a .csv or .gpkg file")
    except Exception as e:
        print(f"ERROR: could not save the file. Error message: {e}")
    return clean
