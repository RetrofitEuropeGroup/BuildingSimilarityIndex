import json
import os
import math
import subprocess

import numpy as np
import pandas as pd
import geopandas
import pyvista as pv
import scipy.spatial as ss
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


import metrics.cityjson as cityjson
import metrics.geometry as geometry
import metrics.shape_index as si

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

def get_errors_from_report(report, objid, cm):
    """Return the report for the feature of the given obj"""
    if not "features" in report:
        return []

    obj = cm["CityObjects"][objid]

    if not "geometry" in obj or len(obj["geometry"]) == 0:
        return []

    if "parents" in obj:
        parid = obj["parents"][0]
    else:
        parid = None

    for f in report["features"]:
        if f["id"] in [parid, objid]:
            if f['validity'] == False:
                return True
            else:
                return []

    return []

def save_filter_stats(outfiltered_2d, outfiltered_3d, output):
    # split the output path to get the output folder
    output_folder_parts = os.path.split(output)[:-1]
    logger_path = os.path.join(*output_folder_parts, "filtered_buildings.csv")

    output_name = os.path.split(output)[-1]

    # update the logger
    new_line = f'{output_name};{round(outfiltered_2d, 4)};{round(outfiltered_3d, 4)}\n'
    if os.path.exists(logger_path) == False:
        with open(logger_path, 'w') as f:
            f.write('output_filename;filtered_2d;filtered_3d\n')
    with open(logger_path, 'a') as f:
        f.write(new_line)


def clean_df(df, output):
    """Cleans the dataframe, it removes the buildings with errors,
    holes, and buildings with index values out of the normal range."""
    ## filter based on some conditions
    # filter out the buldings with actual_volume lower than 40, no holes
    # and actual volume larger than convex hull volume
    clean = df[df['actual_volume'] < df['convex_hull_volume']]
    clean = clean[clean['actual_volume'] >= 40]
    clean = clean[clean['hole_count'] == 0]

    ## filter out the buildings with index values out of the range
    # start with determining which indices are 2d and 3d

    indices_2d = [col for col in df.columns if col.endswith("_2d")] + ["horizontal_elongation"]

    indices_3d = ([col for col in df.columns if col.endswith("_3d")] +
                ["horizontal_elongation"])

    # filter out the buildings with index values out of the range
    min_value = 0
    max_value = 1.2

    before2d = len(clean)
    for ind in indices_2d:
        clean = clean[(clean[ind] >= min_value) & (clean[ind] <= max_value)]
    after2d = len(clean)
    before3d = len(clean)
    for ind in indices_3d:
        clean = clean[(clean[ind] >= min_value) & (clean[ind] <= max_value)]
    outfiltered_2d = (before2d - after2d) / before2d
    outfiltered_3d = (before3d - len(clean)) / before3d

    save_filter_stats(outfiltered_2d, outfiltered_3d, output)
    # filter out the irrelevant columns

    irrelevant_columns = ["type", "lod", "errors", "valid", "orientation_values", "orientation_edges",
                          "hole_count", 'min_vertical_elongation', 'max_vertical_elongation']
    for col in irrelevant_columns:
        if col in clean.columns:
            clean = clean.drop(columns=col)

    return clean

def eligible(cm, id, report):
    """Returns True if the building is eligible for processing"""
    if report == {}:
        return True

    errors = get_errors_from_report(report, id, cm)
    if errors:
        return False

    if cm["CityObjects"][id]['type'] != 'BuildingPart':
        return False

    parent_id = cm["CityObjects"][id]['parents'][0]
    if len(cm['CityObjects'][parent_id]['children']) > 1:
        return False
    return True

def get_parent_attributes(cm, obj):
    """Returns the attributes of the parent of the given object. The parent should be in the same city model."""
    building = cm["CityObjects"][obj]
    if "parents" in building.keys():
        parent_id = building["parents"][0]
        parent = cm["CityObjects"][parent_id]
        return parent['attributes']
    else:
        return None

def get_report(input):
    # create the val3dity report with the same name as the input file
    val3dity_report = f"{input.name[:-5]}_report.json"
    # determine the location of the val3dity command
    # TODO: this can just be 1 line right? No need to check if it's in processing or not
    if 'processing' in os.getcwd():
        val3dity_cmd_location = os.path.join(os.getcwd(), 'metrics/val3dity/val3dity')
    else:
        val3dity_cmd_location = os.path.join(os.getcwd(), 'processing/metrics/val3dity/val3dity')

    if os.path.exists(val3dity_report):
        report = open(val3dity_report, "rb")
    else:
        try:
            subprocess.check_output(f'{val3dity_cmd_location} {input.name} -r {val3dity_report}')
            report = open(val3dity_report, "rb")
        except Exception as e:
            report = {}
            print(f"Warning: Could not run val3dity, continuing without report")
    return report

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

def process_building(building,
                     obj,
                     filter,
                     repair,
                     plot_buildings,
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

    geom = building["geometry"][2]

    mesh = cityjson.to_polydata(geom, vertices).clean()
    try:
        tri_mesh = cityjson.to_triangulated_polydata(geom, vertices).clean()
    except:
        print(f"{obj} geometry parsing crashed! Omitting...")
        return obj, {"type": building["type"]}

    tri_mesh, t = geometry.move_to_origin(tri_mesh)

    if plot_buildings:
        print(f"Plotting {obj}")
        tri_mesh.plot(show_grid=True)

    if repair:
        mfix = MeshFix(tri_mesh)
        mfix.repair()
        fixed = mfix.mesh
    else:
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

    values = {
        "actual_volume": fixed.volume,
        "convex_hull_volume": ch_volume,
        "oorspronkelijkbouwjaar": building['attributes']['oorspronkelijkbouwjaar'],
        "b3_opp_buitenmuur": building['attributes']['b3_opp_buitenmuur'],
        "b3_opp_dak_plat": building['attributes']['b3_opp_dak_plat'],
        "b3_opp_dak_schuin": building['attributes']['b3_opp_dak_schuin'],
        "b3_opp_grond": building['attributes']['b3_opp_grond'],
        "b3_opp_scheidingsmuur": building['attributes']['b3_opp_scheidingsmuur'],
        "hole_count": tri_mesh.n_open_edges,
        "geometry": shape,
    }


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
def process_cityjson(input,
         output,
         filter=None,
         repair=False,
         plot_buildings=False,
         without_indices=False,
         jobs=1,
         density_2d=1.0,
         density_3d=1.0):

    with open(input, "r") as input:
        cm = json.load(input)

        # if no output file is provided, use the name of the input file. But in the output folder
        if output is None:
            output = input.name[:-5] + ".csv"
            output = output.replace('input', 'output')

        # create and open the report
        report = get_report(input)

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
        if eligible(cm, obj, report):
            total_jobs += 1


    num_cores = jobs
    print(f'Using {num_cores} cores to process {total_jobs} buildings')
    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        # add the jobs to the pool
        futures = []
        for obj in cm["CityObjects"]:
            if not eligible(cm, obj, report):
                continue

            building = cm["CityObjects"][obj]
            if 'attributes' not in building:
                building['attributes'] = get_parent_attributes(cm, obj) #TODO: check do we process multiple childs of the same parent?

            indices_list = [] if without_indices else None

            future = pool.submit(process_building,
                                building,
                                obj,
                                filter,
                                repair,
                                plot_buildings,
                                density_2d,
                                density_3d,
                                vertices,
                                indices_list)
            futures.append(future)

        # wait for the jobs to finish and add the results to the stats
        with tqdm(total=total_jobs) as progress:
            stats = {}
            for future in as_completed(futures):
                # retrieve the result
                obj, vals = future.result()
                if not vals is None:
                    stats[obj] = vals
                progress.update(1) # update the progress bar

    df = pd.DataFrame.from_dict(stats, orient="index")
    df.index.name = "id"
    try:
        clean = clean_df(df, output)
    except Exception as e:
        print(f"ERROR: Problem with cleaning the dataframe: {e}")
        clean = df

    try:
        if output.endswith(".csv"):
            clean.to_csv(output)
        elif output.endswith('.gpkg'):
            gdf = geopandas.GeoDataFrame(clean, geometry="geometry")
            gdf.to_file(f"{output}", driver="GPKG")
        else:
            clean.to_excel(output)
    except Exception as e:
        print(f"ERROR: could not save the file: {e}")
        clean.to_csv("emergency.csv")

    if os.path.exists('val3dity.log'): # clean the mess
        os.remove('val3dity.log')

if __name__ == "__main__":
    process_cityjson("data/bag_data_merged/merged_1.city.json", output= "./data/gpkg/test.gpkg")