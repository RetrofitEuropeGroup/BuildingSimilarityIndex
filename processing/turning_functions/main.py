import os
import sys
import shapely

import geopandas as gpd
import pandas as pd
import numpy as np

from tqdm import tqdm
from shapely.affinity import translate



def save_pollist(pollist, file_path):
    """ Saves a list of polygons as geopackage format. Creates path if it does not exist."""
    path = os.path.dirname(file_path)
    if not(os.path.exists(path)):
        os.makedirs(path)

    df = gpd.GeoDataFrame(geometry=pollist)
    driver="GPKG"
    df.to_file(os.path.join(file_path), driver=driver)


def polygon_to_vector(pol):
    """Converts a shapely polygon to a list of coordinates."""
    return [list(i) for i in pol.exterior.coords]


def an(v1):
    """ Computes angle between vector and x-axis. """
    v1 = v1/np.linalg.norm(v1)
    if v1[1] > 0:
        return np.arccos(v1[0])
    else:
        return 2*np.pi - np.arccos(v1[0])


## Therefore, given normalized vector:
def angle(v1,v2):
    """Angle between two vectors """
    return an(v2)-an(v1)


### Utils
# Conversie coordinaten naar lijst
def to_list(coords):
    """Converts coordinates of shapely.geometry.polygons, consisting of two separate large lists of x coords and y coords, into a list of (x, y) coords.
    Auxiliary function for computing the turning function."""
    length = len(coords)
    return [coords[i] for i in range(length)]

# Rotatie van de lijst
def rotate_list(alist, pos):
    """'Rotates' a list of polygon coordinates by cutting the first pos coordinates and appending them at the end."""
    return alist[pos:]+alist[:pos]


def seq_to_diff(v, no_start_zero=True):
    return np.ediff1d(v, to_begin=v[0])

# Function that translates a geometry to start in (0,0)
def update_geo(geometry):
    geo_coords = polygon_to_vector(geometry)
    first_coord = geo_coords[0]
    for i in range(len(geo_coords)):
        geo_coords[i]=np.subtract(geo_coords[i],first_coord)
    return shapely.geometry.Polygon(geo_coords)

### Functions for calculating turning functions and distances
def truncate_angle(an):
    """ rescale angles to be [0, pi] s.t. if an > pi, an-> 2*pi - an.
    """
    if an > np.pi:
        an = 2*np.pi - an
    return an

# Transforming the coord vectors into pairs of vectors corresponding with the normalized cumulative length and corresponding cumulative angle,
# and the total length of the vector
def get_turning(coords_list, norm = True, tot_length = False):
    """ Get turning function from list of polygon coordinates.
    Optional arguments:
    norm(bool): normalize turning function to be of unit length.
    tot_length(bool): return total length or not"""
    size = len(coords_list)
    # Omzetten coordinaten naar verschil vectoren.
    vectors = [np.subtract(coords_list[i+1],coords_list[i]) for i in range(size-1)]
    # Calculate lengths
    lengths = np.array([np.linalg.norm(vectors[i]) for i in range(len(vectors))])
    # Optional: Adding first vector for horizontal orientation
    # Optional: vectors = [np.array([1.,0.])]+vectors
    # Calculate angles, bij de eerste vector hoort hoek 0, bij de n+1-de vector de hoek tov de n-de vector.
    angles = np.array([0]+[angle(vectors[i],vectors[i+1]) for i in range(len(vectors)-1)])
    total_length = np.sum(lengths)
    cum_length = np.cumsum(lengths)#[sum(lengths[:i]) for i in range(1,size)]
    cum_angles = np.cumsum(angles)%(2*np.pi)#[sum(angles[:i])%(2*math.pi) for i in range(1,size)]
    #cum_angles = [truncate_angle(i) for i in cum_angles]
    if norm == True:
        cum_length = cum_length / total_length
    if tot_length == True:
        return(cum_length, cum_angles, total_length)
    return(np.array(cum_length),np.array(cum_angles))



# compare turning functions
## Input: Twee turning function vectoren
# input: twee paren vectoren, eerste vector uit paar is genormaliseerde cumulatieve lengte, tweede vector uit paar is bijbehorende hoek in radialen.
def matching_lengths(v1,v2):
    """ Combines two turning functions, such that they have angle values at the union of their lengh points, instead of only at the points where the angle changes... don't really know how to explain this more clearly."""
    len1 = len(v1[1])
    len2 = len(v2[1])
    i=0
    j=0
    new_v1 = [[],[]]
    new_v2 = [[],[]]
    while i<len1 and j<len2:
        if v1[0][i]<v2[0][j]:# Volgende breekpunt in v1
            new_v1[0].append(v1[0][i]) # v1 zelfde breekpunt
            new_v2[0].append(v1[0][i]) # Voeg breekpunt toe in v2
            new_v1[1].append(v1[1][i]) # Hoek v1
            new_v2[1].append(v2[1][j]) # hoek v2
            i+=1 # Volgende punt in v1
        elif v1[0][i]>v2[0][j]:# Volgende breekpunt in v2
            new_v1[0].append(v2[0][j])
            new_v2[0].append(v2[0][j])
            new_v1[1].append(v1[1][i])
            new_v2[1].append(v2[1][j])
            j+=1
        elif v1[0][i]==v2[0][j]:
            new_v1[0].append(v1[0][i])
            new_v2[0].append(v2[0][j])
            new_v1[1].append(v1[1][i])
            new_v2[1].append(v2[1][j])
            i+=1
            j+=1
    return np.array(new_v1),np.array(new_v2)



# input: twee turning vectoren met dezelfde lengte, optionele specificatie voor gebruikte metriek.
# output: de niet-geminimaliseerde afstand tussen de turning functies
def compare_turning(vec1,vec2,metric='l1'):
    """ Compare turning: computes distance metric between two turning functions. Assumes both turning functions have the same length/number of break points.
    Optional argument:
    metric: 'l1' or 'l2'.

    Computes:
    $[\int_{0}^{1} (\phi_1(s) - \ph_2(s) )^{l} ds ]^{1/l}$
    for two given turning functions with l the specific l-norm.
    """
    diff_len = seq_to_diff(vec1[0])
    subt = np.abs(vec1[1] - vec2[1])
    subt = [min(i, 2*np.pi - i) for i in subt]
    if metric == 'l1':
        tot_err = np.sum(np.multiply(diff_len, subt))
    elif metric == 'l2':
        tot_err = np.sqrt(np.sum(np.multiply(diff_len, np.square(subt))))
    return tot_err

# input: twee lijsten coordinaten, met optionele parameters
# output: de niet-geminimaliseerde afstand tussen de turning functies behorende bij de coordinate vectors
def dist_coords(coords1, coords2, metric='l1', norm=True, tot_length=False):
    tf1, tf2 = get_turning(coords1,norm=norm, tot_length=tot_length),get_turning(coords2,norm=norm, tot_length=tot_length)
    tf1, tf2 = matching_lengths(tf1, tf2)
    return compare_turning(tf1, tf2, metric)/np.pi



# input: twee lijsten coordinaten, met optionele parameters
# output: de niet-geminimaliseerde afstand tussen de turning functies behorende bij de coordinate vectors
def dist_coords2(tf1, tf2, metric='l1'):
    """ Computes normalized distance between two turning functions after matching their lengths"""
    tf1, tf2 = matching_lengths(tf1, tf2)
    return compare_turning(tf1, tf2, metric)/np.pi


def rotate_tf(tf):
    """Rotate_tf:
    Computes new turning function from old turning function by starting at the next initial point.
    i.e. this corresponds to rotating the polygon by one vertex and computing the turning function"""
    x = tf[0]
    y = tf[1]
    xnew = np.append(np.subtract(x[1:], x[0]), 1.0)
    ynew = np.append(np.subtract(y[1:], y[1]), 2*np.pi - y[1])
    def make_pos(el):
        if el < 0.0:
            return 2*np.pi + el
        else:
            return el
    return (xnew, np.array(list(map(make_pos, ynew))))

def rotate_until_corner(tf):
    """ Rotates turning function until a large angle is encountered if any large angles exist.
    Taking large angles as initial points helps with filtering out inaccuracies existing in the data imported from BAG.
    """
    c = 0
    while tf[1][-1] < 0.5 and c < len(tf[1]):
        tf = rotate_tf(tf)
        c += 1
    return tf

## COORDS 2 SHOULD BE FIXED, COORDS 1 SHOULD BE THE ROTATING AND MIRRORING ONE.
def minimize_dist(coords1, coords2, metric='l1', norm=True, tot_length=False):
    """
    Minimizes the turning function distance between two turning functions by rotating mirroring one polygon.
    """
    # if len(coords1) <= len(coords2):
    #     tf1 = get_turning(coords1, norm=norm, tot_length=tot_length)
    #     tf2 = get_turning(coords2, norm=norm, tot_length=tot_length)
    #     tf2_mirror = get_turning([(-1*i[0], -1*i[1]) for i in coords2], norm=norm, tot_length=tot_length)
    #     turn_number = len(tf2[0])-1
    # else:
    tf1 = get_turning(coords2, norm=norm, tot_length=tot_length)
    tf2 = get_turning(coords1, norm=norm, tot_length=tot_length)
    tf2_mirror = get_turning([(-1*i[0], i[1]) for i in coords1], norm=norm, tot_length=tot_length)
    turn_number = len(tf2[0])-1
    tf2 = rotate_until_corner(tf2)
    tf1 = rotate_until_corner(tf1)
    d1 = dist_coords2(tf1, tf2, metric=metric)
    d2 = dist_coords2(tf1, tf2_mirror, metric=metric)
    dist = np.minimum(d1, d2)
    while turn_number > 0:
        tf2 = rotate_tf(tf2)
        tf2_mirror = rotate_tf(tf2_mirror)
        turn_number -= 1
        d = dist_coords2(tf1, tf2, metric=metric)
        d2 = dist_coords2(tf1, tf2_mirror, metric=metric)
        dist = np.amin([dist, d, d2])
    return dist


def convexity(pol):
    """ 2D convexity measure.
    Computed by dividing the area of a polygon by the area of its convex hull"""
    return (pol.area/pol.convex_hull.area)

def rectangularity(pol):
    """ 2D rectangularity measure.
    Computed by dividing the area of a polygon by the area of the minimum rotated rectangle.
    """
    return (pol.area/pol.minimum_rotated_rectangle.area)

def polygon_to_new_space(pol, feature_space, metric='l1'):
    """Polygon_to_new_space: pol(shapely.geometry.polygon), feature_space(list(shapely.geometry.polygon)), metric: string
    Takes a polygon and computes the turning function distance to the set of reference polygons provided in 'feature_space'."""
    pol_coords = to_list(pol.exterior.coords)
    return np.array([minimize_dist(pol_coords, feature_space[i], metric=metric) for i in range(len(feature_space))])

def make_space(pol_list, features, metric='l1'):
    """make_space: constructs feature space from list of polygons. See polygon_to_new_space for details on how each instance is transformed to the feature space."""
    res = np.zeros((len(pol_list),len(features)))
    for i in tqdm(range(len(pol_list)), desc="Executing the turning function "):
        res[i,:] = polygon_to_new_space(pol_list[i], feature_space=features, metric=metric)
    return res


def translate_pol(pol):
    # minx, miny, _, _ = pol.bounds
    """Translates polygon such that the first coordinate is at (0, 0)"""
    xy = pol.exterior.coords[0]
    translate_x = -xy[0]
    translate_y = -xy[1]
    translated_pol = translate(pol, xoff=translate_x, yoff=translate_y)
    return translated_pol

def round_polygon(pol, decimals=1, simplify_tolerance=0.2):
    """Rounds the coordinates of a polygon to the nearest tolerance and decimals."""
    pol = pol.simplify(tolerance=simplify_tolerance)
    xx, yy = pol.exterior.coords.xy
    xx = np.round(xx.tolist(), decimals)
    yy = np.round(yy.tolist(), decimals)
    return shapely.geometry.Polygon(zip(xx, yy))

def remove_straight_vertices(pol, tolerance=5):
    """ Removes vertices located along a straight edge, defined by an angle of 180 +/- 5 degrees."""
    xx, yy = pol.exterior.coords.xy
    nr_coords = len(xx)
    xx_new = []
    yy_new = []
    angles = []
    for i in range(nr_coords):
        p2 = np.array([xx[i], yy[i]])
        p1 = np.array([xx[i-1], yy[i-1]]) - p2
        p3 = np.array([xx[(i+1)%nr_coords], yy[(i+1)%nr_coords]]) - p2
        if not(np.array_equal(p1, p3)):
            denom = (np.sqrt(np.dot(p1, p1))*np.sqrt(np.dot(p3, p3)))
            if denom == 0:
                denom = sys.float_info.epsilon
            angle = 360*np.arccos(np.dot(p1, p3)/denom)/(2*np.pi)
            if not((180-tolerance) < angle and angle < (180 + tolerance)):
                xx_new += [xx[i]]
                yy_new += [yy[i]]
    return shapely.geometry.Polygon(zip(xx_new, yy_new))


def perform_turning_function(df, 
                        reference_shapes_path=None,
                        geometry_column='geometry',
                        metric='l1'
                        ):

    ## One file provided:
    """ The dataframe should be a geopandas-dataframe and should contain a geometry feature (name can be provided) with the 2D shape as a polygon.
    Computes turning function for polygon and computes the minimal turning function distance to all the reference polygons in geometry_features.
    The reference polygons are computed by doing 30 PCA loops of the feature space. Polygons were taken from both real complexes in The Netherlands and generated parameterized polygons.
    """

    if reference_shapes_path is None:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        reference_shapes_path = os.path.join(file_dir, "reference_polygons", "30_ref_pols.gpkg")
    reference_shapes = gpd.read_file(reference_shapes_path)
    
    pols = list(df[geometry_column])
    if type(pols[0]) == shapely.geometry.MultiPolygon:
        pols = [list(i.geoms)[0] for i in pols]
    pols = [remove_straight_vertices(round_polygon(translate_pol(i))) for i in pols]

    # TODO: we could speed this up through parallelization
    # create the feature space based on the turning function and save it to a dataframe
    feature_space = make_space(pols, features=[polygon_to_vector(i) for i in reference_shapes.geometry], metric=metric)    
    columns = list(f"turning_function_{i}" for i in range(len(reference_shapes.geometry)))
    df_turning_function = pd.DataFrame(feature_space, columns=columns)
    df_turning_function = df_turning_function.set_index(df.index)
    
    return df_turning_function
