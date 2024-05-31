import numpy as np
import shapely
import math
import geopandas as gpd
import matplotlib.pyplot as plt
import reference_polygons
from shapely.affinity import translate


def save_pollist(plist, filename, path="./"):
    """ Saves a list of polygons as pickle format. Creates path if it does not exist.""" 
    if not(os.path.exists(path)):
        os.makedirs(path)
    with open(path+filename+".pkl", "wb") as f:
        pickle.dump(plist, f)


def load_pollist(filename, path="./"):
    """ Loads a list of polygons from a pickle file."""
    with open(path+filename+".pkl", "rb") as f:
        plist = pickle.load(f)
    return plist

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
    """Converts coordinates of shapely.polygons, consisting of two separate large lists of x coords and y coords, into a list of (x, y) coords.
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
    geo_coords = pol_to_vec(geometry)
    first_coord = geo_coords[0]
    for i in range(len(geo_coords)):
        geo_coords[i]=np.subtract(geo_coords[i],first_coord)
    return shapely.Polygon(geo_coords)

### Functions for calculating turning functions and distances
def truncate_angle(an):
    if an > np.pi:
        an = 2*np.pi - an
    return an

# Transforming the coord vectors into pairs of vectors corresponding with the normalized cumulative length and corresponding cumulative angle, 
# and the total length of the vector
def get_turning(coords_list, norm = True, tot_length = False):
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
    tf1, tf2 = matching_lengths(tf1, tf2)
    return compare_turning(tf1, tf2, metric)/np.pi


def rotate_tf(tf):
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
    c = 0
    while tf[1][-1] < 0.5 and c < len(tf[1]):
        tf = rotate_tf(tf)
        c += 1
    return tf

def minimize_dist(coords1, coords2, metric='l1', norm=True, tot_length=False):
    if len(coords1) <= len(coords2):
        tf1 = get_turning(coords1, norm=norm, tot_length=tot_length)
        tf2 = get_turning(coords2, norm=norm, tot_length=tot_length)
        tf2_mirror = get_turning([(-1*i[0], -1*i[1]) for i in coords2], norm=norm, tot_length=tot_length)
        turn_number = len(tf2[0])-1
    else:
        tf1 = get_turning(coords2, norm=norm, tot_length=tot_length)
        tf2 = get_turning(coords1, norm=norm, tot_length=tot_length)
        tf2_mirror = get_turning([(-1*i[0], -1*i[1]) for i in coords1], norm=norm, tot_length=tot_length)
        turn_number = len(tf2[0])-1
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

reference_shapes = load_pollist("30_ref_pols", path="reference_polygons/")
reference_shapes_coords = [pol_to_vec(i) for i in reference_shapes]

def convexity(pol):
    return (pol.area/pol.convex_hull.area)

def rectangularity(pol):
    return (pol.area/pol.minimum_rotated_rectangle.area)

def pol_to_new_space(pol, feature_space=pol_features_coords, additional_functions=[convexity, rectangularity], metric='l1'):
    """Pol_to_new_space: pol(shapely.polygon), feature_space(list(shapely.polygon)), additional_functions(list(lambda(shapely.pol) -> float)), metric: string
    Takes a polygon and computes the turning function distance to the set of reference polygons provided in 'feature_space'. 
    Other indices/features can be provided in 'additional_functions' in the form of a function mapping a shapely.polygon to a float.
    By default, adds convexity and rectangularity."""
    pol_coords = to_list(pol.exterior.coords)
    return np.array([minimize_dist(pol_coords, feature_space[i], metric=metric) for i in range(len(feature_space))]+[i(pol) for i in additional_functions])

def make_space(pol_list, features=pol_features_coords, additional_functions=[convexity, rectangularity], metric='l1'):
    """make_space: constructs feature space from list of polygons. See pol_to_new_space for details on how each instance is transformed to the feature space."""
    res = np.zeros((len(pol_list),len(features)))
    for i in range(len(pol_list)):
        res[i,:] = pol_to_new_space(pol_list[i], feature_space=features, additional_functions=[convexity, rectangularity], metric=metric)
        if i%10 == 0:
            print("Working on pol {0}".format(i))
    return res


def translate_pol(pol):
    # minx, miny, _, _ = pol.bounds
    xy = pol.exterior.coords[0]
    translate_x = -xy[0]
    translate_y = -xy[1]
    translated_pol = translate(pol, xoff=translate_x, yoff=translate_y)
    return translated_pol

def round_polygon(pol, decimals=1, simplify_tolerance=0.2):
    pol = pol.simplify(tolerance=0.2)
    xx, yy = pol.exterior.coords.xy
    xx = np.round(xx.tolist(), decimals)
    yy = np.round(yy.tolist(), decimals)
    return shapely.Polygon(zip(xx, yy))
    
