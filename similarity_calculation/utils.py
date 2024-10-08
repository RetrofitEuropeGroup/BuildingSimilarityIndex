import os

import numpy as np
import matplotlib.pyplot as plt

def plot_matrix(matrix, formatted_ids, reference_ids=None):
    if matrix.shape[0] != matrix.shape[1] and reference_ids is None:
        raise ValueError("Matrix is not square and is also not a reference matrix, cannot plot")
    if len(formatted_ids) > 100 or (reference_ids is not None and len(reference_ids) > 100):
        raise ValueError(f"Too many ids to plot, max 100 ids allowed: {len(formatted_ids)}, {len(reference_ids)}")
    
    plt.matplotlib.pyplot.matshow(matrix)
    if reference_ids is not None:
        plt.xticks(range(len(reference_ids)), reference_ids, rotation=90)
        plt.ylabel("Reference IDs")
        plt.xlabel("Other IDs")
    else:
        plt.xticks(range(len(formatted_ids)), formatted_ids, rotation=90)
    plt.yticks(range(len(formatted_ids)), formatted_ids)
    plt.colorbar().set_label('Distance')
    plt.show()

# function to calculate the distance matrix
def mirror(matrix):
    upper_tri = np.triu(matrix)
    lower_tri = upper_tri.T
    mirrored_matrix = upper_tri + lower_tri
    return mirrored_matrix

def save_matrix(matrix, path, header, index):
    if matrix.shape[0] == matrix.shape[1]:
        matrix = mirror(matrix)
    matrix_with_index = matrix.astype(str)
    matrix_with_index = np.insert(matrix_with_index, 0, index, axis=1)
    
    # check if the path is in an existing directory
    parent_dir = os.path.dirname(path)
    if parent_dir == '':
        parent_dir = '.'
    if os.path.isdir(parent_dir) == False:
        os.mkdir(parent_dir)

    np.savetxt(path, matrix_with_index, delimiter=",", fmt="%s", header=header, comments='')
    return matrix # we return the matrix without the index so it can be used for further calculations

def check_csv(path):
    if isinstance(path, str) and path.endswith('.csv') == False:
        raise ValueError("the path must end with '.csv'")
    
def format_ids(all_ids):
    formatted_ids = []
    for id in all_ids:        
        if id.startswith("NL.IMBAG.Pand."): # remove the prefix if it is there
            id = id[14:]
        if '-' in id: # remove the suffix if it is there
            id = id.split('-')[0]
        formatted_ids.append(id)
    return formatted_ids