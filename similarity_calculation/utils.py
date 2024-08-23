import os

import numpy as np
import matplotlib.pyplot as plt

def plot_matrix(matrix, all_ids):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix is not square, cannot plot")
    if matrix.shape[0] != len(all_ids):
        raise ValueError("Number of ids does not match the matrix dimensions")
    if len(all_ids) > 50:
        raise ValueError("Too many ids to plot, max 50 ids allowed")
    
    plt.matplotlib.pyplot.matshow(matrix)
    plt.xticks(range(len(all_ids)), all_ids, rotation=90)
    plt.yticks(range(len(all_ids)), all_ids)
    plt.colorbar()
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
    if os.path.isdir(parent_dir) == False:
        os.mkdir(parent_dir)

    np.savetxt(path, matrix_with_index, delimiter=",", fmt="%s", header=header, comments='')
    return matrix # we return the matrix without the index so it can be used for further calculations

def check_csv(path):
    if isinstance(path, str) and path.endswith('.csv') == False:
        raise ValueError("the path must end with '.csv'")