import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

class similarity:
    def __init__(self, gpkg_path: str, column_weights: dict = None, columns: list = None):
        self.gpkg_path = gpkg_path
        self.gpdf = gpd.read_file(self.gpkg_path)

        self._validate_input(column_weights, columns)

        self.column_weights = column_weights
        self.columns = self._get_columns(column_weights, columns)

        self.prepare_data()
    
    ## helper functions for __init__
    def _validate_input(self, column_weights, columns):
        if column_weights is not None and not isinstance(column_weights, dict):
            raise ValueError("column_weights must be a dictionary")
        if columns is not None and not isinstance(columns, list):
            raise ValueError("columns must be a list")

    def _get_columns(self, column_weights, columns):
        if column_weights is not None:  # overwrite columns if column_weights is given
            return list(column_weights.keys())
        elif columns is None and column_weights is None:
            return self.gpdf.columns.drop(["id", "geometry"])
        else:
            return columns
    
    def normalize(self):
        """normalize the columns in the geopandas dataframe so that every feature has the same weight in the distance calculation"""
        self.prepared_gpdf = self.gpdf.copy()
        
        for column in self.columns:
            if self.gpdf[column].dtype != "float64":
                self.prepared_gpdf[column] = self.gpdf[column].astype(float)
            try:
                self.prepared_gpdf[column] = (self.prepared_gpdf[column] - self.prepared_gpdf[column].mean()) / self.prepared_gpdf[column].std()
            except Exception as e:
                raise ValueError(f"Could not normalize column: {column}. Error: {e}")

    
    def scale(self):
        """scale the columns in the geopandas dataframe according to the given weights, so that the features have different weights in the distance calculation"""
        for column, weight in self.column_weights.items():
            self.prepared_gpdf[column] = self.prepared_gpdf[column] * weight

    def prepare_data(self):
        self.normalize()
        if self.column_weights is not None:
            self.scale()

    ## main functions
    def check_ids(self, id1, id2):
        # check if the ids are in the correct format
        if not id1.startswith("NL.IMBAG.Pand."):
            id1 = f"NL.IMBAG.Pand.{id1}-0"
        if not id2.startswith("NL.IMBAG.Pand."):
            id2 = f"NL.IMBAG.Pand.{id2}-0"

        # check if the ids are in the geopandas dataframe
        if id1 not in self.prepared_gpdf["id"].values:
            raise ValueError(f"id1 {id1} not found in the geopandas dataframe")
        if id2 not in self.prepared_gpdf["id"].values:
            raise ValueError(f"id2 {id2} not found in the geopandas dataframe")
        
        # if all good, return the ids
        return id1, id2

    def calculate_distance(self, id1, id2):

        id1, id2 = self.check_ids(id1, id2)
        
        obj1 = self.prepared_gpdf[self.prepared_gpdf["id"] == id1]
        obj2 = self.prepared_gpdf[self.prepared_gpdf["id"] == id2]

        dist = euclidean_distances(obj1[self.columns], obj2[self.columns])
        return dist[0][0]
    
    def calculate_similarity(self, id1, id2):
        dist = self.calculate_distance(id1, id2)
        if self.column_weights is not None:
            normalized_dist = dist / sum(self.column_weights.values())
        else:
            normalized_dist = dist / len(self.columns)
        return 1 / (1 + normalized_dist)
    
    def mirror(self, matrix):
        upper_tri = np.triu(matrix)
        lower_tri = upper_tri.T
        mirrored_matrix = upper_tri + lower_tri
        return mirrored_matrix

    def save(self, matrix, output_path, header):
        if matrix.shape[0] == matrix.shape[1]:
            print("Matrix is square")
            matrix = self.mirror(matrix)
        
        np.savetxt(output_path, matrix, delimiter=",", fmt='%f', header=header, comments='')
        return matrix

    def plot(self, matrix, all_ids):
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

    def distance_matrix(self, output_path: str = None, save_interval: int = 100, plot: bool = False):
        if isinstance(output_path, str) and output_path.endswith('.csv') == False:
            raise ValueError("output_path must end with '.csv'")

        # create an empty matrix and get all ids
        all_ids = self.prepared_gpdf["id"].values
        header = ",".join(all_ids)
        matrix = np.array([])

        # calculate the distance between all objects
        total_jobs = len(all_ids) * (len(all_ids) - 1) / 2
        progress = tqdm(total=total_jobs, desc="Calculating distance matrix")
        for i, id1 in enumerate(all_ids):
            row = np.array([0]*(i+1)) # set zero for all ids before the diagonal
            for id2 in all_ids[i+1:]:
                dist = self.calculate_distance(id1, id2)
                row = np.append(row, round(dist, 5))

            # add row to matrix
            if matrix.size == 0:
                matrix = row
            else:
                matrix = np.vstack((matrix, row))
            progress.update(len(all_ids) - i - 1)
            
            # save the matrix to a file if the interval is reached
            if isinstance(output_path, str) and matrix.ndim > 1 and matrix.shape[0] % save_interval == 0:
                self.save(matrix, output_path, header)
                #TODO: just append the new rows to the file instead of saving the whole matrix
        
        # make sure the full matrix is saved
        if isinstance(output_path, str):
            mirrored_matrix = self.save(matrix, output_path, header)
            print("Distance matrix calculated and saved to 'distance_matrix.csv'")
        else:
            mirrored_matrix = self.mirror(matrix)
            print("Distance matrix calculated")
        
        if plot:
            self.plot(mirrored_matrix, all_ids)

        return mirrored_matrix, all_ids

if __name__ == '__main__':
    #TODO: check if everything works with the -0 / NL.IMBAG.Pand and without those
    path = "analysis/voorbeeld_woningen.gpkg"
    
    # sim = similarity(path, {'dispersion_index_2d': 1, 'dispersion_index_3d': 1})
    sim = similarity(path)

    sim.distance_matrix(output_path="distance_matrix.csv")
    # gpdf = gpd.read_file(path)