import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances


# TODO: split this up into multiple files

class similarity:
    def __init__(self, gpkg_path: str, column_weights: dict = None, columns: list = None):
        self._validate_input(column_weights, columns)
        
        # load the geopandas dataframe
        self.gpkg_path = gpkg_path
        self.gpdf = gpd.read_file(self.gpkg_path)

        # needed to know which columns are relevant for the distance calculation, and if the columns should be scaled
        self.columns = self._get_columns(column_weights, columns)
        self.column_weights = column_weights

        # prepare (scaling & normalizing) the data for the distance calculation
        self.prepared_gpdf = self.prepare_data(self.gpdf)
        self.prepared_gpdf_ref = None
    
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
    
    # function to prepare the data for the distance calculation
    def prepare_data(self, gpdf):
        normalized_gpdf = self._normalize(gpdf)

        # scale only if needed (if column_weights is given)
        if self.column_weights is not None:
            prepared_gpdf = self._scale(normalized_gpdf)
        else:
            prepared_gpdf = normalized_gpdf
        return prepared_gpdf

    def _normalize(self, gpdf):
        """normalize the columns in the geopandas dataframe so that every feature has the same weight in the distance calculation"""
        prepared_gpdf = gpdf.copy()
        
        for column in self.columns:
            if self.gpdf[column].dtype != "float64":
                prepared_gpdf[column] = prepared_gpdf[column].astype(float)
            try:
                prepared_gpdf[column] = (prepared_gpdf[column] - prepared_gpdf[column].mean()) / prepared_gpdf[column].std()
            except Exception as e:
                raise ValueError(f"Could not normalize column: {column}. Error: {e}")
        
        return prepared_gpdf
    
    def _scale(self, gpdf):
        """scale the columns in the geopandas dataframe according to the given weights, so that the features have different weights in the distance calculation"""
        for column, weight in self.column_weights.items():
            self.gpdf[column] = self.gpdf[column] * weight

    ## main functions
    # function to calculate the distance between two objects
    def check_ids(self, id1, id2):
        # check if the ids are in the correct format
        if not id1.startswith("NL.IMBAG.Pand."):
            id1 = f"NL.IMBAG.Pand.{id1}-0"
        if not id2.startswith("NL.IMBAG.Pand."):
            id2 = f"NL.IMBAG.Pand.{id2}-0"

        # check if the ids are in the geopandas dataframe
        if id1 not in self.prepared_gpdf["id"].values:
            raise ValueError(f"id1 {id1} not found in the geopandas dataframe")
        if self.prepared_gpdf_ref is None and id2 not in self.prepared_gpdf["id"].values:
            raise ValueError(f"id2 {id2} not found in the geopandas dataframe")
        # consider the reference geopandas dataframe if it is given, don't look at the original geopandas dataframe
        if self.prepared_gpdf_ref is not None and id2 not in self.prepared_gpdf_ref["id"].values:
            raise ValueError(f"id2 {id2} not found in the reference geopandas dataframe")
        
        # if all good, return the ids
        return id1, id2

    def calculate_distance(self, id1, id2):
        id1, id2 = self.check_ids(id1, id2)

        # for obj2 consider the reference geopandas dataframe if it is given, otherwise use the original geopandas dataframe
        obj1 = self.prepared_gpdf[self.prepared_gpdf["id"] == id1]
        if self.prepared_gpdf_ref is None:
            obj2 = self.prepared_gpdf[self.prepared_gpdf["id"] == id2]
        else:
            obj2 = self.prepared_gpdf_ref[self.prepared_gpdf_ref["id"] == id2]

        # calculate the euclidean distance between the two objects
        dist = euclidean_distances(obj1[self.columns], obj2[self.columns])
        return dist[0][0]
    
    def calculate_similarity(self, id1, id2):
        dist = self.calculate_distance(id1, id2)
        if self.column_weights is not None:
            normalized_dist = dist / sum(self.column_weights.values())
        else:
            normalized_dist = dist / len(self.columns)
        return 1 / (1 + normalized_dist)
    
    # function to calculate the distance matrix
    def mirror(self, matrix):
        upper_tri = np.triu(matrix)
        lower_tri = upper_tri.T
        mirrored_matrix = upper_tri + lower_tri
        return mirrored_matrix

    def save(self, matrix, output_path, header, fmt='%f'):
        if matrix.shape[0] == matrix.shape[1]:
            print("Matrix is square")
            matrix = self.mirror(matrix)
        
        np.savetxt(output_path, matrix, delimiter=",", fmt=fmt, header=header, comments='')
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

    def is_csv(self, output_path):
        if isinstance(output_path, str) and output_path.endswith('.csv') == False:
            raise ValueError("output_path must end with '.csv'")
        else:
            return True

    def update_matrix(self, row, matrix):
        if matrix.size == 0:
                matrix = row
        else:
            matrix = np.vstack((matrix, row))
        return matrix

    def distance_matrix_reference(self,
                                gpkg_ref: str = None,
                                output_path: str = None,
                                save_interval: int = 100):
        """ a distance matrix, but the x-axis are reference objects and the y-axis 
        are the objects that are compared to the reference objects. The y-axis objects
        are from the original geopandas dataframe, the x-axis objects are from the
        reference geopandas dataframe."""
        # TODO: split this & the regular distance matrix function into smaller functions, which can be reused
        
        # check the input variables
        self.is_csv(output_path)

        if gpkg_ref is None or not isinstance(gpkg_ref, str):
            raise ValueError("gpkg_ref must be a string containing the path to a geopackage file")
        
        # prepare the reference data
        self.gpdf_ref = gpd.read_file(gpkg_ref)
        self.prepared_gpdf_ref = self.prepare_data(self.gpdf_ref)

        # create an empty matrix and get all ids
        matrix = np.array([])
        reference_ids = self.prepared_gpdf_ref["id"].values
        all_ids = self.prepared_gpdf["id"].values

        fmt = '%s'
        header = 'id,' + ",".join(reference_ids)

        progress = tqdm(total=len(all_ids)*len(reference_ids), desc="Calculating distance matrix")
        # loop over all objects and calculate the distance to the reference objects
        for id1 in all_ids:
            row = np.array([id1])
            for id2 in reference_ids:
                dist = self.calculate_distance(id1, id2)
                row = np.append(row, round(dist, 5))
            matrix = self.update_matrix(row, matrix)
            
            progress.update(len(reference_ids)) # update the progress bar
            # save the matrix to a file if the interval is reached
            if isinstance(output_path, str) and matrix.ndim > 1 and matrix.shape[0] % save_interval == 0:
                self.save(matrix, output_path, header, '%s')

        # make sure the full matrix is saved, or just return the matrix
        if isinstance(output_path, str):
            self.save(matrix, output_path, header, fmt=fmt)
            print(f"Distance matrix calculated and saved to '{output_path}'")
        else:
            print("Distance matrix calculated")    
        return matrix

        
    def distance_matrix_regular(self, output_path: str = None, save_interval: int = 100, plot: bool = False):
        self.is_csv(output_path)

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
    path = "analysis/subset20k.gpkg"
    
    # sim = similarity(path, {'dispersion_index_2d': 1, 'dispersion_index_3d': 1})
    sim = similarity(path)

    # TODO: might want to change output_path to a path in the data/metrics folder
    sim.distance_matrix_reference(gpkg_ref='analysis/voorbeeldwoningen.gpkg', output_path='analysis/distance_matrix_reference.csv')
    # gpdf = gpd.read_file(path)