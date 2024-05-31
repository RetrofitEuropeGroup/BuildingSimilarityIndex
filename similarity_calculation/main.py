import geopandas as gpd
import numpy as np

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
    
    ## main functions
    def normalize(self):
        """normalize the columns in the geopandas dataframe so that every feature has the same weight in the distance calculation"""
        self.prepared_gpdf = self.gpdf.copy()
        for column in self.columns:
            self.prepared_gpdf[column] = (self.gpdf[column] - self.gpdf[column].mean()) / self.gpdf[column].std()
    
    def scale(self):
        """scale the columns in the geopandas dataframe according to the given weights, so that the features have different weights in the distance calculation"""
        for column, weight in self.column_weights.items():
            self.prepared_gpdf[column] = self.prepared_gpdf[column] * weight

    def prepare_data(self):
        self.normalize()
        if self.column_weights is not None:
            self.scale()

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
    
    def distance_matrix(self, output_path: str = None, save_interval: int = 100):
        # TODO: you can also make half the matrix and then mirror it
        if isinstance(output_path, str) and output_path.endswith('.csv') == False:
            raise ValueError("output_path must end with '.csv'")

        # create an empty matrix and get all ids
        all_ids = self.prepared_gpdf["id"].values
        header = ",".join(all_ids)
        matrix = np.array([])

        # calculate the distance between all objects
        for id1 in all_ids:
            row = np.array([])
            for id2 in all_ids:
                if id1 == id2:
                    dist = 0
                else:
                    dist = self.calculate_distance(id1, id2)
                row = np.append(row, round(dist, 5))
            
            # add row to matrix
            if matrix.size == 0:
                matrix = row
            else:
                matrix = np.vstack((matrix, row))
            
            # save the matrix to a file if the interval is reached
            if isinstance(output_path, str) and matrix.ndim > 1 and matrix.shape[0] % save_interval == 0:
                np.savetxt(output_path, matrix, delimiter=",", fmt='%f', header=header, comments='')
                #TODO: just append the new rows to the file instead of saving the whole matrix
        
        # make sure the full matrix is saved
        if isinstance(output_path, str):
            np.savetxt(output_path, matrix, delimiter=",", fmt='%f', header=header, comments='')
            print("Distance matrix calculated and saved to 'distance_matrix.csv'")
        else:
            print("Distance matrix calculated")
        
        return matrix, all_ids

if __name__ == '__main__':
    #TODO: check if everything works with the -0 / NL.IMBAG.Pand and without those
    path = "collection/output/merged.gpkg"

    # sim = similarity(path, {'dispersion_index_2d': 1, 'dispersion_index_3d': 1})
    sim = similarity(path)

    id1 = "NL.IMBAG.Pand.0327100000255061-0"
    id2 = "NL.IMBAG.Pand.0327100000264673-0"

    dist = sim.calculate_distance(id1, id2)
    print('Distance between the two objects is: ', dist)

    simi = sim.calculate_similarity(id1, id2)
    print('Similarity between the two objects is: ', simi)

    sim.distance_matrix(output_path="distance_matrix.csv", save_interval=5)