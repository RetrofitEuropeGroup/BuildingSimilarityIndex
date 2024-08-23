import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances


class similarity:
    def __init__(self, feature_space_file: str, column_weights: dict = None, columns: list = None):
        """
        Initializes the SimilarityCalculator object which can be used to calculate the distance between two objects in the feature space data.
        This can be done for two object (calculate_distance), between all objects in the feature space data (distance_matrix) or the distance
        between a few reference buildings and a large set of buildings (distance_matrix_reference).

        Args:
            feature_space_file (str): The file path to the csv file with the feature space data.
            column_weights (dict, optional): A dictionary specifying the weights for the distance calculation of each column. Defaults to None.
            columns (list, optional): A list of column names to consider for the distance calculation, if used all columns bear the same weight. Defaults to None.
        """
        # needed to know which columns are relevant for the distance calculation, and if the columns should be weighted
        self._validate_input(column_weights, columns)
        self.column_weights = column_weights
        self.columns = columns

        self.feature_space_file = feature_space_file
    
    ## helper functions for __init__
    def _validate_input(self, column_weights, columns):
        if column_weights is not None and not isinstance(column_weights, dict):
            raise ValueError("column_weights must be a dictionary")
        if columns is not None and not isinstance(columns, list):
            raise ValueError("columns must be a list")

    def _set_columns(self, columns=None):
        if self.column_weights is not None:  # overwrite columns if column_weights is given
            self.columns = list(self.column_weights.keys())
        elif columns is not None and 'id' in columns:  # if columns is given, remove the id column
            self.columns = columns.drop("id")
        elif columns is not None:
            self.columns = columns
        
        if self.columns is None: # if columns is still None, raise an error
            raise ValueError("Columns must be given if column_weights is not given")

    # functions to prepare the data for the distance calculation
    def _prepare_data(self, feature_space_file, feature_space_ref_file = None):
        df = pd.read_csv(feature_space_file)
        if feature_space_ref_file is not None:
            df_ref = pd.read_csv(feature_space_ref_file)
            ref_ids = df_ref['id']
            df = pd.concat([df, df_ref])
        if self.columns is None:
            self._set_columns(df.columns)

        # removing the columns from self.columns if they are not in the df
        na_cols = []
        for column in self.columns:
            if column not in df:
                print(f'WARNING: column {column} was in column(_weights) but is not in df, removing it')
                na_cols.append(column)
            if max(df[column]) == min(df[column]):
                print(f'WARNING: column {column} has only one value, removing it')
                na_cols.append(column)

        for column in na_cols:
            self.columns.remove(column)
            if self.column_weights is not None:
                del self.column_weights[column]
        
        normalized_df = self._normalize(df)

        # weighted columns only if needed (if column_weights is given)
        if self.column_weights is not None:
            prepared_df = self._weighted_columns(normalized_df)
        else:
            prepared_df = normalized_df
        
        if feature_space_ref_file is not None:
            prepared_df_ref = prepared_df[prepared_df['id'].isin(ref_ids)]
            prepared_df = prepared_df[~prepared_df['id'].isin(ref_ids)]
            return prepared_df, prepared_df_ref
        else:
            return prepared_df

    def _normalize(self, df):
        """normalize the columns in the geopandas dataframe so that every feature has the same weight in the distance calculation"""
        prepared_df = df.copy()
        
        for column in self.columns:
            if df[column].dtype != "float64":
                prepared_df[column] = prepared_df[column].astype(float)
            try:
                prepared_df[column] = (prepared_df[column] - prepared_df[column].mean()) / prepared_df[column].std()
            except Exception as e:
                raise ValueError(f"Could not normalize column: {column}. Error: {e}")
        return prepared_df
    
    def _weighted_columns(self, df):
        """make the columns weighted by multiplying the values with the weights. This will result in some features having a larger impact on distance calculation"""
        weighted_df = df.copy()
        for column, weight in self.column_weights.items():
            weighted_df[column] = df[column] * weight
        return weighted_df

    ## main functions
    # function to calculate the distance between two objects
    def _check_ids(self, id1, id2):
        # check if the ids are in the correct format
        if not id1.startswith("NL.IMBAG.Pand."):
            id1 = f"NL.IMBAG.Pand.{id1}-0"
        if not id2.startswith("NL.IMBAG.Pand."):
            id2 = f"NL.IMBAG.Pand.{id2}-0"

        # check if the ids are in the dataframe
        if id1 not in self.prepared_df["id"].values:
            raise ValueError(f"id1 {id1} not found in the dataframe")

        # consider the reference dataframe if it is given, don't look at the original geopandas dataframe
        if hasattr(self, "prepared_df_ref") and id2 not in self.prepared_df_ref["id"].values:
            raise ValueError(f"id2 {id2} not found in the reference geopandas dataframe")
        if not hasattr(self, "prepared_df_ref") and id2 not in self.prepared_df["id"].values:
            raise ValueError(f"id2 {id2} not found in the dataframe")

        # if all good, return the ids
        return id1, id2
    
    # function to calculate the distance matrix
    def _mirror(self, matrix):
        upper_tri = np.triu(matrix)
        lower_tri = upper_tri.T
        mirrored_matrix = upper_tri + lower_tri
        return mirrored_matrix

    def save(self, matrix, path, header, fmt='%f'):
        if matrix.shape[0] == matrix.shape[1]:
            matrix = self._mirror(matrix)
        
        parent_dir = os.path.dirname(path)
        if os.path.isdir(parent_dir) == False:
            os.mkdir(parent_dir)
        np.savetxt(path, matrix, delimiter=",", fmt=fmt, header=header, comments='')
        return matrix

    def _check_csv(self, path):
        if isinstance(path, str) and path.endswith('.csv') == False:
            raise ValueError("the path must end with '.csv'")

    def _update_matrix(self, row, matrix):
        if matrix.size == 0:
                matrix = row
        else:
            matrix = np.vstack((matrix, row))
        return matrix

    def plot_matrix(self, matrix, all_ids):
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

    def calculate_distance(self, id1, id2, ref=False):
        if hasattr(self, 'prepared_df') == False:
            self.prepared_df = self._prepare_data(self.feature_space_file)

        id1, id2 = self._check_ids(id1, id2)

        # for obj2 consider the reference geopandas dataframe if it is given, otherwise use the original geopandas dataframe
        obj1 = self.prepared_df[self.prepared_df["id"] == id1]
        if ref == False:
            obj2 = self.prepared_df[self.prepared_df["id"] == id2]
        else:
            obj2 = self.prepared_df_ref[self.prepared_df_ref["id"] == id2]
        # calculate the euclidean distance between the two objects
        dist = euclidean_distances(obj1[self.columns], obj2[self.columns])
        return dist[0][0]

    def calculate_similarity(self, id1, id2):
        dist = self.calculate_distance(id1, id2)
        # if self.column_weights is not None:
        #     normalized_dist = dist / sum(self.column_weights.values())
        # else:
        #     normalized_dist = dist / len(self.columns)
        return 1 / (1 + dist)

    def distance_matrix_reference(self,
                                reference_feature_space: str,
                                dist_matrix_path: str = None,
                                save_interval: int = 100):
        """ a distance matrix, but the x-axis are reference objects and the y-axis 
        are the objects that are compared to the reference objects. The y-axis objects
        are from the original geopandas dataframe, the x-axis objects are from the
        reference geopandas dataframe."""
        # TODO: split this & the regular distance matrix function into smaller functions, which can be reused
        self._check_csv(dist_matrix_path)

        if hasattr(self, 'prepared_df') == False:
            self.prepared_df, self.prepared_df_ref = self._prepare_data(self.feature_space_file, reference_feature_space)
        # create an empty matrix and get all ids
        matrix = np.array([])
        reference_ids = self.prepared_df_ref["id"].values
        all_ids = self.prepared_df["id"].values

        fmt = '%s'
        header = 'id,' + ",".join(reference_ids)

        progress = tqdm(total=len(all_ids)*len(reference_ids), desc="Calculating distance matrix")
        # loop over all objects and calculate the distance to the reference objects
        for id1 in all_ids:
            row = np.array([id1])
            for id2 in reference_ids:
                dist = self.calculate_distance(id1, id2, ref=True)
                row = np.append(row, round(dist, 5))
            matrix = self._update_matrix(row, matrix)
            
            progress.update(len(reference_ids)) # update the progress bar
            # save the matrix to a file if the interval is reached
            if isinstance(dist_matrix_path, str) and matrix.ndim > 1 and matrix.shape[0] % save_interval == 0:
                self.save(matrix, dist_matrix_path, header, '%s')
        progress.close()

        # make sure the full matrix is saved, or just return the matrix
        if isinstance(dist_matrix_path, str):
            self.save(matrix, dist_matrix_path, header, fmt=fmt)
            print(f"Distance matrix calculated and saved to '{dist_matrix_path}'")
        else:
            print("Distance matrix calculated")    
        return matrix

    def distance_matrix_regular(self, dist_matrix_path: str = None, save_interval: int = 100, plot_matrix: bool = False):
        self._check_csv(dist_matrix_path)
        
        if hasattr(self, 'prepared_df') == False:
            self.prepared_df = self._prepare_data(self.feature_space_file)

        # create an empty matrix and get all ids
        all_ids = self.prepared_df["id"].values
        header = ",".join(all_ids)
        matrix = np.array([])

        # calculate the distance between all objects
        total_jobs = int(len(all_ids) * (len(all_ids) - 1) / 2)
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
            if isinstance(dist_matrix_path, str) and matrix.ndim > 1 and matrix.shape[0] % save_interval == 0:
                self.save(matrix, dist_matrix_path, header)
                #TODO: just append the new rows to the file instead of saving the whole matrix
        progress.close()

        # make sure the full matrix is saved
        if isinstance(dist_matrix_path, str):
            mirrored_matrix = self.save(matrix, dist_matrix_path, header)
            print("Distance matrix calculated and saved to 'distance_matrix.csv'")
        else:
            mirrored_matrix = self._mirror(matrix)
            print("Distance matrix calculated")
        
        if plot_matrix:
            self.plot_matrix(mirrored_matrix, all_ids)

        return mirrored_matrix, all_ids

if __name__ == '__main__':
    #TODO prio: check if everything works with the -0 / NL.IMBAG.Pand and without those

    sim = similarity("data/feature_space/feature_space.csv")
    # sim.distance_matrix_regular(plot_matrix=True)
    sim.distance_matrix_reference("data/feature_space/feature_space copy.csv", 'test.csv')