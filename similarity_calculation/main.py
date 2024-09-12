import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN, KMeans

from similarity_calculation import utils

class similarity:
    def __init__(self, feature_space_file: str, column_weights: dict = None, columns: list = None, normalize_columns: list = None, verbose: bool = False):
        """
        Initializes the SimilarityCalculator object which can be used to calculate the distance between two objects in the feature space data.
        This can be done for two object (calculate_distance), between all objects in the feature space data (distance_matrix) or the distance
        between a few reference buildings and a large set of buildings (distance_matrix_reference).

        Args:
            feature_space_file (str): The file path to the csv file with the feature space data.
            column_weights (dict, optional): A dictionary specifying the weights for the distance calculation of each column. Defaults to None.
            columns (list, optional): A list of column names to consider for the distance calculation, if used all columns bear the same weight. Defaults to None.
            normalize_columns (list, optional): A list of column names that should be normalized (standard scaler). Defaults to None.
            verbose (bool, optional): If True, print additional information. Defaults to False.
        """
        self.verbose = verbose
        self.feature_space_file = feature_space_file
        self.normalize_columns = normalize_columns
        
        # these are needed as we want to know which columns are relevant for the distance calculation, and if the columns should be weighted
        self._validate_input(column_weights, columns)
        self.column_weights = column_weights
        self._set_columns(columns)
        
    
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
        # if self.columns is None:
        #     self._set_columns(df.columns)

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
            self.columns.remove(column) # TODO: seems to run into an error, trying to remove Index if column_weights is not given
            if self.column_weights is not None:
                del self.column_weights[column]

        df = df[['id'] + self.columns]

        if self.normalize_columns is not None:
            normalized_df = self._normalize(df)
        else:
            normalized_df = df

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
            
            if column in self.normalize_columns:
                prepared_df[column] = (prepared_df[column] - prepared_df[column].mean()) / prepared_df[column].std()
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

    def _update_matrix(self, id1, other_ids, ref, n_zero_distances=0):
        # calculate the distance between the object and all other
        row = np.array([0] * n_zero_distances)
        for id2 in other_ids:
            dist = self.calculate_distance(id1, id2, ref)
            row = np.append(row, round(dist, 5))
        self.progress.update(len(other_ids))
        # update the matrix itself
        if hasattr(self, 'matrix'):
            self.matrix = np.vstack((self.matrix, row))
        else:
            self.matrix = row

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
        return 1 / (1 + dist)

    def distance_matrix_reference(self,
                                reference_feature_space: str,
                                dist_matrix_path: str,
                                save_interval: int = 100):
        """ a distance matrix, but the x-axis are reference objects and the y-axis 
        are the objects that are compared to the reference objects. The y-axis objects
        are from the original geopandas dataframe, the x-axis objects are from the
        reference geopandas dataframe."""
        utils.check_csv(dist_matrix_path)

        if hasattr(self, 'prepared_df') == False:
            self.prepared_df, self.prepared_df_ref = self._prepare_data(self.feature_space_file, reference_feature_space)
        # create an empty matrix and get all ids
        reference_ids = self.prepared_df_ref["id"].values
        formatted_ids = self.prepared_df["id"].values
        header = 'id,' + ",".join(reference_ids)

        self.progress = tqdm(total=len(formatted_ids)*len(reference_ids), desc="Calculating distance matrix")
        # loop over all objects and calculate the distance to the reference objects
        for id1 in formatted_ids:
            self._update_matrix(id1, reference_ids, ref=True)
            
            # save the matrix to a file if the interval is reached
            if isinstance(dist_matrix_path, str) and self.matrix.ndim > 1 and self.matrix.shape[0] % save_interval == 0:
                utils.save_matrix(self.matrix, dist_matrix_path, header, index=formatted_ids)
        self.progress.close()

        # make sure the full matrix is saved, or just return the matrix
        if isinstance(dist_matrix_path, str):
            utils.save_matrix(self.matrix, dist_matrix_path, header, index=formatted_ids)
            print(f'Distance matrix calculated and saved to "{dist_matrix_path}"')
        else:
            print("Distance matrix calculated")
        return self.matrix

    def distance_matrix_regular(self, dist_matrix_path: str, save_interval: int = 100, plot_matrix: bool = False):
        utils.check_csv(dist_matrix_path)
        
        if hasattr(self, 'prepared_df') == False:
            self.prepared_df = self._prepare_data(self.feature_space_file)

        # get all ids
        formatted_ids = self.prepared_df["id"].values
        header = 'id,' + ",".join(formatted_ids)

        # calculate the distance between all objects
        total_jobs = int(len(formatted_ids) * (len(formatted_ids) - 1) / 2)
        self.progress = tqdm(total=total_jobs, desc="Calculating distance matrix")
        for i, id1 in enumerate(formatted_ids):
            self._update_matrix(id1, formatted_ids[i+1:], ref=False, n_zero_distances=i+1)

            # save the matrix to a file if the interval is reached
            if isinstance(dist_matrix_path, str) and self.matrix.ndim > 1 and self.matrix.shape[0] % save_interval == 0:
                mirrored_matrix = utils.save_matrix(self.matrix, dist_matrix_path, header, index=formatted_ids[:i+1])
                #TODO: just append the new rows to the file instead of saving the whole matrix
        self.progress.close()
        
        # make sure the full matrix is saved
        mirrored_matrix = utils.save_matrix(self.matrix, dist_matrix_path, header, index=formatted_ids)
        print(f'INFO: Regular distance matrix calculated and saved to "{dist_matrix_path}"')
        
        if plot_matrix:
            utils.plot_matrix(mirrored_matrix, formatted_ids)

        return mirrored_matrix, formatted_ids

    def get_X(self, na_mode='mean'):
        """Get the feature space data as a pandas dataframe. The data is prepared if needed. Then N/A values are handled 
        in two ways: na_mode == mean imputes the mean of the column, na_mode == drop drops the row with the N/A value"""
        if na_mode not in ['mean', 'drop']:
            raise ValueError("na_mode must be either 'mean' or 'drop'")
        if hasattr(self, 'prepared_df') == False:
            self.prepared_df = self._prepare_data(self.feature_space_file)
        X = self.prepared_df.copy()

        # drop rows with NaN values and save the ids as they are not relevant for the clustering
        if na_mode == 'drop':
            X.dropna(axis='rows', inplace=True)
            # logging if needed
            if self.verbose and len(self.prepared_df) > len(X):
                print(f"INFO: Removed {len(self.prepared_df) - len(X)} rows with NaN values. Cannot compare / cluster buildings with NaN values")
        elif na_mode == 'mean':
            for column in self.columns:
                X.fillna(X[column].mean(), inplace=True)

        # get the ids so they can be used later, then drop them from the dataframe as we don't want to cluster on them
        ids = X['id']
        X.drop('id', axis=1, inplace=True)
        return X, ids

    def db_scan(self, eps=0.5, min_samples=5):
        X, ids = self.get_X(na_mode='mean')

        # perform the dbscan algorithm and add the cluster labels / identification to the dataframe
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X)
        results = pd.DataFrame({'id': ids, 'cluster': db.labels_})
        return results

    def k_means(self, k=5):
        X, ids = self.get_X(na_mode='mean')

        km = KMeans(n_clusters=k, random_state=12).fit(X)
        results = pd.DataFrame({'id': ids, 'cluster': km.labels_})
        return results, km
