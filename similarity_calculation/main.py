import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans

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
        self.raw_df = pd.read_csv(self.feature_space_file, dtype={'id': str})
        
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
        elif columns is not None:  # if columns is given, remove the id column
            self.columns = columns            
        else:
            self.columns = pd.read_csv(self.feature_space_file, dtype={'id': str}).columns.tolist() # TODO: load the fs in init and use it here

        if "id" in self.columns: # ID can never be used for distance calculation
            self.columns.remove("id")

    def remove_na_column(self, column):
        self.columns.remove(column)
        if self.column_weights is not None:
            del self.column_weights[column]

    def check_na_columns(self):
        for column in self.columns:
            if column not in self.raw_df:
                self.remove_na_column(column)
                if self.verbose:
                    print(f'WARNING: column {column} was in column(_weight)s but is not in df, removing it')
            elif max(self.raw_df[column]) == min(self.raw_df[column]):
                self.remove_na_column(column)
                if self.verbose:
                    print(f'WARNING: column {column} has only one value, removing it')

    # functions to prepare the data for the distance calculation
    def _prepare_data(self):
        # removing the columns from self.columns if they are not in the df or if they have 1 value
        self.check_na_columns()

        # select the columns that are needed for the distance calculation
        self.sub_df = self.raw_df[['id'] + self.columns]

        if self.normalize_columns is not None:
            self.sub_df = self._normalize(self.sub_df)

        # weighted columns only if needed (if column_weights is given)
        if self.column_weights is not None:
            self.prepared_df = self._weighted_columns(self.sub_df)
        else:
            self.prepared_df = self.sub_df

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
        if id1 not in self.ids.values:
            raise ValueError(f"id1 {id1} not found in the dataframe")

        # consider the reference dataframe if it is given, don't look at the original geopandas dataframe
        if id2 not in self.ids.values:
            raise ValueError(f"id2 ({id2}) not found in the geopandas dataframe")

        # if all good, return the ids
        return id1, id2

    def _update_matrix(self, id1, other_ids, n_zero_distances=0):
        # calculate the distance between the object and all other
        row = np.array([0] * n_zero_distances)
        for id2 in other_ids:
            dist = self.calculate_distance(id1, id2)
            row = np.append(row, round(dist, 5))

        # update the matrix itself, else statement is only for the first row
        if hasattr(self, 'matrix'):
            self.matrix = np.vstack((self.matrix, row))
        else:
            self.matrix = row
        
        self.progress.update(len(other_ids))

    def calculate_distance(self, id1, id2):
        if hasattr(self, 'X') == False:
            raise ValueError("The feature space data has not been prepared yet. Please run the set_X() function first.")
        id1, id2 = self._check_ids(id1, id2)

        # for obj2 consider the reference geopandas dataframe if it is given, otherwise use the original geopandas dataframe
        obj1 = self.X.iloc[list(self.ids).index(id1)] # TODO: shouldn't we change self.ids to a list?
        obj2 = self.X.iloc[list(self.ids).index(id2)]
        obj1_fs = obj1[self.columns].array.reshape(1, -1)
        obj2_fs = obj2[self.columns].array.reshape(1, -1)

        # calculate the euclidean distance between the two objects
        dist = euclidean_distances(obj1_fs, obj2_fs)
        return dist[0][0]

    def calculate_similarity(self, id1, id2):
        dist = self.calculate_distance(id1, id2)
        return 1 / (1 + dist)

    def distance_matrix_reference(self,
                                reference_ids: list,
                                dist_matrix_path: str,
                                save_interval: int = 100,
                                plot_matrix: bool = False):
        """ a distance matrix, but the x-axis are reference objects and the y-axis 
        are the objects that are compared to the reference objects. The y-axis objects
        are from the original geopandas dataframe, the x-axis objects are from the
        reference geopandas dataframe."""

        utils.check_csv(dist_matrix_path) # check if the path is a csv file

        # get all ids & prepare the data
        self.set_X()
        reference_ids = utils.format_ids(reference_ids)
        regular_ids = self.ids[~self.ids.isin(reference_ids)] # remove the reference ids from the regular ids
        header = 'id,' + ",".join(reference_ids)

        # loop over all objects and calculate the distance to the reference objects
        self.progress = tqdm(total=len(regular_ids)*len(reference_ids), desc="Calculating distance matrix")
        for id1 in regular_ids:
            self._update_matrix(id1, reference_ids)
            
            # save the matrix to a file if the interval is reached
            if isinstance(dist_matrix_path, str) and self.matrix.ndim > 1 and self.matrix.shape[0] % save_interval == 0:
                utils.save_matrix(self.matrix, dist_matrix_path, header, index=regular_ids)
        self.progress.close()

        # make sure the full matrix is saved, or just return the matrix
        utils.save_matrix(self.matrix, dist_matrix_path, header, index=regular_ids)
        if self.verbose:
            print(f'Distance matrix calculated and saved to "{dist_matrix_path}"')

        if plot_matrix:
            utils.plot_matrix(self.matrix, regular_ids, reference_ids)
        return self.matrix

    def distance_matrix_regular(self, dist_matrix_path: str, save_interval: int = 100, plot_matrix: bool = False):
        utils.check_csv(dist_matrix_path)
        
        # get all ids & prepare the data
        self.set_X()
        header = 'id,' + ",".join(self.ids)

        # calculate the distance between all objects
        total_jobs = int(len(self.ids) * (len(self.ids) - 1) / 2)
        self.progress = tqdm(total=total_jobs, desc="Calculating distance matrix")
        for i, id1 in enumerate(self.ids):
            self._update_matrix(id1, self.ids[i+1:], n_zero_distances=i+1)

            # save the matrix to a file if the interval is reached
            if isinstance(dist_matrix_path, str) and self.matrix.ndim > 1 and self.matrix.shape[0] % save_interval == 0:
                mirrored_matrix = utils.save_matrix(self.matrix, dist_matrix_path, header, index=self.ids[:i+1])
                #TODO: just append the new rows to the file instead of saving the whole matrix
        self.progress.close()
        
        # make sure the full matrix is saved
        mirrored_matrix = utils.save_matrix(self.matrix, dist_matrix_path, header, index=self.ids)
        if self.verbose:
            print(f'INFO: Regular distance matrix calculated and saved to "{dist_matrix_path}"')
        
        if plot_matrix:
            utils.plot_matrix(mirrored_matrix, self.ids)
        return mirrored_matrix

    def handle_na(self, na_mode='mean'):
        if na_mode == 'drop':
            self.X.dropna(axis='rows', inplace=True)
            # logging if needed
            if self.verbose and len(self.prepared_df) > len(self.X):
                print(f"INFO: Removed {len(self.prepared_df) - len(self.X)} rows with NaN values. Cannot compare / cluster buildings with NaN values")
        # fill the NaN values with the mean of the column or a zero. Also log the columns with missing values and the number of N/A values
        elif na_mode in ['mean', 'zero']:
            na_values, na_columns = 0, []
            for column in self.columns:
                if self.X[column].isna().sum()>0:
                    na_values += self.X[column].isna().sum()
                    na_columns.append(column)
                    if na_mode == 'mean':
                        self.X.fillna(self.X[column].mean(), inplace=True)
                    elif na_mode == 'zero':
                        self.X.fillna(0, inplace=True)
            # logging if needed
            if self.verbose and len(na_columns) > 0 and na_mode == 'mean':
                print(f"INFO: Filled {na_values} NaN values with the mean. Columns with missing values are: {', '.join(na_columns)}")
            elif self.verbose and len(na_columns) > 0 and na_mode == 'zero':
                print(f"INFO: Filled {na_values} NaN values with zero. Columns with missing values are: {', '.join(na_columns)}")

    def set_X(self, na_mode='mean'):
        """
        Get the feature space data as a pandas dataframe. The data is prepared if needed. Then N/A values are handled in three ways:
            mean - imputes the mean of the column
            drop - drops the row with the N/A value
            zero - fills the N/A value with zero
        """

        if na_mode not in ['mean', 'drop', 'zero']:
            raise ValueError("na_mode must be either 'mean' or 'drop'")
        if hasattr(self, 'prepared_df') == False:
            self._prepare_data()
        self.X = self.prepared_df.copy()

        # drop rows with NaN values and save the ids as they are not relevant for the clustering
        self.handle_na(na_mode)

        # get the ids so they can be used later, then drop them from the dataframe as we don't want to cluster on them
        self.ids = self.X['id']
        self.X.drop('id', axis=1, inplace=True)
        
        return self.X, self.ids

    def db_scan(self, eps=0.5, min_samples=5, na_mode='mean'):
        X, ids = self.set_X(na_mode)

        # perform the dbscan algorithm and add the cluster labels / identification to the dataframe
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X)
        results = pd.DataFrame({'id': ids, 'cluster': db.labels_})
        return results

    def hdb_scan(self, min_cluster_size=5, na_mode='mean'):
        X, ids = self.set_X(na_mode)

        # perform the dbscan algorithm and add the cluster labels / identification to the dataframe
        db = HDBSCAN(min_cluster_size=min_cluster_size, n_jobs=-1).fit(X)
        results = pd.DataFrame({'id': ids, 'cluster': db.labels_})
        return results

    def k_means(self, k=5, na_mode='mean'):
        X, ids = self.set_X(na_mode)

        km = KMeans(n_clusters=k, random_state=12).fit(X)
        results = pd.DataFrame({'id': ids, 'cluster': km.labels_})
        return results, km
