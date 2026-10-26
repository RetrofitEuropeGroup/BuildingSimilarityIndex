import json
import os

import pandas as pd

from processing.metrics.cityStats import calculate_metrics
from processing.turning_functions.main import perform_turning_function


def _format_ids(cj_file):
    formatted_ids = []
    with open(cj_file, "r") as cj:
        cj_data = json.load(cj)
        for id in cj_data["CityObjects"]:
            if '-' in id:
                continue
            id = id.strip("NL.IMBAG.Pand.")
            formatted_ids.append(id)

    return formatted_ids


def _create_feature_space(tile_data_file, formatted_ids, output_file_path, verbose=False):
    """ Create a pd dataframe with the feature space. First step in the process is to calculate the 2d / 3d metrics on the merged cityjson which outputs a geo df, this is then used to execute the turning function"""
    n_processors = max(os.cpu_count() - 1, 1)

    # feature_space_metrics = calculate_metrics(jobs=n_processors, formatted_ids = _formatted_ids, verbose=_verbose)
    feature_space_metrics = calculate_metrics(input=tile_data_file, formatted_ids=formatted_ids, output_path=output_file_path, jobs=n_processors, verbose=verbose)

    # Turning function metrics
    # if hoogbouw:
    # feature_space_tf = perform_turning_function(feature_space_metrics, metric='l1')

    # merge the results in one data frame & return the df
    # feature_space_merged = feature_space_metrics.merge(feature_space_tf, on='id')
    # feature_space_merged.drop(columns=['geometry'], inplace=True) # drop the geometry column as it is a shapely object and cannot be saved to a csv file

    # for col in feature_space_merged.columns:
    #     if feature_space_merged[col].dtype == bool:
    #         feature_space_merged[col] = feature_space_merged[col].astype(int)
    # return feature_space_merged
    return feature_space_metrics


def main(tile_data_folder, feature_space_folder, exists_ok=False, verbose=False):
    # Create feature space
    if len(os.listdir(tile_data_folder)) == 0:
        raise Exception("The bag_data_folder is empty, please provide a folder with cityjson files")
    if not os.path.exists(feature_space_folder):
        os.mkdir(feature_space_folder)

    for file in os.listdir(tile_data_folder):
        if not file.endswith(".city.json"):
            continue

        feature_space_file = file.strip(".city.json") + ".csv"
        if exists_ok == False and os.path.exists(os.path.join(feature_space_folder, feature_space_file)):
            continue

        # calculate the 2d / 3d metrics and turning function features
        cj_file_path = os.path.join(tile_data_folder, file)
        formatted_ids = _format_ids(cj_file_path)
        feature_space = _create_feature_space(cj_file_path, 
                                              formatted_ids=formatted_ids, 
                                              output_file_path=feature_space_file,
                                              verbose=verbose)
        feature_space.drop(columns=['geometry'], inplace=True)
        # if _categorical_columns is not None: # if there are categorical columns, convert them to dummies (one hot encoding)
        #     feature_space = pd.get_dummies(feature_space, columns=_categorical_columns, dtype=int)

        feature_space.to_csv(os.path.join(feature_space_folder, feature_space_file))


if __name__ == "__main__":
    main("/home/nisfernos/data/tiles/", "/home/nisfernos/data/feature_spaces/")