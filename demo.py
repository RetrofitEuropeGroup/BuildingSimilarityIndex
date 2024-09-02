import json
import sys

if __name__ == '__main__':
    print("Running the BuildingSimilarity class")

    # important: the might be needed to change to the path of the directory where the BuildingSimilarity class is located, 
    # but with the current file structure it is just ".."

    path_to_building_similarity = r"." 
    sys.path.append(path_to_building_similarity)

    from BuildingSimilarity import BuildingSimilarity

    # Load the column weights
    with open('demo/column_weights_example.json', 'r') as f:
        column_weights = json.load(f)

    # Create the BuildingSimilarity object
    bs = BuildingSimilarity(bag_data_folder='data/amersfoort',
                            neighborhood_id="BU03071003",
                            verbose=True,
                            column_weights=column_weights,
                            feature_space_file='data/feature_space/fs_amersfoort.csv')


    # Collect the data & process it to a feature space
    bs.collection.collect_id_list()
    bs.processing.run()

    # run the dbscan algorithm
    result_dbscan = bs.similarity.db_scan(eps=0.5, min_samples=2)
    print(result_dbscan.head())