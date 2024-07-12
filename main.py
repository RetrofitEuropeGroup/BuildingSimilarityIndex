from collection import collection
from processing import processing
from similarity_calculation import similarity

def run(all_ids: list):
    # Get the data from the collection
    c = collection()
    c.collect_id_list(all_ids)

    # Process the data
    # input_file = "C:\Users\TimoScheidel\OneDrive - HAN\Future Factory\FF_BuildingSimilarityIndex\analysis\subset20k.city.json"
    # p = processing(input_file=input_file, output_file="subset20k.gpkg")
    p = processing(gpkg_path="data/gpkg/test.gpkg", bag_data_folder="data/bag_data")
    p.run()

    # Calculate the similarity between the data
    s = similarity(p.gpkg_path, columns= ["roughness_index_3d", "actual_volume"])
    dist = s.calculate_distance(all_ids[0], all_ids[1])
    # print(f"distance between '{all_ids[0]}' and '{all_ids[1]}': {dist}")

if __name__ == '__main__':
    # example of the use of run
    all_ids = ["0153100000203775", "0153100000277229", "0772100000262212",
               "0153100000213600", "0327100000255061", "0327100000258432",
                "0327100000252015", "0327100000264673", "0307100000377568",
                "0307100000326243", "0307100000337962", "0402100001519973"]
    run(all_ids)