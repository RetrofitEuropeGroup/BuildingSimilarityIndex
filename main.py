import collection
from processing import processing
from similarity_calculation import similarity

def run(all_ids: list):
    # Get the data from the collection
    collection.main(all_ids)

    # Process the data
    p = processing("collection/input")
    p.run()

    # # Calculate the similarity between the data
    sim = similarity(p.gpkg_path, ["roughness_index_3d", "actual_volume"])
    dist = sim.calculate_distance(all_ids[0], all_ids[1])
    print(f"distance between '{all_ids[0]}' and '{all_ids[1]}': {dist}")

if __name__ == '__main__':
    all_ids = ["0153100000203775", "0153100000277229", "0772100000262212",
               "0153100000213600", "0327100000255061", "0327100000258432",
                "0327100000252015", "0327100000264673", "0307100000377568",
                "0307100000326243", "0307100000337962", "0402100001519973"]
    run(all_ids)