import collection
from processing import processing
import similarity_calculation


if __name__ == '__main__':
    # Get the data from the collection
    all_ids = ["0153100000203775", "0153100000277229", "0772100000262212",
               "0153100000213600", "0327100000255061", "0327100000258432",
                "0327100000252015", "0327100000264673", "0307100000377568",
                "0307100000326243", "0307100000337962", "0402100001519973"]
    collection.main(all_ids)

    # Process the data
    p = processing("collection/input")
    p.run()

    # # Calculate the similarity between the data
    # similarity = similarity_calculation.calculate_similarity(data)

    # # Print the similarity
    # print(similarity.head())