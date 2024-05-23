import collection
from processing import processing
import similarity_calculation


if __name__ == '__main__':
    # Get the data from the collection
    all_ids = ["NL.IMBAG.Pand.0202100000238878", "NL.IMBAG.Pand.0202100000206918"]
    collection.main(all_ids)

    # Process the data
    p = processing("collection/input")
    p.run()

    # # Calculate the similarity between the data
    # similarity = similarity_calculation.calculate_similarity(data)

    # # Print the similarity
    # print(similarity.head())