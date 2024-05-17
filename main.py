import collection
import processing
import similarity_calculation


if __name__ == '__main__':
    # Get the data from the collection
    print(dir(collection))
    data = collection.get_data()

    # Process the data
    data = processing.process_data(data)

    # Calculate the similarity between the data
    similarity = similarity_calculation.calculate_similarity(data)

    # Print the similarity
    print(similarity.head())