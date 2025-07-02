import multiprocessing
import csv
import math

def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def compute_genre_similarity(genres1, genres2):
    set1, set2 = set(genres1.split('|')), set(genres2.split('|'))
    intersection = set1 & set2
    return len(intersection) / math.sqrt(len(set1) * len(set2)) if len(set1) and len(set2) else 0

def predict_rating_parallel(train_data, test_row, similarity_cache, top_n=10):
    similarities = []
    for train_row in train_data:
        pair = (test_row['genres'], train_row['genres'])
        
        if pair not in similarity_cache:
            sim = compute_genre_similarity(test_row['genres'], train_row['genres'])
            similarity_cache[pair] = sim
        else:
            sim = similarity_cache[pair]
        
        similarities.append((sim, float(train_row['rating'])))

    similarities.sort(reverse=True, key=lambda x: x[0])
    top_similarities = similarities[:top_n]

    if top_similarities:
        numerator = sum(sim * rating for sim, rating in top_similarities)
        denominator = sum(sim for sim, _ in top_similarities)
        predicted_rating = numerator / denominator if denominator > 0 else 3.0
    else:
        predicted_rating = 3.0
    
    clamped_rating = max(1.0, min(5.0, round(2 * predicted_rating) / 2))
    return test_row['rId'], clamped_rating

def predict_ratings_using_multiprocessing(train_data, test_data, similarity_cache, top_n=10):
    with multiprocessing.Pool() as pool:
        predictions = pool.starmap(
            predict_rating_parallel, 
            [(train_data, test_row, similarity_cache, top_n) for test_row in test_data]
        )
    return predictions

def write_predictions_to_csv(file_path, predictions):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("rId,rating\n")  
        for rId, rating in predictions:
            f.write(f"{rId},{rating:.1f}\n") 


if __name__ == "__main__":

    train_file_path = 'train.csv'
    test_file_path = 'test.csv'
    
    train_data = load_data(train_file_path)
    test_data = load_data(test_file_path)
    
    similarity_cache = {} 
    
    predictions = predict_ratings_using_multiprocessing(train_data, test_data, similarity_cache)
    
    output_file_path = "contentbasedPredictions.csv"
    write_predictions_to_csv(output_file_path, predictions)
