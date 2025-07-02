import csv
import math

def read_csv(file_name):
    data = []
    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def create_user_item_matrix(train_data):
    user_item_matrix = {}
    for row in train_data:
        user_id = int(row['userId'])
        movie_id = int(row['movieId'])
        rating = float(row['rating'])
        
        if user_id not in user_item_matrix:
            user_item_matrix[user_id] = {}
        
        user_item_matrix[user_id][movie_id] = rating
    return user_item_matrix


def cosine_similarity(user1_ratings, user2_ratings):

    common_movies = set(user1_ratings.keys()) & set(user2_ratings.keys())
    
    if not common_movies:
        return 0  

    numerator = sum(user1_ratings[movie] * user2_ratings[movie] for movie in common_movies)
    denominator = math.sqrt(sum(user1_ratings[movie]**2 for movie in common_movies)) * \
                  math.sqrt(sum(user2_ratings[movie]**2 for movie in common_movies))

    if denominator == 0:
        return 0 
    return numerator / denominator


def predict_rating(user_id, movie_id, user_item_matrix, similarity_matrix):

    weighted_sum = 0
    total_weight = 0

    for other_user_id, other_user_ratings in user_item_matrix.items():
        if movie_id in other_user_ratings:
            similarity = similarity_matrix.get((user_id, other_user_id), 0)
            weighted_sum += similarity * other_user_ratings[movie_id]
            total_weight += abs(similarity)

    if total_weight == 0:
        return 0  

    return weighted_sum / total_weight


def round_rating(predicted_rating):
    return round(predicted_rating * 2) / 2.0

train_data = read_csv('train.csv')
test_data = read_csv('test.csv')

user_item_matrix = create_user_item_matrix(train_data)

similarity_matrix = {}
for user1 in user_item_matrix:
    for user2 in user_item_matrix:
        if user1 < user2: 
            similarity = cosine_similarity(user_item_matrix[user1], user_item_matrix[user2])
            similarity_matrix[(user1, user2)] = similarity
            similarity_matrix[(user2, user1)] = similarity

predictions = []

for row in test_data:
    user_id = int(row['userId'])
    movie_id = int(row['movieId'])
    
    predicted_rating = predict_rating(user_id, movie_id, user_item_matrix, similarity_matrix)
    rounded_rating = round_rating(predicted_rating)

    predictions.append({'rId': row['rId'], 'rating': rounded_rating})

with open('collaborativePredictions.csv', 'w', newline='', encoding='utf-8') as file:
    fieldnames = ['rId', 'rating']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(predictions)

