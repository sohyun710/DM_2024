import random

def load_train_data(file_path):
    train_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",") 
        for line in f:
            values = line.strip().split(",")
            user_id, movie_id, *_, rating = values
            train_data.append({
                'userId': int(user_id),
                'movieId': int(movie_id),
                'rating': float(rating)
            })
    return train_data

def load_test_data(file_path):
    test_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            values = line.strip().split(",")
            rId = int(values[0])
            user_id = int(values[1])
            movie_id = int(values[2])
            test_data.append({
                'rId': rId,
                'userId': user_id,
                'movieId': movie_id
            })
    return test_data

def initialize_factors(num_users, num_movies, num_factors):
    user_factors = {u: [random.uniform(-0.5, 0.5) for _ in range(num_factors)] for u in range(1, num_users + 1)}
    movie_factors = {m: [random.uniform(-0.5, 0.5) for _ in range(num_factors)] for m in range(1, num_movies + 1)}
    return user_factors, movie_factors

def predict_rating(user_factors, movie_factors, user_bias, movie_bias, global_bias, user_id, movie_id):
    latent_pred = sum(u * m for u, m in zip(user_factors[user_id], movie_factors[movie_id]))
    return global_bias + user_bias[user_id] + movie_bias[movie_id] + latent_pred

def train_latent_factor_model_with_global_effect(train_data, num_factors=10, learning_rate=0.01, epochs=10):
    num_users = max(row['userId'] for row in train_data)
    num_movies = max(row['movieId'] for row in train_data)

    user_factors, movie_factors = initialize_factors(num_users, num_movies, num_factors)

    global_bias = sum(row['rating'] for row in train_data) / len(train_data)
    
    user_bias = {u: 0.0 for u in range(1, num_users + 1)}
    movie_bias = {m: 0.0 for m in range(1, num_movies + 1)}

    for epoch in range(epochs):
        for row in train_data:
            user_id = row['userId']
            movie_id = row['movieId']
            actual = row['rating']


            predicted = predict_rating(user_factors, movie_factors, user_bias, movie_bias, global_bias, user_id, movie_id)
            error = actual - predicted

            global_bias += learning_rate * error

            user_bias[user_id] += learning_rate * error
            movie_bias[movie_id] += learning_rate * error

            for k in range(num_factors):
                user_factors[user_id][k] += learning_rate * (error * movie_factors[movie_id][k])
                movie_factors[movie_id][k] += learning_rate * (error * user_factors[user_id][k])

    return user_factors, movie_factors, user_bias, movie_bias, global_bias


def round_rating(rating):
    return 0.5 * round(rating * 2)

def write_predictions_to_csv(file_path, predictions):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("rId,rating\n")
        for rId, rating in predictions:
            f.write(f"{rId},{rating:.1f}\n") 


if __name__ == "__main__":

    train_file_path = "train.csv"
    train_data = load_train_data(train_file_path)

    user_factors, movie_factors, user_bias, movie_bias, global_bias = train_latent_factor_model_with_global_effect(train_data)

    test_file_path = "test.csv"
    test_data = load_test_data(test_file_path)

    predictions = []
    for row in test_data:
        rId = row['rId']
        user_id = row['userId']
        movie_id = row['movieId']
        predicted_rating = predict_rating(user_factors, movie_factors, user_bias, movie_bias, global_bias, user_id, movie_id)

        clamped_rating = max(1.0, min(5.0, round_rating(predicted_rating))) 
        predictions.append((rId, clamped_rating))

    output_file_path = "latentPredictions2.csv"
    write_predictions_to_csv(output_file_path, predictions)

