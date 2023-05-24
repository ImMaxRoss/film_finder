import pandas as pd
import numpy as np
import re
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import train_test_split
import random

def create_user_profile(df_movies):
    user_ratings = []
    total_ratings = 0
    
    while total_ratings < 5:
        random_movie = df_movies.sample()
        print(f"Movie {total_ratings + 1}: {random_movie['title'].values[0]}")
        print(f"Description: {random_movie['description'].values[0]}")
        print(f"Genre: {random_movie['genre'].values[0]}\n")
        
        movie_rating = float(input(f"Enter rating (0-5) for movie {total_ratings + 1}: "))
        
        while (movie_rating < 0) or (movie_rating > 5) or not (isinstance(movie_rating, int) or isinstance(movie_rating, float)):
            print("Rating must be between 0-5 and be a numeric value.")
            movie_rating = float(input(f"Enter rating (0-5) for movie {total_ratings + 1}: "))
        
        if movie_rating != 0:
            user_ratings.append((random_movie['movie_id'].values[0], movie_rating))
            total_ratings += 1
        else:
            print("You have not seen this movie. Moving to the next random movie.\n")

    return pd.DataFrame(user_ratings, columns=['movie_id', 'rating'])

def add_new_user(df, new_user_ratings, user_id):
    new_user_ratings['user_id'] = user_id
    return pd.concat([df, new_user_ratings], ignore_index=True)

def recommend_movies(user_id, algo, data):
    all_movie_ids = data['movie_id'].unique()
    rated_movie_ids = data[data['user_id'] == user_id]['movie_id']
    unrated_movie_ids = np.setdiff1d(all_movie_ids, rated_movie_ids)
    
    predictions = list(map(lambda movie_id: algo.predict(user_id, movie_id), unrated_movie_ids))
    recommendations = pd.DataFrame(predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details']).sort_values(by='est', ascending=False).head(10)
    movie_ids = recommendations['iid']
    recommendations = recommendations.merge(data[['movie_id', 'title', 'genre']].drop_duplicates(), left_on='iid', right_on='movie_id').drop(['movie_id'], axis=1)
    recommendations.rename(columns={'iid': 'movie_id', 'est': 'predicted_rating'}, inplace=True)
    return recommendations

def main():
    df_movies = pd.read_csv('./data/meta_allvid_clean.csv', encoding='utf-8')
    df_movies.drop_duplicates(subset='movie_id', inplace=True)
    df_movies.dropna(subset=['genre', 'description'], inplace=True)

    new_user_ratings = create_user_profile(df_movies)
    
    merged_df = pd.read_csv('./data/collab_model_df.csv', encoding='utf-8')
    
    # Remove rows with non-integer user_id values
    merged_df = merged_df[pd.to_numeric(merged_df['user_id'], errors='coerce').notnull()]

    # Convert user_id to integer
    merged_df['user_id'] = merged_df['user_id'].astype(int)

    # Set new_user_id
    if merged_df.empty:
        new_user_id = 1
    else:
        new_user_id = max(merged_df['user_id']) + 1

    updated_df = add_new_user(merged_df, new_user_ratings, new_user_id)
    
    reader = Reader()
    data = Dataset.load_from_df(updated_df[['user_id', 'movie_id', 'rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    SVDpp_final = SVDpp()
    SVDpp_final.fit(trainset)
    
    genre = input("Enter the genre you're interested in: ")

    recommendations = recommend_movies(new_user_id, SVDpp_final, merged_df)
    recommendations = recommendations[recommendations['genre'].str.contains(genre)]
    
    if recommendations.empty:
        print("No recommendations available in the specified genre.")
    else:
        print("Recommended movies in the specified genre:")
        print(recommendations[['movie_id', 'title', 'predicted_rating']])