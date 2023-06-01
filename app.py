import streamlit as st
import pandas as pd
import numpy as np
import random
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import SVDpp

# Load your dataset and perform all necessary dataframe processing steps
df_merged = pd.read_csv('./data/collab_merged.csv')
df_meta = df_merged.drop(columns=['reviews'], axis=1)
df = df_merged.drop(columns=['reviews', 'genre', 'description', 'title', 'starring'], axis=1)

# Instantiate the Reader object and load your dataframe into a Surprise Dataset object
reader = Reader()
data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)

# Split your dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train your SVDpp model using the training set
SVDpp_one = SVDpp(n_factors=5, n_epochs=20, cache_ratings=True, random_state=42)
SVDpp_one.fit(trainset)


def recommend_movies(trained_model, movie_df, N=5, user_id=None):
    
    # If no user_id provided, default to a random user_id for evaluations
    if user_id is None:
        all_user_ids = movie_df['user_id'].unique().tolist()
        user_id = random.choice(all_user_ids)

    # Get user's watched movies and all available movies
    user_movies = movie_df[movie_df['user_id'] == user_id]['movie_id'].tolist()
    all_movies = movie_df['movie_id'].tolist()
    
    # Determine the set of unseen movies
    unseen_movies = set(all_movies) - set(user_movies)

    # Predict ratings for unseen movies
    predictions = []
    for movie_id in unseen_movies:
        predicted_rating = trained_model.predict(user_id, movie_id).est
        predictions.append({'movie_id': movie_id, 'predicted_rating': predicted_rating})
    
    # Create a DataFrame with predicted ratings
    predictions_df = pd.DataFrame(predictions)

    # Get the top N movies with the highest predicted ratings
    top_N = predictions_df.sort_values('predicted_rating', ascending=False).head(N)
    top_N_movie_ids = top_N['movie_id'].tolist()

    # Get movie details of the top N movies
    top_N_movies = movie_df[movie_df['movie_id'].isin(top_N_movie_ids)]
    top_N_movies.drop_duplicates(subset=['movie_id'], inplace=True)

    # Merge movie details with predicted ratings
    top_N_ratings = pd.merge(top_N, top_N_movies, on='movie_id')
    
    # Return top N movie recommendations without unnecessary columns
    return top_N_ratings.drop(columns=['user_id', 'rating', 'movie_id'], axis=1)


st.title("Movie Recommendation App")

# Dropdown for user_id selection
user_ids = df_merged['user_id'].unique().tolist()
selected_user_id = st.selectbox('Select a user_id:', user_ids, index=0)

def top_rated_movies(trained_model, movie_df, N=5, user_id=None):
    user_movie_ratings = movie_df[movie_df['user_id'] == user_id].sort_values('rating', ascending=False)
    top_N_rated_movies = user_movie_ratings.head(N)
    return top_N_rated_movies.drop(columns=['user_id', 'rating', 'movie_id'], axis=1)

def top_rated_movies(trained_model, movie_df, N=5, user_id=None):
    user_movie_ratings = movie_df[movie_df['user_id'] == user_id].sort_values('rating', ascending=False)
    top_N_rated_movies = user_movie_ratings.head(N)
    return top_N_rated_movies.drop(columns=['user_id', 'rating', 'movie_id'], axis=1)

if st.button("Recommend Movies", key="rec_movies_button"):
    recommendations = recommend_movies(SVDpp_one, df_merged, N=5, user_id=selected_user_id)
    top_rated = top_rated_movies(SVDpp_one, df_merged, N=5, user_id=selected_user_id)

    # Display top rated movies in a table
    st.subheader("Top Rated Movies")
    st.write(top_rated)

    # Display recommendations in a table
    st.subheader("Movie Recommendations")
    st.write(recommendations)
