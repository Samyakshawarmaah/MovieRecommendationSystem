import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load datasets
ratings = pd.read_csv('ratings.csv')  # UserID, MovieID, Rating, Timestamp
movies = pd.read_csv('movies.csv')   # MovieID, Title, Genres

# Content-Based Filtering
# Create a matrix of genres
count = CountVectorizer()
genre_matrix = count.fit_transform(movies['genres'])

# Compute cosine similarity
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Recommend movies based on similarity
def recommend_content(movie_title, cosine_sim, movies_df):
    try:
        idx = movies_df[movies_df['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_movies = [movies_df['title'][i[0]] for i in sim_scores[1:6]]
        return top_movies
    except IndexError:
        return ["Movie not found."]

# Collaborative Filtering
# Load ratings dataset for collaborative filtering
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Evaluate collaborative filtering model
rmse = accuracy.rmse(predictions)

# Predict ratings for a user
# Recommend top movies for a user
def recommend_collaborative(user_id, model, movies_df, ratings_df):
    user_rated = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    all_movies = movies_df['movieId'].unique()
    unrated_movies = [m for m in all_movies if m not in user_rated]
    
    predictions = [(movie, model.predict(user_id, movie).est) for movie in unrated_movies]
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
    recommended_titles = [movies_df[movies_df['movieId'] == movie].iloc[0]['title'] for movie, _ in top_movies]
    return recommended_titles

# Example usage
print("Content-Based Recommendations for 'Toy Story (1995)':")
print(recommend_content('Toy Story (1995)', cosine_sim, movies))

print("Collaborative Filtering Recommendations for User 1:")
print(recommend_collaborative(1, model, movies, ratings))
