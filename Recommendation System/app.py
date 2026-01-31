"""
Movie Recommendation System using Streamlit

This application demonstrates a simple content-based movie recommendation system.
Users can select a movie and get similar movie recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Load movie dataset"""
    try:
        # Try to load the dataset if it exists
        movies = pd.read_csv("dataset.csv")
        return movies
    except FileNotFoundError:
        # Create a sample dataset if file doesn't exist
        st.warning("Dataset not found. Using sample data.")
        sample_data = {
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
                'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                'Goodfellas', 'The Silence of the Lambs', 'Saving Private Ryan'
            ],
            'genres': [
                'Drama', 'Crime Drama', 'Action Thriller', 'Crime Drama',
                'Drama Romance', 'Sci-Fi Thriller', 'Sci-Fi Action',
                'Crime Drama', 'Thriller Horror', 'War Drama'
            ],
            'overview': [
                'Two imprisoned men bond over a number of years',
                'The aging patriarch of an organized crime dynasty',
                'When the menace known as the Joker wreaks havoc',
                'The lives of two mob hitmen, a boxer, a gangster',
                'The presidencies of Kennedy and Johnson, the Vietnam War',
                'A thief who steals corporate secrets through dream-sharing',
                'A computer hacker learns about the true nature of reality',
                'The story of Henry Hill and his life in the mob',
                'A young FBI cadet must receive help from an incarcerated',
                'Following the Normandy Landings, a group of soldiers'
            ],
            'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.6]
        }
        return pd.DataFrame(sample_data)

def create_similarity_matrix(movies):
    """Create similarity matrix based on movie features"""
    # Combine genres and overview for better recommendations
    movies['combined_features'] = movies['genres'] + ' ' + movies['overview']
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return similarity_matrix

def get_recommendations(movie_title, movies, similarity_matrix, n_recommendations=5):
    """Get movie recommendations based on similarity"""
    # Get the index of the movie
    idx = movies[movies['title'] == movie_title].index[0]
    
    # Get similarity scores for all movies
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort by similarity score
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top n similar movies (excluding the movie itself)
    similarity_scores = similarity_scores[1:n_recommendations+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in similarity_scores]
    
    # Return recommended movies with scores
    recommendations = movies.iloc[movie_indices].copy()
    recommendations['similarity_score'] = [score[1] for score in similarity_scores]
    
    return recommendations

def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the Movie Recommendation System!
    
    This application uses **content-based filtering** to recommend movies similar to your selection.
    The recommendations are based on movie genres and descriptions.
    """)
    
    # Load data
    with st.spinner("Loading movie data..."):
        movies = load_data()
    
    # Create similarity matrix
    similarity_matrix = create_similarity_matrix(movies)
    
    # Sidebar
    st.sidebar.header("Settings")
    n_recommendations = st.sidebar.slider(
        "Number of recommendations:",
        min_value=1,
        max_value=10,
        value=5
    )
    
    # Main content
    st.markdown("---")
    st.subheader("Select a Movie")
    
    # Movie selection
    selected_movie = st.selectbox(
        "Choose a movie you like:",
        movies['title'].tolist()
    )
    
    # Display selected movie info
    if selected_movie:
        st.markdown("### Selected Movie")
        movie_info = movies[movies['title'] == selected_movie].iloc[0]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Title:** {movie_info['title']}")
            st.write(f"**Genre:** {movie_info['genres']}")
            st.write(f"**Overview:** {movie_info['overview']}")
        with col2:
            if 'rating' in movie_info:
                st.metric("Rating", f"{movie_info['rating']}/10")
        
        # Get recommendations
        st.markdown("---")
        st.subheader("Recommended Movies")
        
        recommendations = get_recommendations(
            selected_movie, 
            movies, 
            similarity_matrix, 
            n_recommendations
        )
        
        # Display recommendations
        for idx, row in recommendations.iterrows():
            with st.expander(f"📽️ {row['title']} (Similarity: {row['similarity_score']:.2%})"):
                st.write(f"**Genre:** {row['genres']}")
                st.write(f"**Overview:** {row['overview']}")
                if 'rating' in row:
                    st.write(f"**Rating:** {row['rating']}/10")
    
    # Display dataset info
    st.markdown("---")
    if st.checkbox("Show dataset information"):
        st.subheader("Dataset Overview")
        st.write(f"Total movies in dataset: {len(movies)}")
        st.dataframe(movies)

if __name__ == "__main__":
    main()
