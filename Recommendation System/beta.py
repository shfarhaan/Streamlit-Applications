"""
Movie Recommendation System - Beta Version

This is an enhanced version with additional features like filtering and search.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    """Load movie dataset with caching"""
    try:
        movies = pd.read_csv("dataset.csv")
        return movies
    except FileNotFoundError:
        # Sample dataset
        sample_data = {
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
                'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                'Goodfellas', 'The Silence of the Lambs', 'Saving Private Ryan',
                'Star Wars', 'Interstellar', 'The Lion King', 'Gladiator',
                'The Departed', 'The Prestige', 'Memento', 'Fight Club',
                'The Green Mile', 'The Usual Suspects'
            ],
            'genres': [
                'Drama', 'Crime Drama', 'Action Thriller', 'Crime Drama',
                'Drama Romance', 'Sci-Fi Thriller', 'Sci-Fi Action',
                'Crime Drama', 'Thriller Horror', 'War Drama',
                'Sci-Fi Adventure', 'Sci-Fi Drama', 'Animation Drama',
                'Action Drama', 'Crime Thriller', 'Mystery Thriller',
                'Mystery Thriller', 'Drama Thriller', 'Drama Fantasy',
                'Mystery Thriller'
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
                'Following the Normandy Landings, a group of soldiers',
                'Luke Skywalker joins forces with a Jedi Knight',
                'A team of explorers travel through a wormhole',
                'Lion cub prince flees his kingdom after his father',
                'A former Roman General sets out to exact vengeance',
                'An undercover cop and a mole in the police',
                'Two magicians engage in competitive one-upmanship',
                'A man with short-term memory loss attempts to track',
                'An insomniac office worker and a soap salesman',
                'The lives of guards on Death Row are affected',
                'A sole survivor tells of the twisty events'
            ],
            'year': [1994, 1972, 2008, 1994, 1994, 2010, 1999, 1990, 1991, 1998,
                    1977, 2014, 1994, 2000, 2006, 2006, 2000, 1999, 1999, 1995],
            'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.6,
                      8.6, 8.6, 8.5, 8.5, 8.5, 8.5, 8.4, 8.8, 8.6, 8.5]
        }
        return pd.DataFrame(sample_data)

@st.cache_data
def create_similarity_matrix(_movies):
    """Create similarity matrix with caching"""
    movies_copy = _movies.copy()
    movies_copy['combined_features'] = movies_copy['genres'] + ' ' + movies_copy['overview']
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(movies_copy['combined_features'])
    
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(movie_title, movies, similarity_matrix, n_recommendations=5, min_rating=0):
    """Get filtered movie recommendations"""
    idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Filter by minimum rating if applicable
    recommendations = []
    for i, score in similarity_scores[1:]:
        if 'rating' in movies.columns:
            if movies.iloc[i]['rating'] >= min_rating:
                recommendations.append((i, score))
        else:
            recommendations.append((i, score))
        
        if len(recommendations) >= n_recommendations:
            break
    
    movie_indices = [i[0] for i in recommendations]
    similarity_scores = [i[1] for i in recommendations]
    
    result = movies.iloc[movie_indices].copy()
    result['similarity_score'] = similarity_scores
    
    return result

def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    
    st.title("🎬 Movie Recommendation System (Beta)")
    st.markdown("### Enhanced version with filtering and search")
    
    # Load data
    movies = load_data()
    similarity_matrix = create_similarity_matrix(movies)
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Genre filter
    if 'genres' in movies.columns:
        all_genres = sorted(set([g.strip() for genres in movies['genres'] for g in genres.split()]))
        selected_genres = st.sidebar.multiselect("Filter by Genre:", all_genres)
    
    # Year range filter
    if 'year' in movies.columns:
        year_range = st.sidebar.slider(
            "Year Range:",
            int(movies['year'].min()),
            int(movies['year'].max()),
            (int(movies['year'].min()), int(movies['year'].max()))
        )
    
    # Rating filter
    if 'rating' in movies.columns:
        min_rating = st.sidebar.slider(
            "Minimum Rating:",
            0.0,
            10.0,
            0.0,
            0.5
        )
    else:
        min_rating = 0.0
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider("Number of Recommendations:", 1, 15, 5)
    
    # Apply filters
    filtered_movies = movies.copy()
    if 'genres' in movies.columns and selected_genres:
        filtered_movies = filtered_movies[
            filtered_movies['genres'].apply(lambda x: any(g in x for g in selected_genres))
        ]
    if 'year' in movies.columns:
        filtered_movies = filtered_movies[
            (filtered_movies['year'] >= year_range[0]) & 
            (filtered_movies['year'] <= year_range[1])
        ]
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search functionality
        search_term = st.text_input("🔎 Search for a movie:", "")
        
        if search_term:
            search_results = filtered_movies[
                filtered_movies['title'].str.contains(search_term, case=False)
            ]
            if not search_results.empty:
                selected_movie = st.selectbox(
                    "Select from search results:",
                    search_results['title'].tolist()
                )
            else:
                st.warning("No movies found. Try a different search term.")
                selected_movie = st.selectbox(
                    "Or select from all movies:",
                    filtered_movies['title'].tolist()
                )
        else:
            selected_movie = st.selectbox(
                "Select a movie:",
                filtered_movies['title'].tolist()
            )
    
    with col2:
        st.metric("Total Movies", len(filtered_movies))
        if 'rating' in movies.columns:
            avg_rating = filtered_movies['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.1f}/10")
    
    # Display selected movie
    if selected_movie:
        st.markdown("---")
        st.subheader("🎥 Selected Movie")
        
        movie_info = movies[movies['title'] == selected_movie].iloc[0]
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### {movie_info['title']}")
            st.write(f"**Overview:** {movie_info['overview']}")
        with col2:
            st.write(f"**Genre:** {movie_info['genres']}")
            if 'year' in movie_info:
                st.write(f"**Year:** {movie_info['year']}")
        with col3:
            if 'rating' in movie_info:
                st.metric("⭐ Rating", f"{movie_info['rating']}/10")
        
        # Get and display recommendations
        st.markdown("---")
        st.subheader("🎯 Recommended Movies")
        
        recommendations = get_recommendations(
            selected_movie,
            movies,
            similarity_matrix,
            n_recommendations,
            min_rating
        )
        
        if recommendations.empty:
            st.info("No recommendations match your filters. Try adjusting the filters.")
        else:
            # Display as cards
            for idx, row in recommendations.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"#### {row['title']}")
                        st.write(row['overview'])
                    with col2:
                        st.write(f"**{row['genres']}**")
                        if 'year' in row:
                            st.write(f"Year: {row['year']}")
                    with col3:
                        if 'rating' in row:
                            st.metric("Rating", f"{row['rating']}/10")
                        st.metric("Match", f"{row['similarity_score']:.0%}")
                    st.markdown("---")
    
    # Dataset statistics
    with st.expander("📊 View Dataset Statistics"):
        st.subheader("Dataset Overview")
        st.write(f"Total movies: {len(movies)}")
        
        if 'genres' in movies.columns:
            st.write("**Genre Distribution:**")
            genre_counts = {}
            for genres in movies['genres']:
                for genre in genres.split():
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            st.bar_chart(pd.Series(genre_counts).sort_values(ascending=False))
        
        st.dataframe(movies)

if __name__ == "__main__":
    main()
