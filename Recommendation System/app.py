import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# Function to upload and load dataset
def load_dataset():
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)  # Adjust for different file types
        return df


# Data preprocessing function
def preprocess_data(df):
    # Placeholder: Handle duplicates
    df.drop_duplicates(inplace=True)
    
    # Placeholder: Convert date column to datetime format
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    # Placeholder: Extract year and month from the release date
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    
    # Placeholder: Drop unnecessary columns
    df.drop(['release_date', 'unnecessary_column'], axis=1, inplace=True)

    # Placeholder: Handle categorical features (one-hot encoding)
    df = pd.get_dummies(df, columns=['genre'], prefix='genre')

    # Placeholder: Normalize numerical features
    numerical_columns = ['rating', 'duration', 'budget']
    df[numerical_columns] = (df[numerical_columns] - df[numerical_columns].mean()) / df[numerical_columns].std()

    preprocessed_data = df.copy()

    return preprocessed_data


# Handling missing values function
def handle_missing_values(df):
    # Check and handle missing values
    st.subheader("Handling Missing Values")
    
    # Display the original dataset with missing values
    st.write("Original Dataset with Missing Values:")
    st.dataframe(df.style.highlight_null(null_color="red"))

    # Handling missing values
    df_cleaned = df.dropna()  # You can customize this based on your handling strategy
    
    # Display the cleaned dataset without missing values
    st.write("Cleaned Dataset without Missing Values:")
    st.dataframe(df_cleaned)

    return df_cleaned



# Fine-tuning and algorithm selection
def fine_tune_and_select_algorithm(preprocessed_data):

    # Provide options for algorithm selection
    selected_algorithm = st.multiselect("Select Ensemble Algorithms (up to 3)", ["Algorithm1", "Algorithm2", "Algorithm3"])

    # Algorithm-specific fine-tuning
    tuned_data = preprocessed_data.copy()

    if "Algorithm1" in selected_algorithm:
        # Fine-tuning for Algorithm1
        st.subheader("Fine-Tuning for Algorithm1")
        
        # Example: Linear scaling for Algorithm1
        algorithm1_scaling_factor = st.slider("Set Scaling Factor for Algorithm1", 0.1, 2.0, 1.0)
        tuned_data["Algorithm1_Output"] = preprocessed_data["Algorithm1_Input"] * algorithm1_scaling_factor

    # Repeat for other selected algorithms
    if "Algorithm2" in selected_algorithm:
        # Fine-tuning for Algorithm2
        st.subheader("Fine-Tuning for Algorithm2")
        
        # Example: Linear scaling for Algorithm2
        algorithm2_scaling_factor = st.slider("Set Scaling Factor for Algorithm2", 0.1, 2.0, 1.0)
        tuned_data["Algorithm2_Output"] = preprocessed_data["Algorithm2_Input"] * algorithm2_scaling_factor

    if "Algorithm3" in selected_algorithm:
        # Fine-tuning for Algorithm3
        st.subheader("Fine-Tuning for Algorithm3")
        
        # Example: Linear scaling for Algorithm3
        algorithm3_scaling_factor = st.slider("Set Scaling Factor for Algorithm3", 0.1, 2.0, 1.0)
        tuned_data["Algorithm3_Output"] = preprocessed_data["Algorithm3_Input"] * algorithm3_scaling_factor

    return tuned_data


# Genre selection
def select_genres():
    selected_genres = st.multiselect("Select Movie Genres", ["Action", "Drama", "Comedy", "Sci-Fi", "..."])
    return selected_genres

# Generate movie recommendations
def generate_recommendations(tuned_data, selected_genres):
    # Implement recommendation generation logic

    # Filter movies based on selected genres
    genre_filtered_data = tuned_data[tuned_data['Genre'].isin(selected_genres)]

    # You can add more sophisticated recommendation logic here, such as collaborative filtering, content-based filtering, etc.
    # For demonstration purposes, let's assume a simple recommendation based on the highest-rated movies.

    # Sort the data by rating in descending order
    sorted_data = genre_filtered_data.sort_values(by='Rating', ascending=False)

    # Take the top 5 recommended movies
    top_recommendations = sorted_data.head(5)

    recommendations = {
        "selected_genres": selected_genres,
        "top_recommendations": top_recommendations[['Title', 'Genre', 'Rating']]
    }

    return recommendations



# Display recommendations
def display_recommendations(recommendations):
    # Display recommendations using Streamlit components

    st.title("Movie Recommendations")

    # Display selected genres
    st.subheader("Selected Genres:")
    st.write(", ".join(recommendations["selected_genres"]))

    # Display top recommendations
    st.subheader("Top Recommendations:")
    recommendations_table = recommendations["top_recommendations"]
    st.table(recommendations_table)

    # Additional visualizations or details can be added based on your preferences
    # For example, you can display movie posters, additional information, etc.
    # ...

    # Optionally, provide a button to refresh recommendations or go back to the main page
    if st.button("Back to Main Page"):
        st.experimental_rerun()


# Streamlit app structure
def main():
    st.title("Movie Recommendation System")

    # Page selection sidebar
    page = st.sidebar.selectbox("Select Page", ["Upload Data", "Preprocess Data", "Handle Missing Values",
                                                "Fine-Tune and Select Algorithm", "Select Genres", "Display Recommendations"])

    # Display corresponding page based on user selection
    if page == "Upload Data":
        df = load_dataset()
    elif page == "Preprocess Data":
        preprocessed_data = preprocess_data(df)
    elif page == "Handle Missing Values":
        df_cleaned = handle_missing_values(df)
    elif page == "Fine-Tune and Select Algorithm":
        tuned_data = fine_tune_and_select_algorithm(preprocessed_data)
    elif page == "Select Genres":
        selected_genres = select_genres()
    elif page == "Display Recommendations":
        recommendations = generate_recommendations(tuned_data, selected_genres)
        display_recommendations(recommendations)

if __name__ == "__main__":
    main()