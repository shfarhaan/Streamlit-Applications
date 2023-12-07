import streamlit as st
import pandas as pd
import numpy as np

# Page 1: Upload Dataset
def page_upload_dataset():
    st.title("Upload Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        
        # Store the dataset in a Streamlit session state for access in other pages
        st.session_state.dataset = df

        st.success("Dataset uploaded successfully!")
        st.write(df.head())

# Page 2: Perform Missing Values Handling
def page_missing_values_handling():
    st.title("Missing Values Handling")
    
    # Access the dataset from the session state
    if 'dataset' not in st.session_state:
        st.warning("Please upload a dataset on the first page.")
        st.stop()

    df = st.session_state.dataset

    # Display missing values summary
    st.subheader("Missing Values Summary")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # Handle missing values (replace NaN with mean for numerical columns as an example)
    st.subheader("Handle Missing Values")
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.floating):
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)


    st.success("Missing values handled successfully!")
    st.write(df.head())

# Page 3: Other analysis or processing
def page_other_analysis():
    st.title("Other Analysis or Processing")
    
    # Access the dataset from the session state
    if 'dataset' not in st.session_state:
        st.warning("Please upload a dataset on the first page.")
        st.stop()

    df = st.session_state.dataset

    # Perform other analysis or processing here
    # ...

# Streamlit app navigation
def main():
    st.set_page_config(page_title="Data Analysis App", page_icon="📊")
    st.sidebar.title("Navigation")
    page_options = ["Upload Dataset", "Missing Values Handling", "Other Analysis"]
    page_selection = st.sidebar.radio("Go to", page_options)

    if page_selection == "Upload Dataset":
        page_upload_dataset()
    elif page_selection == "Missing Values Handling":
        page_missing_values_handling()
    elif page_selection == "Other Analysis":
        page_other_analysis()

if __name__ == "__main__":
    main()
