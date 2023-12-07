import streamlit as st
import pandas as pd

# Page 1: Upload Dataset
def page_upload_dataset():
    st.title("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the dataset and store it in SessionState
        df = pd.read_csv(uploaded_file)
        session_state.data = df
        st.success("Dataset uploaded successfully!")

# Page 2: Perform missing values handling
def page_missing_values_handling():
    st.title("Missing Values Handling")
    if "data" not in st.session_state:
        st.warning("Please upload a dataset on the first page.")
        st.stop()
    
    # Display the dataset
    st.write("### Original Dataset")
    st.write(st.session_state.data)
    
    # Perform missing values handling (e.g., fillna)
    df_filled = st.session_state.data.fillna(method="ffill")  # Replace with your own handling method
    st.write("### Dataset with Missing Values Handling")
    st.write(df_filled)

# Page 3: Perform descriptive and inferential statistics
def page_statistics():
    st.title("Descriptive and Inferential Statistics")
    if "data" not in st.session_state:
        st.warning("Please upload a dataset on the first page.")
        st.stop()

    # Display the dataset
    st.write("### Original Dataset")
    st.write(st.session_state.data)

    # Perform descriptive and inferential statistics
    # Add your statistical analysis code here

# Main function to run the application
def main():
    st.set_page_config(page_title="Data Analysis App", page_icon="📊")

    # Initialize SessionState to store data between pages
    if "data" not in st.session_state:
        st.session_state.data = None

    # Navigation bar
    pages = ["Upload Dataset", "Missing Values Handling", "Statistics"]
    page = st.sidebar.selectbox("Select Page", pages)

    # Display the selected page
    if page == "Upload Dataset":
        page_upload_dataset()
    elif page == "Missing Values Handling":
        page_missing_values_handling()
    elif page == "Statistics":
        page_statistics()

if __name__ == "__main__":
    main()


