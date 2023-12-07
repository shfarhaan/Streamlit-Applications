import streamlit as st
import pandas as pd

# Page 1: Upload Dataset with Descriptive and Inferential Statistics
def page_upload_dataset():
    st.title("Upload Dataset")

    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the dataset and store it in SessionState
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df

        # Display basic information about the dataset
        st.write("### Basic Information about the Dataset")
        st.write(f"Number of Rows: {df.shape[0]}")
        st.write(f"Number of Columns: {df.shape[1]}")
        st.write(f"Column Names: {', '.join(df.columns)}")

        # Display the first few rows of the dataset
        st.write("### Preview of the Dataset")
        st.write(df.head())

        # Descriptive Statistics
        st.write("### Descriptive Statistics")
        st.write(df.describe())

        # Inferential Statistics (example: correlation matrix)
        st.write("### Inferential Statistics (Correlation Matrix)")
        correlation_matrix = df.corr()
        st.write(correlation_matrix)

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
    session_state = st.session_state
    if not hasattr(session_state, "data"):
        session_state.data = None

    # Create a drop-down menu for navigation
    page_options = ["Upload Dataset", "Missing Values Handling", "Statistics"]
    selected_page = st.sidebar.selectbox("Select Page", page_options)

    # Display the selected page
    if selected_page == "Upload Dataset":
        page_upload_dataset()
    elif selected_page == "Missing Values Handling":
        page_missing_values_handling()
    elif selected_page == "Statistics":
        page_statistics()

if __name__ == "__main__":
    main()
