import streamlit as st
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


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

    # Choose method for missing values handling
    method_options = ["Forward Fill", "Backward Fill", "Mean", "Median", "Custom"]  # Add more options if needed
    selected_method = st.selectbox("Select Method for Missing Values Handling", method_options)

    # Perform missing values handling based on the selected method
    if selected_method == "Forward Fill":
        df_filled = st.session_state.data.fillna(method="ffill")
    elif selected_method == "Backward Fill":
        df_filled = st.session_state.data.fillna(method="bfill")
    elif selected_method == "Mean":
        df_filled = st.session_state.data.fillna(st.session_state.data.mean())
    elif selected_method == "Median":
        df_filled = st.session_state.data.fillna(st.session_state.data.median())
    elif selected_method == "Custom":
        # Custom handling method using user input
        custom_value = st.text_input("Enter a custom value for missing values:", "")
        try:
            custom_value = float(custom_value)
            df_filled = st.session_state.data.fillna(custom_value)
        except ValueError:
            st.warning("Please enter a valid numeric value.")

    # Display the dataset with missing values handling
    st.write(f"### Dataset with {selected_method} Handling")
    st.write(df_filled)


# Page 3: Perform visualization and evaluation metrics
def page_visualization():
    st.title("Visualization and Evaluation Metrics")

    # Check if a dataset is uploaded
    if "data" not in st.session_state:
        st.warning("Please upload a dataset on the first page.")
        st.stop()

    # Display the dataset
    st.write("### Original Dataset")
    st.write(st.session_state.data)

    # Visualization options
    st.sidebar.subheader("Visualization Options")
    selected_visualization = st.sidebar.selectbox("Select Visualization", ["Histogram", "Scatter Plot", "Box Plot"])

    # Machine learning options
    st.sidebar.subheader("Machine Learning Options")
    selected_algorithm = st.sidebar.selectbox("Select Algorithm", ["Decision Tree", "Random Forest", "SVM"])

    # Cross-validation options
    st.sidebar.subheader("Cross-Validation Options")
    selected_cv = st.sidebar.selectbox("Select Cross-Validation", ["Stratified K-Fold", "KFold"])

    # Perform selected visualization
    if selected_visualization == "Histogram":
        st.subheader("Histogram")
        column_name = st.selectbox("Select a column for histogram", st.session_state.data.columns)
        plt.hist(st.session_state.data[column_name])
        st.pyplot()
    elif selected_visualization == "Scatter Plot":
        st.subheader("Scatter Plot")
        x_column = st.selectbox("Select X-axis column", st.session_state.data.columns)
        y_column = st.selectbox("Select Y-axis column", st.session_state.data.columns)
        plt.scatter(st.session_state.data[x_column], st.session_state.data[y_column])
        st.pyplot()
    elif selected_visualization == "Box Plot":
        st.subheader("Box Plot")
        x_column = st.selectbox("Select X-axis column", st.session_state.data.columns)
        y_column = st.selectbox("Select Y-axis column", st.session_state.data.columns)
        sns.boxplot(data=st.session_state.data, x=x_column, y=y_column)
        st.pyplot()

    # Perform machine learning with selected algorithm and cross-validation
    if selected_algorithm and selected_cv:
        st.subheader("Machine Learning Evaluation Metrics")

        # Assuming 'target' is the name of the target column in your dataset
        target_column = 'target'

        # Split the data into features and target
        X = st.session_state.data.drop(columns=[target_column])
        y = st.session_state.data[target_column]

        # Add your machine learning model training and evaluation code here
        # For example:
        if selected_algorithm == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()
        elif selected_algorithm == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
        elif selected_algorithm == "SVM":
            from sklearn.svm import SVC
            model = SVC()

        if selected_cv == "Stratified K-Fold":
            from sklearn.model_selection import StratifiedKFold
            cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        elif selected_cv == "KFold":
            from sklearn.model_selection import KFold
            cv_method = KFold(n_splits=5, shuffle=True, random_state=42)

        # Evaluate the model using cross-validation
        cv_results = cross_val_score(model, X, y, cv=cv_method, scoring='accuracy')

        # Display cross-validation results
        st.write("Cross-Validation Results:")
        st.write("Accuracy: {:.2f} (+/- {:.2f})".format(cv_results.mean(), cv_results.std() * 2))

        # Train the model on the entire dataset for prediction
        model.fit(X, y)

        # Make predictions on the same dataset (for simplicity, use the same data for training and testing)
        y_pred = model.predict(X)

        # Display precision, recall, and f1-score
        st.write("Precision: {:.2f}".format(precision_score(y, y_pred)))
        st.write("Recall: {:.2f}".format(recall_score(y, y_pred)))
        st.write("F1-Score: {:.2f}".format(f1_score(y, y_pred)))

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
    elif selected_page == "Visualizationa and Performance Metrics":
        page_visualization()

if __name__ == "__main__":
    main()
