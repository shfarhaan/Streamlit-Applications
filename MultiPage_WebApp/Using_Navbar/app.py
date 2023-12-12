import streamlit as st
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
        fig, ax = plt.subplots()
        ax.hist(st.session_state.data[column_name])
        st.pyplot(fig)
    elif selected_visualization == "Scatter Plot":
        st.subheader("Scatter Plot")
        x_column = st.selectbox("Select X-axis column", st.session_state.data.columns)
        y_column = st.selectbox("Select Y-axis column", st.session_state.data.columns)
        fig, ax = plt.subplots()
        ax.scatter(st.session_state.data[x_column], st.session_state.data[y_column])
        st.pyplot(fig)
    elif selected_visualization == "Box Plot":
        st.subheader("Box Plot")
        x_column = st.selectbox("Select X-axis column", st.session_state.data.columns)
        y_column = st.selectbox("Select Y-axis column", st.session_state.data.columns)
        fig, ax = plt.subplots()
        sns.boxplot(data=st.session_state.data, x=x_column, y=y_column, ax=ax)
        st.pyplot(fig)

    # Perform machine learning with selected algorithm and cross-validation
    if selected_algorithm and selected_cv:
        st.subheader("Machine Learning Evaluation Metrics")

        # Allow the user to select the target column
        target_column = st.selectbox("Select Target Column", st.session_state.data.columns)

        # Split the data into features and target
        X = st.session_state.data.drop(columns=[target_column])
        y = st.session_state.data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocess categorical columns with one-hot encoding
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        numeric_columns = X.select_dtypes(exclude=['object']).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_columns),
                ('cat', OneHotEncoder(), categorical_columns)
            ]
        )

        # Create a pipeline with the preprocessing step and the model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', get_model(selected_algorithm))
        ])

        if selected_cv == "Stratified K-Fold":
            cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        elif selected_cv == "KFold":
            cv_method = KFold(n_splits=5, shuffle=True, random_state=42)

        # Evaluate the model using cross-validation
        cv_results = cross_val_score(pipeline, X, y, cv=cv_method, scoring='accuracy')

        # Display cross-validation results
        st.write("Cross-Validation Results:")
        st.write("Accuracy: {:.2f} (+/- {:.2f})".format(cv_results.mean(), cv_results.std() * 2))

        pipeline.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = pipeline.predict(X_test)

        # Display precision, recall, and f1-score
        st.write("### Classification Metrics:")
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")

        # Display confusion matrix
        st.write("### Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=pipeline.classes_, yticklabels=pipeline.classes_, ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig)

# Function to get the appropriate model based on the selected algorithm
def get_model(selected_algorithm):
    if selected_algorithm == "Decision Tree":
        return DecisionTreeClassifier()
    elif selected_algorithm == "Random Forest":
        return RandomForestClassifier()
    elif selected_algorithm == "SVM":
        return SVC()



# Main function to run the application
def main():
    st.set_page_config(page_title="Data Analysis App", page_icon="📊")

    # Initialize SessionState to store data between pages
    session_state = st.session_state
    if not hasattr(session_state, "data"):
        session_state.data = None

    # Create a drop-down menu for navigation
    page_options = ["Upload Dataset", "Missing Values Handling", "Visualizationa and Metrics"]
    selected_page = st.sidebar.selectbox("Select Page", page_options)

    # Display the selected page
    if selected_page == "Upload Dataset":
        page_upload_dataset()
    elif selected_page == "Missinow g Values Handling":
        page_missing_values_handling()
    elif selected_page == "Visualizationa and Metrics":
        page_visualization()

if __name__ == "__main__":
    main()
