import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Function for hyperparameter tuning and evaluation
def hyperparameter_tuning(X_train, y_train, X_test, y_test, algorithm, param_grid):
    clf = algorithm()
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Evaluation on the testing set
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return best_model, accuracy

# Streamlit App
st.title("Boosting Algorithm Hyperparameter Tuning")

# Sidebar for user input
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

algorithm_choice = st.sidebar.selectbox("Select Boosting Algorithm", 
                                        ["AdaBoost", "Gradient Boosting", "XGBoost"])
st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameter Tuning Options")
n_estimators = st.sidebar.slider("Number of Estimators", 50, 200, 100, step=50)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.2, 0.1, step=0.01)

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # EDA
    st.header("Exploratory Data Analysis (EDA)")

    # Display basic statistics
    st.subheader("Dataset Overview:")
    st.write(df.head())

    # Display summary statistics
    st.subheader("Summary Statistics:")
    st.write(df.describe())

    # Handle missing values
    st.subheader("Handling Missing Values:")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    # Visualize missing values
    st.subheader("Visualization of Missing Values:")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    st.pyplot()




    # Drop rows with missing values (you can customize this based on your strategy)
    df.dropna(inplace=True)

    # Display updated dataset after handling missing values
    st.subheader("Updated Dataset after Handling Missing Values:")
    st.write(df.head())

    # Visualizations
    st.subheader("Visualizations:")
    # Customize your visualizations based on your dataset and objectives

    # Split the dataset into features and target variable
    X = df.drop("target_variable_column_name", axis=1)  # Replace with your target variable column name
    y = df["target_variable_column_name"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameter search spaces for different boosting algorithms
    ada_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    gb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7],
                 'min_samples_split': [2, 4, 6]}
    xgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7],
                  'min_child_weight': [1, 3, 5]}

    # Perform hyperparameter tuning based on user choice
    if algorithm_choice == "AdaBoost":
        best_model, accuracy = hyperparameter_tuning(X_train, y_train, X_test, y_test, AdaBoostClassifier,
                                                     {'n_estimators': [n_estimators], 'learning_rate': [learning_rate]})
    elif algorithm_choice == "Gradient Boosting":
        best_model, accuracy = hyperparameter_tuning(X_train, y_train, X_test, y_test, GradientBoostingClassifier,
                                                     {'n_estimators': [n_estimators], 'learning_rate': [learning_rate],
                                                      'max_depth': [3, 5, 7], 'min_samples_split': [2, 4, 6]})
    else:  # XGBoost
        best_model, accuracy = hyperparameter_tuning(X_train, y_train, X_test, y_test, xgb.XGBClassifier,
                                                     {'n_estimators': [n_estimators], 'learning_rate': [learning_rate],
                                                      'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]})

    # Display results
    st.header("Hyperparameter Tuning Results")
    st.write(f"Best Hyperparameters: {best_model.get_params()}")
    st.write(f"Accuracy on Testing Set: {accuracy:.2f}")

    # Visualize feature importances for tree-based models
    if algorithm_choice in ["Gradient Boosting", "XGBoost"]:
        st.header("Feature Importances")
        feature_importances = best_model.feature_importances_
        feature_names = [f"Feature {i}" for i in range(1, X.shape[1] + 1)]

        fig, ax = plt.subplots()
        ax.barh(feature_names, feature_importances)
        st.pyplot(fig)
else:
    st.warning("Upload a CSV file to get started.")
