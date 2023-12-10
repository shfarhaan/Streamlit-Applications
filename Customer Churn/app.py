# churn_app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Main Page
def main():
    st.title("Customer Churn Prediction")
    
    # Dropdown for page selection
    page = st.sidebar.selectbox("Select Page", ["Upload Data", "Explore Data", "Predict Churn"])

    # Display the selected page
    if page == "Upload Data":
        upload_data()
    elif page == "Explore Data":
        explore_data()
    elif page == "Predict Churn":
        predict_churn()

# Page 1: Upload Data
def upload_data():
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.session_state.df = df
        st.success("File uploaded successfully!")

# Page 2: Explore Data
def explore_data():
    st.header("Explore Data")

    if hasattr(st.session_state, "df"):
        st.header("Data Overview")
        st.dataframe(st.session_state.df.describe())

        st.header("Data Visualization")
        st.subheader("Correlation Matrix")
        corr_matrix = st.session_state.df.corr()
        st.write(sns.heatmap(corr_matrix, annot=True, cmap="coolwarm").figure)

        st.subheader("Customer Churn Distribution")
        churn_count = st.session_state.df['Churn'].value_counts()
        st.bar_chart(churn_count)

# Page 3: Predict Churn
def predict_churn():
    st.header("Predict Churn")

    if hasattr(st.session_state, "df"):
        st.header("Build a Churn Prediction Model")

        df = st.session_state.df.select_dtypes(include=['float64', 'int64'])

        X = df.drop(columns=['Churn'])
        y = df['Churn']

        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.subheader("Model Evaluation")
        st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.text("Confusion Matrix:")
        st.text(confusion_matrix(y_test, y_pred))

# Run the app
if __name__ == "__main__":
    main()
