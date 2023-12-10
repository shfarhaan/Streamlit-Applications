import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page 1: Upload Data
def upload_data():
    st.title("Customer Churn Prediction")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Upload Data", "Explore Data", "Predict Churn"], key="upload_data")

    if page == "Upload Data":
        st.header("Upload Your Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            # Save the dataframe for later use
            st.session_state.df = df

            st.success("File uploaded successfully!")

# Page 2: Explore Data
def explore_data():
    st.title("Explore Data")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Upload Data", "Explore Data", "Predict Churn"], key="explore_data")

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
    st.title("Predict Churn")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Upload Data", "Explore Data", "Predict Churn"], key="predict_churn")

    if hasattr(st.session_state, "df"):
        st.header("Build a Churn Prediction Model")

        # Preprocess the data (you may need to adapt this based on your dataset)
        df = st.session_state.df
        X = df.drop(columns=['Churn'])
        y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a simple RandomForestClassifier (you may need to fine-tune the model)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        st.subheader("Model Evaluation")
        st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.text("Confusion Matrix:")
        st.text(confusion_matrix(y_test, y_pred))

# Run the app
if __name__ == "__main__":
    upload_data()
    explore_data()
    predict_churn()
