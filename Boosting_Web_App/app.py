import streamlit as st
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter search spaces
ada_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
gb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7],
             'min_samples_split': [2, 4, 6]}
xgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7],
              'min_child_weight': [1, 3, 5]}

# Streamlit App
st.title("Boosting Algorithm Hyperparameter Tuning")

# Sidebar for user input
algorithm_choice = st.sidebar.selectbox("Select Boosting Algorithm", 
                                        ["AdaBoost", "Gradient Boosting", "XGBoost"])
st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameter Tuning Options")
n_estimators = st.sidebar.slider("Number of Estimators", 50, 200, 100, step=50)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.2, 0.1, step=0.01)

# Display dataset information
st.header("Dataset Information")
st.write(f"Number of samples: {X.shape[0]}")
st.write(f"Number of features: {X.shape[1]}")

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
