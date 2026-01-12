# Streamlit-Applications

This repository contains various applications built using **Streamlit**, a Python framework for creating interactive web apps. Streamlit allows you to turn your data analysis and machine learning models into beautiful and easy-to-use web apps with just a few lines of code.

## 🚀 Features

All applications have been updated with:
- ✅ Latest Streamlit APIs (v1.39.0+)
- ✅ Modern Python package versions
- ✅ Fixed deprecated methods and functions
- ✅ Individual requirements.txt files per project
- ✅ Enhanced error handling and user experience

## 📋 Prerequisites

- Python 3.8 or higher (tested with Python 3.12.3)
- pip package manager

## 🛠️ Installation

### Option 1: Install all dependencies at once

Clone the repository and install all dependencies:

```bash
git clone https://github.com/shfarhaan/Streamlit-Applications.git
cd Streamlit-Applications
pip install -r requirements.txt
```

### Option 2: Install per project

Navigate to a specific project directory and install its dependencies:

```bash
cd Streamlit-Applications/<project-name>
pip install -r requirements.txt
```

## 📁 Applications

### 1. Iris EDA Web App
**Location:** `Iris_EDA_Web_App/`

Performs exploratory data analysis (EDA) on the classic Iris dataset. Features include:
- Interactive data preview and statistics
- Multiple visualization options (bar plots, correlation matrices)
- Image manipulation with contrast and size controls
- Species information with images

**Run:**
```bash
cd Iris_EDA_Web_App
streamlit run app.py
# or
streamlit run iris_app.py
```

### 2. Boosting with Visualization
**Location:** `Boosting with Visualization/`

Interactive hyperparameter tuning for boosting algorithms:
- Support for AdaBoost, Gradient Boosting, and XGBoost
- Upload custom CSV datasets
- Automated EDA with missing value visualization
- Feature importance plots
- Real-time hyperparameter tuning with GridSearchCV

**Run:**
```bash
cd "Boosting with Visualization"
streamlit run app.py
```

### 3. Boosting Web App
**Location:** `Boosting_Web_App/`

Simplified boosting algorithm comparison using synthetic data:
- Pre-generated dataset for quick testing
- Compare multiple boosting algorithms
- Interactive parameter adjustment
- Feature importance visualization

**Run:**
```bash
cd Boosting_Web_App
streamlit run app.py
```

### 4. Customer Churn Prediction
**Location:** `Customer Churn/`

Multi-page application for predicting customer churn:
- Upload and explore customer data
- Correlation analysis and visualization
- Random Forest classification model
- Model evaluation metrics and confusion matrix

**Run:**
```bash
cd "Customer Churn"
streamlit run app.py
```

### 5. MultiPage Web App
**Location:** `MultiPage_WebApp/`

Two different approaches to creating multi-page Streamlit applications:

#### Using Radio Buttons (`Using_Radio_Btn/`)
- Page navigation with radio buttons
- Dataset upload and preview
- Missing value handling with multiple strategies
- Clean and simple interface

#### Using Navbar/Selectbox (`Using_Navbar/`)
- Advanced navigation with sidebar selectbox
- Comprehensive missing value handling options
- Multiple visualization types (histogram, scatter, box plots)
- Machine learning model comparison (Decision Tree, Random Forest, SVM)
- Cross-validation support (Stratified K-Fold, KFold)
- Evaluation metrics with confusion matrix

**Run:**
```bash
cd MultiPage_WebApp/Using_Radio_Btn
streamlit run app.py

# or

cd MultiPage_WebApp/Using_Navbar
streamlit run app.py
```

### 6. Recommendation System
**Location:** `Recommendation System/`

Content-based movie recommendation system with two versions:

#### Standard Version (`app.py`)
- Content-based filtering using TF-IDF and cosine similarity
- Select a movie and get similar recommendations
- Adjustable number of recommendations
- Movie details display

#### Beta Version (`beta.py`)
- Enhanced UI with wide layout
- Advanced filtering (genre, year range, rating)
- Search functionality
- Dataset statistics and visualizations
- Card-based recommendation display

**Run:**
```bash
cd "Recommendation System"
streamlit run app.py
# or for beta version
streamlit run beta.py
```

### 7. Streamlit Basics Tutorial
**Location:** `Streamlit Basics/`

Comprehensive tutorial covering all Streamlit components:
- Text elements (title, header, markdown, code)
- Data display (dataframe, table, metrics)
- Interactive widgets (sliders, buttons, inputs, checkboxes)
- File upload functionality
- Charts (line, area, bar)
- Status messages and expanders
- Sidebar components

**Run:**
```bash
cd "Streamlit Basics"
streamlit run "tutorial 1.py"
```

## 🔧 Common Issues and Solutions

### Issue: "No module named 'streamlit'"
**Solution:** Install Streamlit using `pip install streamlit`

### Issue: Deprecated warnings
**Solution:** This repository has been updated to use the latest APIs. Make sure you're using the versions specified in requirements.txt

### Issue: Dataset not found
**Solution:** Some apps like the Recommendation System will use sample data if the dataset file is missing. For other apps, make sure you're running the app from its directory.

## 📚 Documentation References

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit API Reference](https://docs.streamlit.io/library/api-reference)
- [Streamlit Community](https://discuss.streamlit.io)

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork this repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For feedback, suggestions, or questions, contact: **shfarhaan21@gmail.com**

## ⭐ Acknowledgments

- Thanks to the Streamlit team for creating an amazing framework
- All datasets used are publicly available or synthetically generated
- Built with ❤️ by Sazzad Hussain Farhaan

## 📝 License

This project is open source and available for educational purposes.