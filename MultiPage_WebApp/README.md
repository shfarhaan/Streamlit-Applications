### MultiPage Web App Examples

Two different approaches to creating multi-page applications in Streamlit.

#### Overview
This directory contains examples demonstrating different navigation patterns for building multi-page Streamlit applications. Each approach has its own advantages depending on your use case.

---

## Using Radio Buttons (`Using_Radio_Btn/`)

### Description
A simple and clean approach using radio buttons in the sidebar for page navigation.

### Features
- 📡 Radio button navigation
- 📤 CSV file upload
- 📊 Dataset preview with head display
- 🔧 Missing value handling with multiple strategies:
  - Forward fill
  - Backward fill
  - Mean imputation
  - Median imputation
  - Custom value
- 💾 Session state management for data persistence

### Best For
- Small to medium applications (2-5 pages)
- Linear workflows
- Simple navigation needs
- Educational purposes

### Running
```bash
cd Using_Radio_Btn
streamlit run app.py
```

---

## Using Navbar/Selectbox (`Using_Navbar/`)

### Description
A more advanced implementation using selectbox for navigation, with comprehensive data analysis and machine learning features.

### Features
- 🎯 Dropdown/selectbox navigation
- 📤 Dataset upload with statistics
- 📈 Comprehensive data exploration:
  - Descriptive statistics
  - Inferential statistics (correlation matrix)
- 🔧 Advanced missing value handling:
  - Forward fill (ffill)
  - Backward fill (bfill)
  - Mean imputation
  - Median imputation
  - Custom value replacement
- 📊 Multiple visualization types:
  - Histograms
  - Scatter plots
  - Box plots
- 🤖 Machine Learning integration:
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Machine (SVM)
- ✅ Cross-validation options:
  - Stratified K-Fold
  - K-Fold
- 📉 Evaluation metrics:
  - Accuracy scores
  - Precision, Recall, F1-Score
  - Confusion matrix
- 🎨 One-hot encoding for categorical features
- 🔄 Full ML pipeline with preprocessing

### Best For
- Complex applications (5+ pages)
- Data science workflows
- ML model comparison
- Production-ready applications

### Running
```bash
cd Using_Navbar
streamlit run app.py
```

---

## Comparison

| Feature | Radio Button | Navbar/Selectbox |
|---------|--------------|------------------|
| Navigation Type | Radio buttons | Selectbox dropdown |
| Complexity | Simple | Advanced |
| Pages | 3 | 3 |
| ML Features | ❌ | ✅ |
| Visualizations | Basic | Advanced |
| Best For | Learning | Production |
| Cross-validation | ❌ | ✅ |
| Model Comparison | ❌ | ✅ |

---

## Requirements

Both applications require:
```bash
streamlit>=1.39.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
matplotlib>=3.9.0
seaborn>=0.13.0
```

Install with:
```bash
cd MultiPage_WebApp
pip install -r requirements.txt
```

---

## Key Concepts Demonstrated

### 1. Session State
Both apps use `st.session_state` to persist data across page changes:
```python
st.session_state.dataset = df
```

### 2. Page Functions
Each page is implemented as a separate function:
```python
def page_upload_dataset():
    st.title("Upload Dataset")
    # page content

def page_analysis():
    st.title("Analysis")
    # page content
```

### 3. Navigation Logic
Different navigation patterns are shown:

**Radio Buttons:**
```python
page_selection = st.sidebar.radio("Go to", page_options)
if page_selection == "Page 1":
    page1()
```

**Selectbox:**
```python
selected_page = st.sidebar.selectbox("Select Page", page_options)
if selected_page == "Page 1":
    page1()
```

### 4. Modern Pandas APIs
The navbar version demonstrates updated pandas methods:
- `df.ffill()` instead of deprecated `df.fillna(method='ffill')`
- `df.bfill()` instead of deprecated `df.fillna(method='bfill')`

---

## Usage Tips

1. **Start Simple**: Begin with the Radio Button version to understand the basics
2. **Scale Up**: Move to the Navbar version for more complex needs
3. **Customize**: Both are templates you can modify for your use case
4. **Data Format**: Both expect CSV files with numeric and categorical columns

---

## Sample Dataset Format

Your CSV should have:
- Multiple columns (numeric and/or categorical)
- A target column for ML (navbar version)
- Optional missing values to demonstrate handling

Example:
```csv
feature1,feature2,feature3,target_variable_column_name
1.2,A,5.5,0
2.3,B,6.1,1
...
```

---

## Troubleshooting

**Issue**: "Please upload a dataset on the first page"  
**Solution**: Make sure to upload a CSV file on the first page before navigating to other pages.

**Issue**: Missing value handling not working  
**Solution**: Check that your dataset actually has missing values (NaN).

**Issue**: ML models fail (navbar version)  
**Solution**: Ensure your target column name matches the one in the code or select appropriate columns.

---

## Next Steps

After exploring these examples, try:
1. Adding more pages
2. Implementing additional ML models
3. Adding data export functionality
4. Creating custom visualizations
5. Building a real application for your use case

---

## Credits

**Created by:** Sazzad Hussain Farhaan  
**Email:** shfarhaan21@gmail.com

---
*Updated with modern Streamlit and pandas APIs*
