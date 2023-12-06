### Step 1: Install Streamlit
Make sure you have Python installed on your system. You can install Streamlit using pip:

```bash
pip install streamlit
```

### Step 2: Create a Python Script
Create a new Python script, for example, `app.py`. This script will contain your Streamlit application.

### Step 3: Import Streamlit and Create a Basic App
Open `app.py` and import the Streamlit library. Create a basic Streamlit app with a title and a simple text.

```python
import streamlit as st

# Set page title
st.title("My Streamlit App")

# Add text to the app
st.write("This is a simple Streamlit app.")
```

### Step 4: Run Your App Locally
Open a terminal, navigate to the directory containing `app.py`, and run the following command:

```bash
streamlit run app.py
```

This will launch a local server, and you can view your app in a web browser at `http://localhost:8501`. Make sure to replace `app.py` with your actual script name if it's different.

### Step 5: Add Interactivity
Enhance your app by adding interactive elements. For example, you can use sliders, buttons, and text input fields. Update `app.py`:

```python
import streamlit as st

# Set page title
st.title("Interactive Streamlit App")

# Add a slider
value = st.slider("Select a value", 0, 100, 50)

# Add a button
if st.button("Click me"):
    st.write(f"You selected: {value}")
```

### Step 6: Display Data
You can easily display charts and tables using Streamlit. Add the following to your script:

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page title
st.title("Data Visualization with Streamlit")

# Generate random data
data = pd.DataFrame(np.random.randn(100, 2), columns=['A', 'B'])

# Display data table
st.dataframe(data)

# Display a line chart
st.line_chart(data)
```

### Step 7: Deploy Your App
Once you're satisfied with your app, you can deploy it to platforms like Streamlit Sharing, Heroku, or any other hosting service.

Congratulations! You've created a simple Streamlit app with basic interactivity and data visualization. Feel free to explore more features and widgets in the [Streamlit documentation](https://docs.streamlit.io).