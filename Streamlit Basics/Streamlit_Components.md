Streamlit provides a variety of components and widgets that you can use to build interactive and user-friendly web applications. Here are some of the key components along with examples:

### 1. **st.title()**
Display a title for your app.

```python
import streamlit as st

st.title("My Streamlit App")
```

### 2. **st.header() and st.subheader()**
Create headers and subheaders.

```python
st.header("This is a Header")
st.subheader("This is a Subheader")
```

### 3. **st.text()**
Display plain text.

```python
st.text("Hello, Streamlit!")
```

### 4. **st.markdown()**
Render Markdown text.

```python
st.markdown("## This is a Markdown Header")
st.markdown("This is some *italicized* and **bold** text.")
```

### 5. **st.image()**
Display an image.

```python
from PIL import Image

image = Image.open("example_image.jpg")
st.image(image, caption="Example Image", use_column_width=True)
```

### 6. **st.dataframe()**
Display a DataFrame.

```python
import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
st.dataframe(df)
```

### 7. **st.table()**
Display a static table.

```python
st.table(df)
```

### 8. **st.plotly_chart()**
Render a Plotly chart.

```python
import plotly.express as px

fig = px.scatter(df, x="A", y="B")
st.plotly_chart(fig)
```

### 9. **st.line_chart() and st.bar_chart()**
Quickly render line and bar charts.

```python
st.line_chart(df)
st.bar_chart(df)
```

### 10. **st.pyplot()**
Display a Matplotlib figure.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(df["A"], df["B"])
st.pyplot(fig)
```

### 11. **st.selectbox()**
Create a dropdown select box.

```python
selected_option = st.selectbox("Select an option", ["Option 1", "Option 2", "Option 3"])
```

### 12. **st.checkbox()**
Create a checkbox.

```python
is_checked = st.checkbox("Check me")
```

### 13. **st.radio()**
Create a radio button.

```python
selected_radio = st.radio("Select one", ["Option 1", "Option 2", "Option 3"])
```

### 14. **st.slider()**
Create a slider.

```python
selected_value = st.slider("Select a value", min_value=0, max_value=10, value=5)
```

### 15. **st.button()**
Create a button.

```python
if st.button("Click me"):
    st.success("Button clicked!")
```

### 16. **st.text_input() and st.text_area()**
Get user input through text input and text area.

```python
user_input = st.text_input("Enter text", "Default text")
user_text_area = st.text_area("Enter text here", "Default text")
```

### 17. **st.file_uploader()**
Allow users to upload files.

```python
uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])
```

### 18. **st.sidebar()**
Create a sidebar for additional controls.

```python
with st.sidebar:
    st.title("Sidebar Title")
    selected_sidebar_option = st.radio("Select from sidebar", ["Option 1", "Option 2"])
```

### 19. **st.spinner()**
Display a spinner to indicate loading.

```python
with st.spinner("Loading..."):
    # Perform some time-consuming operation
    time.sleep(5)
    st.success("Operation complete!")
```

### 20. **st.cache()**
Cache the results of a function to improve performance.

```python
@st.cache
def expensive_computation(a, b):
    # Expensive computation here
    return result

result = expensive_computation(5, 10)
```

### 21. **st.empty()**
Create an empty slot to dynamically fill later.

```python
placeholder = st.empty()
# Later in the code
placeholder.text("This will replace the empty slot.")
```

### 22. **st.latex()**
Render LaTeX mathematical expressions.

```python
st.latex(r"\int_0^\infty e^{-x^2} \,dx")
```

### 23. **st.pyplot()**
Render a Matplotlib figure.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
st.pyplot(fig)
```

### 24. **st.text_area()**
Create a multiline text input area.

```python
user_input = st.text_area("Enter your text", "Default text")
```

### 25. **st.write()**
A versatile function that can render text, data, or charts based on the input type.

```python
st.write("This is a simple text.")
st.write(df)  # Renders a DataFrame
st.write(fig)  # Renders a Matplotlib figure
```

### 26. **st.code()**
Render code blocks with syntax highlighting.

```python
code = """
def hello_world():
    print("Hello, World!")

hello_world()
"""
st.code(code, language="python")
```

### 27. **st.download_button()**
Create a download button for downloading files.

```python
download_button = st.download_button("Download CSV", df.to_csv(), key="download_button")
```

### 28. **st.balloons()**
Display celebratory balloons animation.

```python
if st.button("Celebrate"):
    st.balloons()
```

### 29. **st.help()**
Show information or help text.

```python
st.help(pd.DataFrame)
```

### 30. **st.video()**
Embed a video into your app.

```python
st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
```

These components provide a wide range of options for building interactive and engaging Streamlit applications. You can mix and match these components to create a rich and dynamic user interface for your data analysis or machine learning projects. Refer to the official [Streamlit documentation](https://docs.streamlit.io) for more details and examples.