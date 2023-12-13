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

Certainly! Let's continue with more Streamlit components:

### 31. **st.warning() and st.error()**
Display warning and error messages.

```python
st.warning("This is a warning message.")
st.error("Oops! An error occurred.")
```

### 32. **st.success()**
Display a success message.

```python
st.success("Operation was successful!")
```

### 33. **st.write() - Displaying Images**
Display images using `st.write()`.

```python
from PIL import Image

image = Image.open("example_image.jpg")
st.write("Here's an image:")
st.image(image, caption="Example Image", use_column_width=True)
```

### 34. **st.text_input() - Password Input**
Create a password input field.

```python
password = st.text_input("Enter Password", type="password")
```

### 35. **st.select_slider()**
Create a slider with discrete values.

```python
selected_value = st.select_slider("Select a value", options=[1, 2, 3, 4, 5])
```

### 36. **st.multiselect()**
Create a multiselect dropdown.

```python
selected_options = st.multiselect("Select options", ["Option 1", "Option 2", "Option 3"])
```

### 37. **st.date_input() and st.time_input()**
Get date and time input from the user.

```python
selected_date = st.date_input("Select a date")
selected_time = st.time_input("Select a time")
```

### 38. **st.file_uploader() - Multiple Files**
Allow users to upload multiple files.

```python
uploaded_files = st.file_uploader("Choose files", type=["csv", "txt"], accept_multiple_files=True)
```

### 39. **st.progress()**
Display a progress bar.

```python
progress = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    progress.progress(percent_complete + 1)
```

### 40. **st.write() - Audio**
Embed an audio file into your app.

```python
audio_file = open("example_audio.mp3", "rb").read()
st.audio(audio_file, format="audio/mp3")
```

### 41. **st.write() - Video with Loop**
Embed a video and set it to loop.

```python
video_file = open("example_video.mp4", "rb").read()
st.video(video_file, format="video/mp4", start_time=0)
```

### 42. **st.beta_columns()**
Create multiple columns for layout.

```python
col1, col2 = st.beta_columns(2)
with col1:
    st.write("Content of column 1")
with col2:
    st.write("Content of column 2")
```

### 43. **st.radio() - Horizontal Orientation**
Create a horizontal radio button group.

```python
selected_option = st.radio("Select one", ["Option 1", "Option 2", "Option 3"], format_func=lambda x: f"{x}")
```

### 44. **st.expander()**
Create an expandable/collapsible container.

```python
with st.expander("See more"):
    st.write("This is hidden by default.")
```

### 45. **st.echo()**
Display code and its output.

```python
code = """
def hello_world():
    print("Hello, World!")

hello_world()
"""
with st.echo():
    # This code and its output will be displayed
    exec(code)
```

These additional components provide even more flexibility and functionality for creating interactive Streamlit applications. Experiment with these features to tailor your app to your specific needs. Streamlit's documentation is an excellent resource for exploring advanced usage and customization: [Streamlit Documentation](https://docs.streamlit.io).