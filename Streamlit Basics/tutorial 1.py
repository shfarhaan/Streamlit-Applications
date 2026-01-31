"""
Streamlit Tutorial 1: Basic Components

This tutorial demonstrates the basic components and widgets available in Streamlit.
It covers text elements, data display, interactive widgets, and basic layouts.
"""

import streamlit as st
import pandas as pd
import numpy as np

def main():
    # Title and header
    st.title("Streamlit Tutorial 1: Basic Components")
    st.header("Welcome to Streamlit!")
    st.subheader("Learn the basics of Streamlit components")
    
    # Text elements
    st.markdown("---")
    st.markdown("## Text Elements")
    st.text("This is simple text")
    st.markdown("This is **markdown** text with *italics*")
    st.caption("This is a caption - smaller text for annotations")
    
    # Code display
    st.markdown("---")
    st.markdown("## Code Display")
    code = '''def hello():
    print("Hello, Streamlit!")'''
    st.code(code, language='python')
    
    # Data display
    st.markdown("---")
    st.markdown("## Data Display")
    
    # Create sample dataframe
    df = pd.DataFrame({
        'Column A': [1, 2, 3, 4, 5],
        'Column B': [10, 20, 30, 40, 50],
        'Column C': ['A', 'B', 'C', 'D', 'E']
    })
    
    st.write("Display dataframe:")
    st.dataframe(df)
    
    st.write("Display as table:")
    st.table(df.head(3))
    
    # Metrics
    st.markdown("---")
    st.markdown("## Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", "25°C", "1.2°C")
    col2.metric("Wind", "12 km/h", "-3 km/h")
    col3.metric("Humidity", "75%", "5%")
    
    # Interactive widgets
    st.markdown("---")
    st.markdown("## Interactive Widgets")
    
    # Text input
    name = st.text_input("Enter your name:", "")
    if name:
        st.write(f"Hello, {name}!")
    
    # Number input
    age = st.number_input("Enter your age:", min_value=0, max_value=120, value=25)
    st.write(f"You are {age} years old")
    
    # Slider
    value = st.slider("Select a value", 0, 100, 50)
    st.write(f"Selected value: {value}")
    
    # Select box
    option = st.selectbox(
        'Choose your favorite color:',
        ('Red', 'Green', 'Blue', 'Yellow')
    )
    st.write(f"You selected: {option}")
    
    # Multiselect
    options = st.multiselect(
        'Choose your favorite programming languages:',
        ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust'],
        ['Python']
    )
    st.write('You selected:', options)
    
    # Checkbox
    if st.checkbox('Show/Hide data'):
        st.write(df)
    
    # Radio buttons
    genre = st.radio(
        "Choose your favorite genre:",
        ('Comedy', 'Drama', 'Documentary', 'Action')
    )
    st.write(f"You selected: {genre}")
    
    # Button
    if st.button('Click me!'):
        st.success('Button clicked!')
        st.balloons()
    
    # File uploader
    st.markdown("---")
    st.markdown("## File Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.dataframe(data.head())
    
    # Charts
    st.markdown("---")
    st.markdown("## Simple Charts")
    
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )
    
    st.line_chart(chart_data)
    st.area_chart(chart_data)
    st.bar_chart(chart_data)
    
    # Status elements
    st.markdown("---")
    st.markdown("## Status Elements")
    
    st.success("This is a success message!")
    st.info("This is an info message")
    st.warning("This is a warning message")
    st.error("This is an error message")
    
    # Expander
    st.markdown("---")
    st.markdown("## Container Elements")
    
    with st.expander("Click to expand"):
        st.write("Hidden content revealed!")
        st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)
    
    # Sidebar
    st.sidebar.title("Sidebar")
    st.sidebar.write("This is the sidebar")
    sidebar_option = st.sidebar.selectbox(
        'Select an option:',
        ('Option 1', 'Option 2', 'Option 3')
    )
    st.sidebar.write(f"Selected: {sidebar_option}")
    
    # Footer
    st.markdown("---")
    st.markdown("### Tutorial Complete!")
    st.markdown("Explore more at [Streamlit Documentation](https://docs.streamlit.io)")

if __name__ == "__main__":
    main()
