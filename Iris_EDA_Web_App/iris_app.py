"""
## App: Iris EDA App

Description
This is a simple Exploratory Data Analysis of the Iris Dataset depicting the various 
species built with Streamlit.
We can preview the dataset,column names as well as show some basic plot with matplotlib and
seaborn.
There is also an image manipulation of a specie with changeable contrast and width using st.slider()

Purpose
To show a simple EDA of Iris using Streamlit framework. 

"""
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np
import os

# Plotting Pkgs
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance


def main():
    st.title("Iris EDA App")
    st.subheader("EDA Web App with Streamlit ")
    st.markdown("""
    	#### Description
    	+ This is a simple Exploratory Data Analysis  of the Iris Dataset depicting the various species built with Streamlit.

    	#### Purpose
    	+ To show a simple EDA of Iris using Streamlit framework. 
    	""")

    # Your code goes below
    # Our Dataset
    my_dataset = "iris.csv"

    # To Improve speed and cache data
    @st.cache_data
    def explore_data(dataset):
        df = pd.read_csv(os.path.join(dataset))
        return df

        # Load Our Dataset

    data = explore_data(my_dataset)

    # Show Dataset
    if st.checkbox("Preview DataFrame"):
        if st.button("Head"):
            st.write(data.head())
        if st.button("Tail"):
            st.write(data.tail())
        else:
            st.write(data.head(2))

    # Show Entire Dataframe
    if st.checkbox("Show All DataFrame"):
        st.dataframe(data)

    # Show All Column Names
    if st.checkbox("Show All Column Name"):
        st.text("Columns:")
        st.write(data.columns)

    # Show Dimensions and Shape of Dataset
    data_dim = st.radio('What Dimension Do You Want to Show', ('Rows', 'Columns'))
    if data_dim == 'Rows':
        st.text("Showing Length of Rows")
        st.write(len(data))
    if data_dim == 'Columns':
        st.text("Showing Length of Columns")
        st.write(data.shape[1])

    # Show Summary of Dataset
    if st.checkbox("Show Summary of Dataset"):
        st.write(data.describe())

    # Selection of Columns
    species_option = st.selectbox('Select Columns',
                                  ('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'))
    if species_option == 'sepal_length':
        st.write(data['sepal_length'])
    elif species_option == 'sepal_width':
        st.write(data['sepal_width'])
    elif species_option == 'petal_length':
        st.write(data['petal_length'])
    elif species_option == 'petal_width':
        st.write(data['petal_width'])
    elif species_option == 'species':
        st.write(data['species'])
    else:
        st.write("Select A Column")

    # Show Plots
    if st.checkbox("Simple Bar Plot with Matplotlib "):
        fig, ax = plt.subplots()
        data.plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # Show Correlation Plots
    if st.checkbox("Simple Correlation Plot with Matplotlib "):
        fig = plt.figure()
        plt.matshow(data.corr(), fignum=fig.number)
        st.pyplot(fig)

    # Show Correlation Plots with Sns
    if st.checkbox("Simple Correlation Plot with Seaborn "):
        fig, ax = plt.subplots()
        sns.heatmap(data.corr(), annot=True, ax=ax)
        # Use Matplotlib to render seaborn
        st.pyplot(fig)

    # Show Plots
    if st.checkbox("Bar Plot of Groups or Counts"):
        v_counts = data.groupby('species')
        st.bar_chart(v_counts)

    # Iris Image Manipulation
    @st.cache_data
    def load_image(img):
        im = Image.open(os.path.join(img))
        return im

    # Select Image Type using Radio Button
    species_type = st.radio('What is the Iris Species do you want to see?', ('Setosa', 'Versicolor', 'Virginica'))

    if species_type == 'Setosa':
        st.text("Showing Setosa Species")
        st.image(load_image('imgs/iris_setosa.jpg'))
    elif species_type == 'Versicolor':
        st.text("Showing Versicolor Species")
        st.image(load_image('imgs/iris_versicolor.jpg'))
    elif species_type == 'Virginica':
        st.text("Showing Virginica Species")
        st.image(load_image('imgs/iris_virginica.jpg'))

    # Show Image or Hide Image with Checkbox
    if st.checkbox("Show Image/Hide Image"):
        my_image = load_image('iris_setosa.jpg')
        enh = ImageEnhance.Contrast(my_image)
        num = st.slider("Set Your Contrast Number", 1.0, 3.0)
        img_width = st.slider("Set Image Width", 300, 500)
        st.image(enh.enhance(num), width=img_width)



if __name__ == "__main__":
    main()
