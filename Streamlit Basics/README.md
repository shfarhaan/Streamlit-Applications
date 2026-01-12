### Streamlit Basics Tutorial

A comprehensive guide to learning Streamlit from scratch.

#### Overview
This tutorial provides a complete introduction to Streamlit, covering all essential components and best practices. Perfect for beginners who want to build interactive web applications with Python.

#### What You'll Learn

##### 1. Text Elements
- Titles and headers
- Markdown formatting
- Text and captions
- Code display

##### 2. Data Display
- DataFrames and tables
- Metrics and statistics
- Data visualization

##### 3. Interactive Widgets
- Text and number inputs
- Sliders and select boxes
- Checkboxes and radio buttons
- Buttons with callbacks
- File uploaders
- Multi-select dropdowns

##### 4. Visualizations
- Line charts
- Area charts
- Bar charts
- Integration with Matplotlib/Seaborn

##### 5. Layout Elements
- Columns for side-by-side content
- Expanders for collapsible sections
- Sidebar for navigation

##### 6. Status Messages
- Success, info, warning, and error messages
- Special effects (balloons, snow)

#### Requirements
```bash
streamlit>=1.39.0
pandas>=2.2.0
numpy>=1.26.0
```

#### Installation

1. Install Streamlit and dependencies:
```bash
pip install -r requirements.txt
```

#### Running the Tutorial

Navigate to the Streamlit Basics directory and run:

```bash
streamlit run "tutorial 1.py"
```

This will launch the tutorial in your default web browser at `http://localhost:8501`.

#### Tutorial Structure

The tutorial is organized into sections:
1. **Text Elements** - Learn how to display different types of text
2. **Data Display** - Work with DataFrames and tables
3. **Interactive Widgets** - Add user input controls
4. **Charts** - Create data visualizations
5. **Status Elements** - Provide user feedback
6. **Container Elements** - Organize your layout

#### Tips for Learning

1. **Experiment**: Modify the code and see what happens
2. **Read the comments**: Each section is well-documented
3. **Try uploading files**: Use sample CSV files to test the file uploader
4. **Customize**: Change colors, sizes, and options to make it your own
5. **Check the sidebar**: Many controls are demonstrated there

#### Next Steps

After completing this tutorial, try:
1. Building your own data dashboard
2. Creating a machine learning model interface
3. Exploring the other applications in this repository
4. Reading the [official Streamlit documentation](https://docs.streamlit.io)

#### Additional Resources

- **Streamlit Components**: See `Streamlit_Components.md` for detailed component reference
- **Official Docs**: https://docs.streamlit.io
- **API Reference**: https://docs.streamlit.io/library/api-reference
- **Community Forum**: https://discuss.streamlit.io
- **Gallery**: https://streamlit.io/gallery

#### Common Commands

```bash
# Run the tutorial
streamlit run "tutorial 1.py"

# Run with auto-reload on file changes
streamlit run "tutorial 1.py" --server.runOnSave true

# Run on a different port
streamlit run "tutorial 1.py" --server.port 8502

# Get help
streamlit --help
```

#### Contributing

Found an issue or have a suggestion? Feel free to:
1. Open an issue on GitHub
2. Submit a pull request
3. Contact the maintainer

#### About

**Created by:** Sazzad Hussain Farhaan  
**Contact:** shfarhaan21@gmail.com

---
*Updated with Streamlit 1.39.0+ features and modern Python practices*