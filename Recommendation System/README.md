### Movie Recommendation System

A content-based movie recommendation system built with Streamlit, demonstrating machine learning techniques for personalized recommendations.

#### Overview
This application implements a content-based filtering approach using TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to recommend movies based on their genres and descriptions.

#### Features

##### Standard Version (app.py)
- 🎬 Content-based movie recommendations
- 📊 TF-IDF vectorization for feature extraction
- 🔍 Cosine similarity for finding similar movies
- 🎯 Adjustable number of recommendations
- 📈 Similarity score display
- 📖 Detailed movie information
- 💾 Sample dataset included (works without external data)

##### Beta Version (beta.py)
- ✨ All features from standard version, plus:
- 🎨 Enhanced UI with wide layout
- 🔎 Advanced search functionality
- 🎭 Genre filtering
- 📅 Year range filter
- ⭐ Minimum rating filter
- 📊 Dataset statistics and visualizations
- 📇 Card-based recommendation display
- 📈 Genre distribution charts

#### How It Works

1. **Feature Engineering**: Combines movie genres and overviews into a single text feature
2. **Vectorization**: Converts text to numerical vectors using TF-IDF
3. **Similarity Calculation**: Computes cosine similarity between all movies
4. **Recommendation**: Returns top-N most similar movies based on selected movie

#### Requirements
```bash
streamlit>=1.39.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
```

#### Installation
```bash
pip install -r requirements.txt
```

#### Running the App

##### Standard Version
```bash
streamlit run app.py
```

##### Beta Version (Enhanced)
```bash
streamlit run beta.py
```

#### Dataset

The application includes a built-in sample dataset with popular movies. You can also provide your own `dataset.csv` file with the following structure:

```csv
title,genres,overview,year,rating
Movie Title,Genre1 Genre2,Movie description here,2020,8.5
```

**Required columns:**
- `title`: Movie name
- `genres`: Space-separated genres
- `overview`: Movie description

**Optional columns:**
- `year`: Release year
- `rating`: Movie rating (0-10)

#### Usage Tips

1. **Selecting a Movie**: Choose from the dropdown or use search (beta version)
2. **Adjusting Recommendations**: Use the sidebar slider to change the number of recommendations
3. **Filtering** (beta only): Apply genre, year, and rating filters for refined results
4. **Exploring Data**: Check the dataset statistics expander for insights

#### Technical Details

- **Algorithm**: Content-Based Filtering
- **Vectorization**: TF-IDF with English stop words removal
- **Similarity Metric**: Cosine Similarity
- **Caching**: Uses `@st.cache_data` for performance optimization

#### Example Use Cases

1. **Movie Discovery**: Find movies similar to your favorites
2. **Genre Exploration**: Filter by genre to discover new categories
3. **Quality Control**: Set minimum rating to get only highly-rated recommendations
4. **Era-Specific**: Filter by year range to find classics or recent releases

#### Comparison: Standard vs Beta

| Feature | Standard | Beta |
|---------|----------|------|
| Basic Recommendations | ✅ | ✅ |
| Similarity Scores | ✅ | ✅ |
| Search Functionality | ❌ | ✅ |
| Genre Filtering | ❌ | ✅ |
| Year Range Filter | ❌ | ✅ |
| Rating Filter | ❌ | ✅ |
| Statistics Dashboard | ❌ | ✅ |
| Enhanced UI | ❌ | ✅ |

#### Future Enhancements

Potential improvements:
- Collaborative filtering integration
- User ratings and feedback
- Watch history tracking
- Movie posters and trailers
- API integration (TMDB, IMDB)
- Hybrid recommendation system
- User profiles

#### Troubleshooting

**Issue**: No dataset.csv found  
**Solution**: The app will automatically use sample data. No action needed.

**Issue**: Recommendations seem off  
**Solution**: The quality depends on the dataset. With the sample data, recommendations are based on limited information. Use a larger dataset for better results.

**Issue**: Slow performance  
**Solution**: The app uses caching. First run might be slow, but subsequent runs will be faster.

#### Credits

**Created by:** Sazzad Hussain Farhaan  
**Email:** shfarhaan21@gmail.com

#### References

- [Content-Based Filtering](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering)
- [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---
*Built with Streamlit 1.39.0+ and scikit-learn 1.5.0+*
