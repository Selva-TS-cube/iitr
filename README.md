# 🎬 Movie Recommendation System

A content-based movie recommendation system using TF-IDF vectorization, K-Means clustering, and cosine similarity. This project demonstrates machine learning techniques for building personalized movie recommendations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Architecture](#technical-architecture)
- [Algorithm Details](#algorithm-details)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Performance Metrics](#performance-metrics)
- [Future Enhancements](#future-enhancements)

---

## Overview

This movie recommendation system uses **content-based filtering** to suggest movies similar to a user's preferences. It analyzes movie attributes such as genres, cast, and director to find similarities between films.

### Key Highlights

| Feature | Description |
|---------|-------------|
| **4,700+ Movies** | Comprehensive dataset with rich metadata |
| **25 Clusters** | K-Means clustering for efficient similarity search |
| **Hybrid Scoring** | Combines content similarity with popularity |
| **Fuzzy Search** | Handles typos and partial movie names |

---

## Features

### 🔍 Smart Movie Search
- **Fuzzy Matching**: Find movies even with typos (e.g., "Avtar" → "Avatar")
- **Case Insensitive**: Search works regardless of capitalization
- **Suggestions**: Get alternative movie suggestions if exact match not found

### 🎯 Recommendation Types

1. **Content-Based**: Pure similarity based on genres, cast, and director
2. **Hybrid**: Combines content similarity (70%) with popularity score (30%)
3. **Filtered**: Filter recommendations by genre or release year range

### 📊 Rich Analytics
- Movie release trends over time
- Genre distribution analysis
- Top directors and highest-rated films
- Budget vs Revenue insights
- Cluster visualization using PCA

---

## Project Structure

```
Content recommendation sys/
├── data/
│   └── movies.csv              # Movie dataset (4,800+ movies)
├── recommender/
│   ├── __init__.py             # Package initialization
│   └── movie-recomendation.ipynb  # Main recommendation notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
└── venv/                       # Virtual environment
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd "Content recommendation sys"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook recommender/movie-recomendation.ipynb
   ```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.5.0 | Data manipulation |
| numpy | ≥1.23.0 | Numerical operations |
| scikit-learn | ≥1.2.0 | ML algorithms (TF-IDF, KMeans, PCA) |
| matplotlib | ≥3.6.0 | Static visualizations |
| seaborn | ≥0.12.0 | Statistical plots |
| plotly | ≥5.11.0 | Interactive visualizations |
| jupyter | ≥1.0.0 | Notebook environment |

---

## Usage

### Quick Start

```python
# After running all cells in the notebook, use the recommender:

# Basic recommendation
recommender.get_recommendations('Avatar', top_n=10)

# With fuzzy search (handles typos)
recommender.get_recommendations('Avtar', top_n=5)

# Hybrid recommendation (content + popularity)
recommender.get_recommendations('The Dark Knight', hybrid=True)

# Filter by genre
recommender.get_recommendations('Avatar', genre_filter='Action')

# Filter by year range
recommender.get_recommendations('Inception', year_range=(2010, 2020))
```

### Running the Notebook

1. Open `recommender/movie-recomendation.ipynb` in Jupyter
2. Run all cells sequentially (Kernel → Restart & Run All)
3. Use the `recommender` object to get recommendations

---

## Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Movie Recommendation System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Raw Data   │───▶│ Preprocessing │───▶│ Feature Engineering│   │
│  │  (movies.csv)│    │  & Cleaning   │    │  (profile creation)│   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                                                   │               │
│                                                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Cosine      │◀───│   TF-IDF     │◀───│   Movie Profile   │   │
│  │  Similarity  │    │   Matrix     │    │ (genres+cast+dir) │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│         │                   │                                     │
│         │                   ▼                                     │
│         │            ┌──────────────┐                            │
│         │            │   K-Means    │                            │
│         │            │  Clustering  │                            │
│         │            │ (25 clusters)│                            │
│         │            └──────────────┘                            │
│         │                   │                                     │
│         ▼                   ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              MovieRecommender Class                       │    │
│  │  • find_movie() - Fuzzy search                           │    │
│  │  • get_recommendations() - Main recommendation method    │    │
│  │  • explain_recommendation() - Why this was recommended   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Loading**: Load movies.csv with metadata
2. **Preprocessing**: Handle missing values, extract features
3. **Feature Engineering**: Create movie profiles, calculate profit/ROI
4. **Vectorization**: Convert text to TF-IDF vectors
5. **Clustering**: Group similar movies using K-Means
6. **Recommendation**: Find similar movies using cosine similarity

---

## Algorithm Details

### 1. TF-IDF Vectorization

**Term Frequency-Inverse Document Frequency** converts movie profiles (text) into numerical vectors.

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

Where:
- TF(t, d) = Count of term t in document d / Total terms in d
- IDF(t) = log(Total documents / Documents containing t)
```

**Configuration:**
- Stop words: English (removed common words like "the", "a")
- Max features: 5,000 (vocabulary size)
- N-grams: Unigrams (single words)

### 2. K-Means Clustering

Groups movies into **25 clusters** based on TF-IDF similarity.

```
Algorithm Steps:
1. Initialize 25 random centroids
2. Assign each movie to nearest centroid
3. Update centroids to cluster means
4. Repeat until convergence
```

**Why Clustering?**
- Reduces search space from 4,700+ to ~190 movies per cluster
- Improves recommendation speed
- Groups thematically similar movies

### 3. Cosine Similarity

Measures angle between two movie vectors (0 to 1).

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product of vectors
- ||A|| = magnitude of vector A
```

**Interpretation:**
- 1.0 = Identical movies
- 0.5 = Moderately similar
- 0.0 = Completely different

### 4. Hybrid Scoring

Combines content similarity with popularity:

```python
hybrid_score = (content_weight × similarity) + 
               ((1 - content_weight) × popularity_score)

Default: content_weight = 0.7 (70% content, 30% popularity)
```

**Popularity Score:**
```python
popularity_score = (vote_count × vote_average) / max_popularity
```

---

## API Reference

### MovieRecommender Class

#### `__init__(df, tfidf_matrix)`

Initialize the recommender with movie data and TF-IDF matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| df | DataFrame | Movie dataset with required columns |
| tfidf_matrix | sparse matrix | Pre-computed TF-IDF vectors |

---

#### `find_movie(title, threshold=0.6)`

Find a movie by title using fuzzy matching.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| title | str | - | Movie title to search |
| threshold | float | 0.6 | Minimum match similarity (0-1) |

**Returns:** `(found_title, index)` or `(None, None)` if not found

---

#### `get_recommendations(title, top_n=10, genre_filter=None, year_range=None, hybrid=False, content_weight=0.7)`

Get movie recommendations based on a given movie.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| title | str | - | Movie title to get recommendations for |
| top_n | int | 10 | Number of recommendations to return |
| genre_filter | str | None | Filter by genre (e.g., "Action") |
| year_range | tuple | None | (min_year, max_year) filter |
| hybrid | bool | False | Use hybrid scoring |
| content_weight | float | 0.7 | Content weight in hybrid mode |

**Returns:** DataFrame with columns:
- `original_title`: Movie name
- `genres`: Movie genres
- `director`: Director name
- `release_year`: Year of release
- `vote_average`: Average rating
- `similarity`: Content similarity score
- `hybrid_score`: (only if hybrid=True)

---

#### `explain_recommendation(source_movie, target_movie)`

Explain why a movie was recommended.

| Parameter | Type | Description |
|-----------|------|-------------|
| source_movie | str | Original movie |
| target_movie | str | Recommended movie |

**Returns:** Dictionary with:
- `source_movie`: Input movie name
- `recommended_movie`: Recommended movie name
- `similarity_score`: Cosine similarity (0-1)
- `shared_genres`: Set of common genres
- `same_director`: Boolean
- `same_cluster`: Boolean

---

## Examples

### Example 1: Basic Recommendation

```python
>>> recommender.get_recommendations('Avatar', top_n=5)

   original_title              genres         director  release_year  vote_average  similarity
0  Aliens                      Action Sci-Fi  James Cameron  1986         7.9        0.334
1  Guardians of Galaxy         Action Sci-Fi  James Gunn     2014         7.9        0.272
2  Star Trek Into Darkness     Action Sci-Fi  J.J. Abrams    2013         7.4        0.245
3  Star Trek Beyond            Action Sci-Fi  Justin Lin     2016         6.9        0.241
4  Alien                       Horror Sci-Fi  Ridley Scott   1979         8.1        0.211
```

### Example 2: Hybrid Recommendation

```python
>>> recommender.get_recommendations('The Dark Knight', top_n=5, hybrid=True)

# Results weighted by both similarity AND popularity
# Popular movies rank higher than pure content-based
```

### Example 3: Filtered Recommendations

```python
# Only Action movies from 2010-2020
>>> recommender.get_recommendations(
...     'Inception', 
...     genre_filter='Action', 
...     year_range=(2010, 2020)
... )
```

### Example 4: Recommendation Explanation

```python
>>> recommender.explain_recommendation('Avatar', 'Aliens')

{
    'source_movie': 'Avatar',
    'recommended_movie': 'Aliens',
    'similarity_score': 0.334,
    'shared_genres': {'Action', 'Science Fiction'},
    'same_director': True,
    'same_cluster': True
}
```

---

## Performance Metrics

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Movies | 4,738 |
| Features (TF-IDF) | 5,000 |
| Clusters | 25 |
| Avg Movies/Cluster | ~190 |

### Clustering Quality

| Metric | Value |
|--------|-------|
| Min Cluster Size | ~50 |
| Max Cluster Size | ~400 |
| Mean Cluster Size | ~190 |

### Recommendation Speed

| Operation | Time |
|-----------|------|
| Find Movie | <10ms |
| Get 10 Recommendations | <50ms |
| Full Search (no cluster) | ~500ms |

---

## Future Enhancements

### Planned Features

1. **Collaborative Filtering**
   - Add user ratings data
   - Implement matrix factorization (SVD)
   - Combine with content-based for better accuracy

2. **Deep Learning Embeddings**
   - Use Word2Vec/Doc2Vec for movie descriptions
   - Implement neural collaborative filtering

3. **Web Interface**
   - Build Flask/FastAPI backend
   - Create React frontend
   - Add movie poster integration

4. **Real-time Updates**
   - Handle new movies without full retraining
   - Incremental cluster updates

### Research Directions

- [ ] Evaluate with precision/recall metrics
- [ ] A/B test hybrid vs pure content-based
- [ ] Explore graph-based recommendations
- [ ] Add diversity in recommendations

---

## License

This project is for educational purposes.

---

## Acknowledgments

- Dataset: TMDB Movie Metadata
- Libraries: scikit-learn, pandas, plotly

---

*Last Updated: January 2026*
