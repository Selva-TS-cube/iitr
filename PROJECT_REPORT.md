# Project Report: Movie Recommendation System

## Student & Project Details

| Field | Details |
|-------|---------|
| **Student Name** | Selvendran |
| **Mentor Name** | [Mentor Name] |
| **Project Title** | Content-Based Movie Recommendation System using TF-IDF and K-Means Clustering |

---

## 1. Problem Statement

**Background:** Streaming platforms offer thousands of movies, causing "choice overload" where users spend more time browsing than watching. This project builds a content-based recommendation system to suggest relevant movies.

**AI Task:** Recommend similar movies by analyzing genres, keywords, cast, and director using TF-IDF vectorization, K-Means clustering, and cosine similarity.

**Objectives:** Build recommendation engine with fuzzy search, hybrid scoring, and genre filtering.

---

## 2. Approach

**Data:** TMDB 5000 dataset (tmdb_5000_movies.csv + tmdb_5000_credits.csv) with ~4,800 movies.

**Pipeline:**
1. Merge datasets → 2. Parse JSON columns (genres, keywords, cast, crew) → 3. Create movie profiles → 4. TF-IDF vectorization (5,000 features) → 5. K-Means clustering (25 clusters) → 6. Cosine similarity for recommendations

**Tools:** Python, pandas, scikit-learn, plotly, Jupyter Notebook

---

## 3. Key Results

**Features Built:**
- Fuzzy movie search (handles typos)
- Content-based recommendations with similarity scores
- Hybrid recommendations (70% content + 30% popularity)
- Genre filtering

**Example Output (Avatar recommendations):**

| Movie | Similarity | Director |
|-------|------------|----------|
| Aliens | 0.334 | James Cameron |
| Guardians of Galaxy | 0.272 | James Gunn |
| Star Trek Into Darkness | 0.245 | J.J. Abrams |

**Performance:** <50ms for 10 recommendations, 95% search space reduction via clustering

**Limitations:** No user personalization (content-only), English movies only, static model

---

## 4. Learnings

**Technical:** TF-IDF for text vectorization, K-Means for grouping, cosine similarity for sparse vectors, hybrid scoring for robustness.

**Challenges & Solutions:**
- High-dimensional data → max_features limit
- Typos in search → difflib fuzzy matching
- Popularity bias → hybrid scoring

**Future:** Add collaborative filtering, deep learning embeddings, web interface

---

## References & AI Disclosure

**Dataset:** TMDB Movie Metadata (Kaggle)

**Tools:** scikit-learn, pandas, plotly, Jupyter

**AI Tools Used:** Google Gemini (Antigravity) for code generation and documentation assistance. Core algorithm design done independently.
