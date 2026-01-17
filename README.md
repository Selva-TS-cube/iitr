# 🎬 Movie Recommendation System

Ever spent more time scrolling through Netflix than actually watching something? This project solves that problem! It's a smart movie recommendation engine that suggests films you'll actually want to watch based on what you already love.

## What Does It Do?

Give it a movie you like, and it'll find similar ones by looking at:

- **Genres** - Action, Comedy, Drama, etc.
- **Keywords** - Themes and plot elements
- **Cast** - Your favorite actors
- **Director** - The creative vision behind the film

### Cool Features

🔍 **Typo-Friendly Search** - Type "Avtar" instead of "Avatar"? No problem, it figures it out!

🎯 **Smart Recommendations** - Uses machine learning to find genuinely similar movies, not just random suggestions

⚡ **Fast Results** - Gets you 10 recommendations in under 50 milliseconds

🎭 **Genre Filters** - Want only action movies? Easy!

📊 **Popularity Boost** - Can mix in popular movies so you don't just get obscure films

---

## Quick Start

### 1. Set Up

```bash
cd "Content recommendation sys"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook recommender/movie-recomendation.ipynb
```

### 3. Get Recommendations

Run all cells in the notebook, then try:

```python
# Find movies like Avatar
recommender.recommend('Avatar', top_n=10)

# Works even with typos!
recommender.recommend('Avtar')

# Mix in popular movies
recommender.recommend('The Dark Knight', hybrid=True)

# Only want Action movies?
recommender.recommend('Avatar', genre='Action')
```

---

## How It Works (The Simple Version)

1. **Loads movie data** - ~4,800 movies from TMDB with all their details
2. **Creates a "profile"** for each movie by combining genres, keywords, cast, and director
3. **Converts text to numbers** using TF-IDF (a fancy way of weighing important words)
4. **Groups similar movies** into 25 clusters using K-Means
5. **Finds the closest matches** when you ask for recommendations

Think of it like this: movies are organized into 25 "neighborhoods" based on their characteristics. When you pick a movie, the system looks at its neighbors to find similar ones.

---

## Project Files

```
📁 Content recommendation sys/
├── 📁 data/
│   ├── tmdb_5000_movies.csv    ← Movie info (genres, ratings, etc.)
│   └── tmdb_5000_credits.csv   ← Cast and crew info
├── 📁 recommender/
│   └── movie-recomendation.ipynb  ← The main notebook
├── requirements.txt
├── README.md
└── PROJECT_REPORT.md
```

---

## Example Output

Ask for movies like **Avatar**:

| Movie                   | Why It Matched                        | Rating |
| ----------------------- | ------------------------------------- | ------ |
| Aliens                  | Same director (James Cameron), Sci-Fi | 7.9    |
| Guardians of the Galaxy | Similar cast/genre vibes              | 7.9    |
| Star Trek Into Darkness | Space adventure, action               | 7.4    |

---

## What's Under the Hood?

For the curious minds:

- **TF-IDF Vectorization** - Turns movie descriptions into math
- **K-Means Clustering** - Groups 4,800 movies into 25 manageable clusters
- **Cosine Similarity** - Measures how "close" two movies are
- **Fuzzy Matching** - Handles typos using Python's difflib

---

## What Could Be Better?

This is a content-based system, meaning it looks at movie characteristics only. It doesn't know what YOU personally like - it just knows what movies are similar to each other.

**Future ideas:**

- Add user ratings for personalized recommendations
- Include movie descriptions for deeper understanding
- Build a nice web interface

---

## Requirements

- Python 3.8+
- pandas, numpy
- scikit-learn
- plotly
- jupyter

All listed in `requirements.txt` - just run `pip install -r requirements.txt`

---

## Dataset

Uses the **TMDB 5000 Movie Dataset** from Kaggle - a great collection of movie metadata including cast, crew, genres, and ratings.

---

Made by **Selvendran S - TS Techy**
