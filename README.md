# 🎬 Movie Recommendation System (NLP + ML Based)

This is a smart, interactive movie recommendation system built using Natural Language Processing (NLP) and traditional machine learning concepts.

It can suggest movies in two powerful ways:
1. Based on a list of movies you previously liked
2. Based on your mood or a descriptive phrase (e.g., "rainy day thriller", "group stuck in a house")

---

## 🚀 Features

- 📚 Uses `Sentence-BERT` to understand movie plot overviews
- 🔍 Recommends based on semantic similarity (not just keywords)
- 🧠 Supports both liked-movie input & free-form natural language input
- ⚡ Caches embeddings to speed up performance
- 🧪 CLI-based interface for fast experimentation

---

## 🗂️ Project Structure

.
├── main.py # Main CLI application

├── recommend_by_movies.py # Recommender based on liked titles

├── recommend_by_feeling.py # Recommender based on mood/feeling

├── embeddings/
│ └── embedder.py # Embedding logic using SentenceTransformers

├── data/
│ ├── movies_metadata.csv # Your input dataset (from Kaggle)
│ └── movie_vectors.npy # Cached embeddings (auto-generated)

└── README.md

