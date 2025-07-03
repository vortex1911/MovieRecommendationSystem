import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embeddings.embedder import embed_text

def recommend_from_liked_movies(movie_titles, movies_df, top_n=5):
    movies_df = movies_df.dropna(subset=['overview'])
    movie_titles = [t.strip().lower() for t in movie_titles.split(",")]

    liked_plots = []
    for title in movie_titles:
        match = movies_df[movies_df['title'].str.lower() == title]
        if not match.empty:
            liked_plots.append(match.iloc[0]['overview'])

    if not liked_plots:
        return ["No matching titles found in dataset."]

    liked_vecs = np.array([embed_text(plot) for plot in liked_plots])
    preference_vector = np.mean(liked_vecs, axis=0)

    all_embeddings = np.array([embed_text(text) for text in movies_df['overview']])
    similarities = cosine_similarity([preference_vector], all_embeddings)[0]

    top_indices = similarities.argsort()[-top_n-1:][::-1]

    results = []
    for i in top_indices:
        title = movies_df.iloc[i]['title']
        if title.lower() not in movie_titles:
            results.append(title)
        if len(results) == top_n:
            break
    return results
