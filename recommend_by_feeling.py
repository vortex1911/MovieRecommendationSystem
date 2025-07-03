import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embeddings.embedder import embed_text

def recommend_by_feeling(feeling_text, movies_df, top_n=5):
    movies_df = movies_df.dropna(subset=['overview'])
    input_vec = embed_text(feeling_text)
    all_vecs = np.array([embed_text(text) for text in movies_df['overview']])
    similarities = cosine_similarity([input_vec], all_vecs)[0]

    top_indices = similarities.argsort()[-top_n:][::-1]
    return movies_df.iloc[top_indices]['title'].tolist()
