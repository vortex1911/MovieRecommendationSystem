import os
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

_model = None
VEC_PATH = "data/movie_vectors.npy"

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def embed_text(text):
    return get_embedder().encode(text, convert_to_numpy=True)

def get_all_movie_embeddings(overviews):
    if os.path.exists(VEC_PATH):
        return np.load(VEC_PATH)

    print("Embedding all movie overviews. This may take a few minutes...")
    vectors = np.array([embed_text(text) for text in tqdm(overviews)])
    np.save(VEC_PATH, vectors)
    return vectors
