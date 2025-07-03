import pandas as pd
from recommend_by_movies import recommend_from_liked_movies
from recommend_by_feeling import recommend_by_feeling

MOVIE_PATH = "data/movies_metadata.csv"

def clean_data(df):
    if 'overview' not in df.columns:
        raise Exception("movies.csv must have an 'overview' column.")
    df = df[['title', 'overview']]  # only keep what's needed
    return df.dropna(subset=['title', 'overview'])

def main():
    df = pd.read_csv(MOVIE_PATH)
    df = clean_data(df)

    print("🎬 Movie Recommender")
    print("1. Recommend based on movies you liked")
    print("2. Recommend based on your mood/feeling")
    choice = input("Choose 1 or 2: ")

    if choice == "1":
        liked = input("Enter movies you liked (comma-separated):\n> ")
        recs = recommend_from_liked_movies(liked, df)
    elif choice == "2":
        mood = input("Describe what you're in the mood for:\n> ")
        recs = recommend_by_feeling(mood, df)
    else:
        print("Invalid choice.")
        return

    print("\n🍿 Recommended Movies:")
    for i, movie in enumerate(recs, 1):
        print(f"{i}. {movie}")

if __name__ == "__main__":
    main()
