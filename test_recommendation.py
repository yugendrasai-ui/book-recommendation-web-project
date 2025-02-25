import sys
import os
import pickle
import pandas as pd
import traceback

# Add the project directory to Python path
project_dir = r'F:\book recomendation web'
sys.path.append(project_dir)

# Import the BookRecommendationSystem class
from book_recommendation_system import BookRecommendationSystem

def test_recommendation_system():
    try:
        # Paths
        model_path = os.path.join(project_dir, 'models', 'book_recommendation_model.pk3')
        data_path = os.path.join(project_dir, 'data', 'data.csv')

        # Print paths
        print(f"Model Path: {model_path}")
        print(f"Data Path: {data_path}")

        # Verify file existence
        print(f"Model file exists: {os.path.exists(model_path)}")
        print(f"Data file exists: {os.path.exists(data_path)}")

        # Load the model
        with open(model_path, 'rb') as f:
            recommender = pickle.load(f)
        print("Model loaded successfully!")

        # Load the dataset
        books_df = pd.read_csv(data_path)
        print(f"Dataset loaded. Total books: {len(books_df)}")

        # Test recommendations
        print("\nTesting Science Fiction Recommendations:")
        sci_fi_recs = recommender.recommend(
            filtered_df=books_df,
            genre='Science fiction', 
            top_n=5
        )
        print(sci_fi_recs[['title', 'authors', 'categories', 'average_rating']])

        print("\nTesting Recommendations with Author Preference:")
        author_recs = recommender.recommend(
            filtered_df=books_df,
            author_preference='Isaac Asimov', 
            top_n=5
        )
        print(author_recs[['title', 'authors', 'categories', 'average_rating']])

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    test_recommendation_system()