import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re

class BookRecommendationSystem:
    def __init__(self, books_df):
        books_df = books_df.reset_index(drop=True)
        self.books_df = self._preprocess_data(books_df)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._create_feature_matrix()
    
    def _clean_author_name(self, author):
        author = str(author).lower().strip()
        author = re.sub(r'[^a-z\s]', '', author)
        author = re.sub(r'\s+', ' ', author)
        return author
    
    def _preprocess_data(self, df):
        df['title'] = df['title'].fillna('')
        df['authors'] = df['authors'].fillna('Unknown')
        df['categories'] = df['categories'].fillna('Uncategorized')
        df['normalized_authors'] = df['authors'].apply(self._clean_author_name)
        df['combined_features'] = (
            df['title'] + ' ' + 
            df['authors'] + ' ' + 
            df['categories']
        )
        numeric_columns = ['published_year', 'average_rating', 'num_pages', 'ratings_count']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
        return df
    
    def _create_feature_matrix(self):
        self.feature_matrix = self.vectorizer.fit_transform(self.books_df['combined_features'])
    
    def recommend(self, filtered_df=None, genre=None, author_preference=None, top_n=5):
        if filtered_df is None:
            filtered_df = self.books_df.copy()
        else:
            filtered_df = self._preprocess_data(filtered_df)
        
        filtered_df = filtered_df.reset_index(drop=True)
        recs_df = filtered_df.copy()
        
        if genre:
            recs_df = recs_df[recs_df['categories'].str.contains(genre, case=False, na=False)]
        
        if author_preference:
            norm_author_pref = self._clean_author_name(author_preference)
            author_mask = recs_df['normalized_authors'].str.contains(norm_author_pref, case=False, na=False)
            
            if author_mask.any():
                author_books = recs_df[author_mask].copy()
                other_books = recs_df[~author_mask].copy()
                author_books['similarity_score'] = 1.0
                
                if not other_books.empty:
                    author_query = self.vectorizer.transform([author_preference])
                    other_matrix = self.vectorizer.transform(other_books['combined_features'])
                    other_similarities = cosine_similarity(author_query, other_matrix)[0]
                    other_books['similarity_score'] = other_similarities
                    recs_df = pd.concat([author_books, other_books])
                else:
                    recs_df = author_books
            else:
                author_query = self.vectorizer.transform([author_preference])
                recs_df['similarity_score'] = cosine_similarity(
                    author_query, 
                    self.vectorizer.transform(recs_df['combined_features'])
                )[0]
        else:
            recs_df['similarity_score'] = recs_df['ratings_count'] / recs_df['ratings_count'].max()
        
        recs_df['weighted_rating'] = (
            0.5 * recs_df['average_rating'] + 
            0.3 * recs_df['similarity_score'] +
            0.2 * (recs_df['ratings_count'] / recs_df['ratings_count'].max())
        )
        
        recommendations = recs_df.sort_values('weighted_rating', ascending=False).head(top_n)
        return recommendations

    @classmethod
    def load_model(cls, model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

def create_recommendation_system(data_path):
    books_df = pd.read_csv(data_path)
    recommender = BookRecommendationSystem(books_df)
    return recommender

def save_recommendation_system(recommender, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(recommender, f)
    print(f"Recommendation system saved to {model_path}")

if __name__ == '__main__':
    data_path = r'F:\book recomendation web\data\data.csv'
    model_path = r'F:\book recomendation web\models\book_recommendation_model.pk3'
    
    recommender = create_recommendation_system(data_path)
    save_recommendation_system(recommender, model_path)
    
    print("\nSample Recommendations:")
    print("Science Fiction Recommendations:")
    sci_fi_recs = recommender.recommend(genre='Science fiction', top_n=5)
    print(sci_fi_recs[['title', 'authors', 'categories', 'average_rating', 'similarity_score', 'weighted_rating']])
    
    print("\nRecommendations with Author Preference:")
    author_recs = recommender.recommend(author_preference='Isaac Asimov', top_n=5)
    print(author_recs[['title', 'authors', 'categories', 'average_rating', 'similarity_score', 'weighted_rating']])