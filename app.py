from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
import pickle
import pandas as pd
import traceback
import importlib.util

# Add the project directory to Python path
project_dir = r'F:\book recomendation web'
sys.path.append(project_dir)

# Dynamic import of BookRecommendationSystem
def import_class_from_file(module_name, class_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# Import BookRecommendationSystem
try:
    BookRecommendationSystem = import_class_from_file(
        'book_recommendation_system', 
        'BookRecommendationSystem', 
        os.path.join(project_dir, 'book_recommendation_system.py')
    )
except Exception as e:
    print(f"Error importing BookRecommendationSystem: {e}")
    raise

# Create Flask application
app = Flask(__name__, 
            template_folder=os.path.join(project_dir, 'templates'),
            static_folder=project_dir)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

class RecommendationService:
    def __init__(self, 
                 model_path=r'F:\book recomendation web\models\book_recommendation_model.pk3', 
                 data_path=r'F:\book recomendation web\data\data.csv'):
        try:
            print("Model Path:", model_path)
            print("Data Path:", data_path)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Load the pre-trained model using the new method
            self.recommender = BookRecommendationSystem.load_model(model_path)
            print("Recommendation system loaded successfully!")
            
            # Load the dataset
            self.books_df = pd.read_csv(data_path)
            print(f"Dataset loaded. Total books: {len(self.books_df)}")
        
        except Exception as e:
            print(f"Error initializing recommendation service: {e}")
            print(traceback.format_exc())
            raise
    
    def get_recommendations(self, genre=None, author_preference=None, top_n=5):
        try:
            # Get recommendations using the loaded recommender
            recommendations = self.recommender.recommend(
                filtered_df=self.books_df,
                genre=genre, 
                author_preference=author_preference, 
                top_n=top_n
            )
            
            # Format recommendations
            formatted_recommendations = [
                {
                    'title': row['title'],
                    'author': row['authors'],
                    'rating': row['average_rating'],
                    'year': row['published_year'],
                    'categories': row['categories']
                }
                for _, row in recommendations.iterrows()
            ]
            
            return formatted_recommendations
        
        except Exception as e:
            print(f"Recommendation error: {e}")
            print(traceback.format_exc())
            return []

# Global variable to store the recommendation service
recommender_service = None

def initialize_recommender_service():
    global recommender_service
    
    try:
        recommender_service = RecommendationService()
        print("Recommendation service initialized successfully!")
    except Exception as e:
        print(f"Error initializing recommendation service: {e}")
        print(traceback.format_exc())
        raise

@app.route('/')
def home():
    return render_template('brs.html')

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        print("Received Request Data:", data)
        
        genre = data.get('genre', '')
        author_preference = data.get('authorsPreference', '')
        
        if recommender_service is None:
            initialize_recommender_service()
        
        recommendations = recommender_service.get_recommendations(
            genre=genre, 
            author_preference=author_preference
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    
    except Exception as e:
        print(f"Recommendation error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def main():
    try:
        initialize_recommender_service()
        app.run(host='0.0.0.0', port=8080, debug=True)
    
    except Exception as e:
        print(f"Failed to start the application: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    main()