from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
import pickle
import pandas as pd
import traceback

# Get the absolute path of the current project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the project directory to Python path
sys.path.append(BASE_DIR)

# Import BookRecommendationSystem directly
from book_recommendation_system import BookRecommendationSystem

# Create Flask application
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=BASE_DIR)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

class RecommendationService:
    def __init__(self, 
                 model_path=os.path.join(BASE_DIR, 'models', 'book_recommendation_model.pk3'), 
                 data_path=os.path.join(BASE_DIR, 'data', 'data.csv')):
        try:
            print("Model Path:", model_path)
            print("Data Path:", data_path)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            try:
                # Try to load the pre-trained model
                with open(model_path, 'rb') as f:
                    self.recommender = pickle.load(f)
                print("Recommendation system loaded successfully!")
            except Exception as load_error:
                print(f"Error loading model: {load_error}")
                print("Creating new BookRecommendationSystem...")
                # If loading fails, create a new model
                books_df = pd.read_csv(data_path)
                self.recommender = BookRecommendationSystem(books_df)
                # Save the new model
                with open(model_path, 'wb') as f:
                    pickle.dump(self.recommender, f)
                print("New recommendation system created and saved!")
            
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