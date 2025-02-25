import pickle
import os

model_path = r'F:\book recomendation web\models\book_recommendation_model.pk3'
data_path = r'F:\book recomendation web\data\data.csv'

print("Model file exists:", os.path.exists(model_path))
print("Data file exists:", os.path.exists(data_path))

# Try to load the model
try:
    with open(model_path, 'rb') as f:
        recommender = pickle.load(f)
    print("Model loaded successfully")
    print("Model type:", type(recommender))
except Exception as e:
    print("Error loading model:", e)