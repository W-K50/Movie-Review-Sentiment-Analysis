from flask import Flask, request, jsonify, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

# Download stopwords (first-time use)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the saved model and vectorizer
model = joblib.load(r"C:\Users\hp\Desktop\Movie Review Sentiment Analysis\sentiment_model.pkl")
vectorizer = joblib.load(r"C:\Users\hp\Desktop\Movie Review Sentiment Analysis\tfidf_vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    words = text.split()  # Tokenization
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data.get("review", "")

    if not review:
        return jsonify({"error": "No review text provided!"}), 400

    # Preprocess and predict sentiment
    processed_review = preprocess_text(review)
    review_vector = vectorizer.transform([processed_review])
    
    # Get prediction and probability estimates
    prediction = model.predict(review_vector)[0]
    probabilities = model.predict_proba(review_vector)[0]
    
    # Confidence score (max probability)
    confidence = np.max(probabilities)

    sentiment = "Positive" if prediction == 1 else "Negative"

    return jsonify({"review": review, "sentiment": sentiment, "confidence": float(confidence)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
