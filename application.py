from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

application = Flask(__name__)

@application.route("/")
def index():
    return "Welcome to the Sentiment Analysis API! Send a POST request to /predict with a JSON body containing the text to analyze."

loaded_model = None
with open("basic_classifier.pkl", "rb") as f1:
    loaded_model = pickle.load(f1)

vectorizer = None
with open("count_vectorizer.pkl", "rb") as f2:
    vectorizer = pickle.load(f2)
    
@application.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Extract the text from the data
        text = data["text"]

        # Transform the input text using the vectorizer
        transformed_text = vectorizer.transform([text])

        # Use the model to make a prediction
        prediction = loaded_model.predict(transformed_text)

        # Return the prediction as JSON
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    application.run(port=5000, debug=True)