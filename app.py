# libraries
import random
import numpy as np
import pickle
import json
import os
import shutil
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from train import train_model  # Custom training module

# Downloading necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

# Initialize Flask app
app = Flask(__name__)

# Configure file upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model and data
model = load_model("chatbot_model.h5")
print("✅ Model loaded successfully.")
model.summary()

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Chat endpoint
@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    if msg.lower().startswith('my name is'):
        name = msg[11:].strip()
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.lower().startswith('hi my name is'):
        name = msg[14:].strip()
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    return res

# Tokenize and clean sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence into bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("✅ Found in bag:", w)
    print("🧠 BOW Vector:", bag)
    return np.array(bag)

# Predict intent
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=True)
    res = model.predict(np.array([p]))[0]
    print("🔍 Prediction Probabilities:", res)

    ERROR_THRESHOLD = 0.05  # Lowered threshold for better matching
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print("🤖 Predicted:", return_list)
    return return_list

# Updated getResponse function to handle unknown queries
def getResponse(ints, intents_json):
    if not ints:
        return "I'm sorry, I don't understand that. Can you please rephrase your question?"

    tag = ints[0]["intent"]
    
    # Check if the tag exists in intents_json
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])

    # Default response if no tag is found
    return "I’m not sure how to respond to that. Could you ask something else?"

# Upload route for training
@app.route("/upload", methods=["POST"])
def upload_file():
    global model, intents, words, classes  # Ensure global variables are updated

    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Copy uploaded file to intents.json for consistency
            shutil.copy(filepath, "intents.json")

            # Run training
            response = train_model(filepath)

            # Reload trained artifacts
            model = load_model("chatbot_model.h5")
            print("✅ Model reloaded after training.")
            with open("intents.json", "r", encoding="utf-8") as f:
                intents = json.load(f)
            words = pickle.load(open("words.pkl", "rb"))
            classes = pickle.load(open("classes.pkl", "rb"))

            return jsonify({"message": response, "status": "success"}), 200
        except Exception as e:
            return jsonify({"message": f"Error during training: {str(e)}", "status": "error"}), 500
    else:
        return jsonify({"message": "Invalid file type. Only JSON files allowed."}), 400

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
