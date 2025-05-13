# libraries
import random
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to train the model
def train_model(file_path):
    # Initialize data
    words = []
    classes = []
    documents = []
    ignore_words = ["?", "!", ".", ",", "(", ")", "'s", "``", "''"]

    # Load the provided JSON file
    try:
        with open(file_path, 'r') as data_file:
            intents = json.load(data_file)
    except Exception as e:
        return f"Error reading the JSON file: {str(e)}"
    
    # Process patterns in the intents file
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            # Tokenize each word in the pattern
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Add to documents
            documents.append((word_list, intent["tag"]))
            # Add tag to classes if not already present
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    # Lemmatize words and remove stop words
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))  # Remove duplicates and sort the words
    classes = sorted(list(set(classes)))  # Sort classes

    # Display a summary of the processed data
    print(f"{len(documents)} documents")
    print(f"{len(classes)} classes: {classes}")
    print(f"{len(words)} unique lemmatized words: {words}")

    # Save words and classes to pickle files
    pickle.dump(words, open("words.pkl", "wb"))
    pickle.dump(classes, open("classes.pkl", "wb"))

    # Prepare training data
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        # Create bag of words for each document
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

        # Create the bag of words
        for word in words:
            bag.append(1 if word in pattern_words else 0)

        # Create output row (one-hot encoding for classes)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    # Shuffle and separate the training data into input (X) and output (Y)
    random.shuffle(training)
    train_x = [x[0] for x in training]
    train_y = [x[1] for x in training]

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    print("Training data created")

    # Define the neural network model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))
    model.summary()

    # Compile the model using SGD optimizer
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # Train the model
    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    # Save the trained model
    model.save("chatbot_model.h5")

    print("Model created and saved successfully")

    return "Model training completed and saved successfully."

# Example of how to call the train_model function
# result = train_model("/path/to/uploaded/intents.json")
# print(result)
