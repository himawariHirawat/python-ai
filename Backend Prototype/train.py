import nltk
import numpy as np
import random
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from nltk.stem import WordNetLemmatizer
import pickle
import os

def delete_train_file(file_name):
    if os.path.exists("D:\\invergy\\Backend Prototype\\" + file_name):
        os.remove("D:\\invergy\\Backend Prototype\\" + file_name)
        print("Found a file of previous training. Successfully deleted this file")

delete_train_file("chatbot_model.h5")
delete_train_file("training_data.pkl")
# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents file
with open('D:\\invergy\\Backend Prototype\\database.json') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Extract data from intents
words = []
classes = []
documents = []
ignore_words = ['?', '!']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle and convert training data to array
random.shuffle(training)
training = np.array(training)

# Split data into training and testing sets
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build the model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=8, verbose=1)

# Save the model
model.save('chatbot_model.h5')

# Save words, classes, and training data
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open('training_data.pkl', 'wb'))