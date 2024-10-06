import streamlit as st
import json
import random
import os
import re
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="AI Mental Health Assistance", page_icon="🧠", layout="wide")

# Simple tokenization function
def simple_tokenize(text):
    return re.findall(r'\w+', text.lower())

# Simple lemmatization function (just removes 's' from the end of words)
def simple_lemmatize(word):
    return word[:-1] if word.endswith('s') else word

# Load or initialize chat history
def load_chat_history():
    file_path = 'chat_history.json'
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            st.warning("Chat history file is corrupted. Starting with an empty history.")
    return []

def save_chat_history(messages):
    file_path = 'chat_history.json'
    try:
        with open(file_path, 'w') as file:
            json.dump(messages, file)
    except IOError:
        st.warning("Unable to save chat history.")

# Load the intents data
@st.cache_resource
def load_intents(file_path):
    try:
        with open(file_path) as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading intents file: {e}")
        return {"intents": []}  # Return empty intents instead of stopping the app

data = load_intents('intents.json')

words = []
classes = []
documents = []

# Process intents data
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = simple_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean words
words = [simple_lemmatize(word) for word in words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [simple_lemmatize(word) for word in word_patterns]
    
    # Create bag of words
    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Function to create and train the model
@st.cache_resource
def create_and_train_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Train the model
    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=0)
    return model

# Create and train the model
model = create_and_train_model(train_x, train_y)

# Utility functions to process input and predict response
def clean_up_sentence(sentence):
    sentence_words = simple_tokenize(sentence)
    sentence_words = [simple_lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I'm not sure I understand. Can you please rephrase?"

# Function to load mood data
def load_mood_data():
    file_path = 'mood_data.json'
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            st.warning("Mood data file is corrupted. Starting with empty data.")
    return []

# Function to save mood data
def save_mood_data(mood_data):
    file_path = 'mood_data.json'
    try:
        with open(file_path, 'w') as file:
            json.dump(mood_data, file)
    except IOError:
        st.warning("Unable to save mood data.")

# Chatbot UI elements
st.title("🧠 AI Mental Health Assistance")
st.markdown("Welcome to your personal mental health assistant. Feel free to share your thoughts and concerns.")

# Sidebar with options and mood tracking
with st.sidebar:
    st.header("Options")
    if st.button("Start a New Chat"):
        st.session_state.messages = []
        save_chat_history(st.session_state.messages)
        st.rerun()

    st.header("Mood Tracker")
    mood = st.slider("How are you feeling today?", 1, 5, 3)
    mood_submitted = st.button("Submit Mood")

    if mood_submitted:
        mood_data = load_mood_data()
        mood_data.append({"date": datetime.now().isoformat(), "mood": mood})
        save_mood_data(mood_data)
        st.success("Mood recorded successfully!")

    if st.button("View Mood History"):
        mood_data = load_mood_data()
        if mood_data:
            dates = [datetime.fromisoformat(entry['date']) for entry in mood_data]
            moods = [entry['mood'] for entry in mood_data]
            st.line_chart({"Mood": moods}, use_container_width=True)
            st.write("Mood history (1: Very Low, 5: Very High)")
        else:
            st.info("No mood data available yet.")

# Chat session state and handling input
if 'messages' not in st.session_state:
    st.session_state.messages = load_chat_history()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("💬 What's on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    intents = predict_class(prompt)
    response = get_response(intents, data)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    save_chat_history(st.session_state.messages)

# Footer note
st.markdown("---")
st.markdown("Remember, this chatbot is here to listen and offer support. For professional help, please consult a licensed mental health professional.")
