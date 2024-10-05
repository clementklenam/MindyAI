import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import random
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import os
from datetime import datetime

nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Set page config
st.set_page_config(page_title=" AI Mental Health Assistance", page_icon="🧠", layout="wide")



# Ensure necessary NLTK data is downloaded at runtime
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')  # Tokenizer models
        nltk.download('wordnet')  # WordNet lemmatizer
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        st.stop()

# Load or initialize chat history
def load_chat_history():
    if os.path.exists('chat_history.json'):
        with open('chat_history.json', 'r') as file:
            return json.load(file)
    return []

def save_chat_history(messages):
    with open('chat_history.json', 'w') as file:
        json.dump(messages, file)

# Load or initialize mood data
def load_mood_data():
    if os.path.exists('mood_data.json'):
        with open('mood_data.json', 'r') as file:
            return json.load(file)
    return []

def save_mood_data(mood_data):
    with open('mood_data.json', 'w') as file:
        json.dump(mood_data, file)

# Download NLTK data
download_nltk_data()

# Load the intents data
@st.cache_resource
def load_intents(file_path):
    with open(file_path) as file:
        return json.load(file)

data = load_intents('intents.json')

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',']

for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_chars]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
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

    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=0)
    return model

# Create and train the model
model = create_and_train_model(train_x, train_y)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
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
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "I'm not sure I understand. Can you please rephrase?"

# Custom CSS for improved UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1500px;
        margin: 0 auto;
    }
    .stChatInput {
        border-radius: 15px;
    }
    .stChatMessage {
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .bot-message {
        background-color: #f0f0f0;
    }
    .st-emotion-cache-1v0mbdj > img {
        border-radius: 50%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.title("🧠 Mental Health Chatbot")
st.markdown("Welcome to your personal mental health assistant. Feel free to share your thoughts and concerns.")

# Sidebar for options and mood tracking
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

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = load_chat_history()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("💬 What's on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Get chatbot response
    ints = predict_class(prompt)
    response = get_response(ints, data)

    # Display chatbot response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Save chat history
    save_chat_history(st.session_state.messages)

# Display a helpful message if the chat is empty
if not st.session_state.messages:
    st.info("👋 Hello! How are you feeling today? Feel free to share your thoughts or concerns.")

# Footer
st.markdown("---")
st.markdown("Remember, I'm here to listen and offer support, but for professional help, please consult a licensed mental health professional.")