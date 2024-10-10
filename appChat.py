import streamlit as st
import json
import random
import os
import re
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import asyncio
import requests
from streamlit_lottie import st_lottie

# Optimize loading Lottie animation
@st.cache_data(show_spinner=False)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource(show_spinner=False)
def get_lottie_animation():
    lottie_url = "https://lottie.host/c850a9c6-a7b6-49bd-af86-ae5e07564b46/9qJ1pZd7qn.json"
    return load_lottieurl(lottie_url)

def display_lottie_animation():
    lottie_json = get_lottie_animation()
    if lottie_json:
        st_lottie(lottie_json, speed=1, width=300, height=300, key="loading_animation")
    else:
        st.warning("Lottie animation could not be loaded.")

# Tokenization and lemmatization
@st.cache_data(show_spinner=False)
def simple_tokenize(text):
    return re.findall(r'\w+', text.lower())

@st.cache_data(show_spinner=False)
def simple_lemmatize(word):
    return word[:-1] if word.endswith('s') else word

# Load or initialize chat history
@st.cache_data(show_spinner=False)
def load_chat_history():
    file_path = 'chat_history.json'
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            st.warning("Chat history file is corrupted. Starting with an empty history.")
    return []

@st.cache_data(show_spinner=False)
def save_chat_history(messages):
    file_path = 'chat_history.json'
    try:
        with open(file_path, 'w') as file:
            json.dump(messages, file)
    except IOError:
        st.warning("Unable to save chat history.")

# Load intents data
@st.cache_data(show_spinner=False)
def load_intents(file_path):
    try:
        with open(file_path) as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading intents file: {e}")
        return {"intents": []}

@st.cache_resource(show_spinner=False)
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

# Processing input and predicting response
@st.cache_data(show_spinner=False)
def clean_up_sentence(sentence):
    sentence_words = simple_tokenize(sentence)
    sentence_words = [simple_lemmatize(word) for word in sentence_words]
    return sentence_words

@st.cache_data(show_spinner=False)
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

@st.cache_data(show_spinner=False)
def predict_class(sentence, model, words, classes):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

@st.cache_data(show_spinner=False)
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I'm not sure I understand. Can you please rephrase?"

# Async chatbot response handling
async def handle_message_async(prompt, model, words, classes, data):
    intents = predict_class(prompt, model, words, classes)
    response = get_response(intents, data)
    return response

@st.cache_resource(show_spinner=False)
def initialize_model_and_data():
    data = load_intents('intents.json')
    words, classes, documents = [], [], []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            word_list = simple_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(list(set([simple_lemmatize(word) for word in words])))
    classes = sorted(list(set(classes)))

    training = []
    output_empty = [0] * len(classes)
    for document in documents:
        bag = [1 if word in document[0] else 0 for word in words]
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x, train_y = list(training[:, 0]), list(training[:, 1])

    model = create_and_train_model(train_x, train_y)
    return model, words, classes, data

# Main async function
async def main():
    st.set_page_config(page_title="AI Mental Health Assistance", page_icon="🧠", layout="wide")

    model, words, classes, data = initialize_model_and_data()

    st.title("🧠 AI Mental Health Assistance")
    st.markdown("Welcome to your personal mental health assistant. Feel free to share your thoughts and concerns.")

    # Sidebar options
    with st.sidebar:
        st.header("Options")
        if st.button("Start a New Chat"):
            st.session_state.messages = []
            save_chat_history(st.session_state.messages)
            st.rerun()

    # Handle chat session state and input
    if 'messages' not in st.session_state:
        st.session_state.messages = load_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("💬 What's on your mind?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display placeholder message and animation while processing
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with placeholder.container():
                display_lottie_animation()
                st.text("Thinking...")

            # Async processing and response
            response = await handle_message_async(prompt, model, words, classes, data)
            placeholder.empty()
            placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        save_chat_history(st.session_state.messages)

    # Hide default Streamlit elements
    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Disclaimer**: This chatbot is not a substitute for professional mental health care.")

if __name__ == "__main__":
    asyncio.run(main())
