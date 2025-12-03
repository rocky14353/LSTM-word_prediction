import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

st.title("LSTM Word Prediction")

# load the model
model = load_model('lstm_text_gen_model.h5') 

user_input = st.text_area("Enter a partial sentence:")

#unload my picke file for Tokenizer

with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

# Reverse word index for fast lookup
reverse_word_index = {i: w for w, i in tokenizer.word_index.items()}

max_input_len = 13  # assuming this was the max length used during training

encoded_input =tokenizer.texts_to_sequences([user_input])[0]

# pad tthe sequence

if len(encoded_input)>=max_input_len:
    encoded_input = encoded_input[-max_input_len:]
else:
    encoded_input = pad_sequences([encoded_input],maxlen=max_input_len,padding='pre')

if st.button("Predict Next Word"):

    if not user_input.strip():
        st.error("Please enter text before predicting.")
        st.stop()
    
    predicted_probs = model.predict(encoded_input, verbose=0)
    predicted_index = np.argmax(predicted_probs,axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            predicted_word = word
            break   
    st.write(f"Predicted Next Word: {predicted_word}")
    st.write(f"{tf.__version__}")
else:
    st.write("Please enter a partial sentence and click 'Predict Next Word' to see the result.")