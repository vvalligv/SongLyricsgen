import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load the saved model checkpoint
model = load_model('model_checkpoint.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

def top_k_sampling(predictions, k=50):
    top_k_indices = np.argsort(predictions)[-k:]
    top_k_probabilities = predictions[top_k_indices]
    top_k_probabilities /= np.sum(top_k_probabilities)
    return np.random.choice(top_k_indices, p=top_k_probabilities)

def nucleus_sampling(predictions, p=0.9):
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_probabilities = predictions[sorted_indices]
    cumulative_probabilities = np.cumsum(sorted_probabilities)
    cutoff_index = np.where(cumulative_probabilities > p)[0][0]
    top_indices = sorted_indices[:cutoff_index + 1]
    top_probabilities = sorted_probabilities[:cutoff_index + 1]
    top_probabilities /= np.sum(top_probabilities)
    return np.random.choice(top_indices, p=top_probabilities)

def generate_song_lyrics(seed_text, model, tokenizer, max_length=100, num_words=150, temperature=1.0, k=50, p=0.9, sampling_method='top_k'):
    generated_text = seed_text

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')

        predicted_probabilities = model.predict(token_list, verbose=0)[0, -1, :]

        predicted_probabilities = np.log(predicted_probabilities + 1e-7) / temperature
        predicted_probabilities = np.exp(predicted_probabilities) / np.sum(np.exp(predicted_probabilities))

        if sampling_method == 'top_k':
            predicted_index = top_k_sampling(predicted_probabilities, k=k)
        elif sampling_method == 'nucleus':
            predicted_index = nucleus_sampling(predicted_probabilities, p=p)
        else:
            raise ValueError("Invalid sampling method")

        predicted_word = tokenizer.index_word.get(predicted_index, '<unknown>')

        if predicted_word == '<unknown>':
            continue

        generated_text += ' ' + predicted_word

    return generated_text

# Example usage
seed_text = input("Enter seed text: ")
num_words = 150  # Increase the number of words to generate
new_lyrics = generate_song_lyrics(seed_text, model, tokenizer, max_length=100, num_words=num_words, temperature=1.0, k=50, p=0.9, sampling_method='top_k')
print(new_lyrics)
