import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import ast


# Load the preprocessed dataset
df = pd.read_csv('your_dataset.csv')

# Convert padded_sequences to numerical format if necessary
df['padded_sequences'] = df['padded_sequences'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
sequence_lengths = df['padded_sequences'].apply(len)
print(sequence_lengths.unique())

X = np.array(df['padded_sequences'].tolist(), dtype=np.int32)
print(f"Shape of X: {X.shape}")

# Create labels by shifting sequences
y = np.roll(X, -1, axis=1)

# Ensure y is in the correct format
y = np.array([seq.tolist() for seq in y], dtype=np.int32)
y = np.array(y, dtype=np.int32)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build GRU model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=256, input_length=X.shape[1]))  # Ensure input_length is correct
model.add(GRU(128, return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(10000, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, np.expand_dims(y_train, -1), epochs=50, batch_size=64, validation_data=(X_val, np.expand_dims(y_val, -1)), callbacks=[early_stopping])

# Define tokenizer and fit on texts
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['processed_lyrics'])

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def top_k_sampling(predictions, k=50):
    top_k_indices = np.argsort(predictions)[-k:]  # Get the indices of the top k probabilities
    top_k_probabilities = predictions[top_k_indices]  # Get the probabilities of the top k indices
    top_k_probabilities /= np.sum(top_k_probabilities)  # Normalize the probabilities
    return np.random.choice(top_k_indices, p=top_k_probabilities)  # Sample from the top k

def nucleus_sampling(predictions, p=0.9):
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_probabilities = predictions[sorted_indices]
    cumulative_probabilities = np.cumsum(sorted_probabilities)
    cutoff_index = np.where(cumulative_probabilities > p)[0][0]
    top_indices = sorted_indices[:cutoff_index + 1]
    top_probabilities = sorted_probabilities[:cutoff_index + 1]
    top_probabilities /= np.sum(top_probabilities)
    return np.random.choice(top_indices, p=top_probabilities)

def genlyr(seed_text, model, tokenizer, max_length=100, num_words=100, temperature=0.8, k=50, sampling_method='top_k'):
    generated_text = seed_text
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')
        
        predicted_probabilities = model.predict(token_list, verbose=0)[0, -1, :]
        
        # Adjust temperature
        predicted_probabilities = np.log(predicted_probabilities + 1e-7) / temperature
        predicted_probabilities = np.exp(predicted_probabilities) / np.sum(np.exp(predicted_probabilities))
        
        # Apply sampling
        if sampling_method == 'top_k':
            predicted_index = top_k_sampling(predicted_probabilities, k=k)
        elif sampling_method == 'nucleus':
            predicted_index = nucleus_sampling(predicted_probabilities, p=0.9)
        else:
            raise ValueError("Invalid sampling method")
        
        predicted_word = tokenizer.index_word.get(predicted_index, '<unknown>')
        
        if predicted_word == '<unknown>':
            continue  # Skip unknown words
        
        generated_text += ' ' + predicted_word
        seed_text += ' ' + predicted_word
    
    return generated_text

# Get seed text from user and generate new lyrics
seed_text = input("Enter text:")
new_lyrics = genlyr(seed_text, model, tokenizer, max_length=100, num_words=100, temperature=0.8, k=50, sampling_method='top_k')
print(new_lyrics)




