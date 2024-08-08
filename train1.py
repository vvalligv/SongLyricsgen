
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import ast
import pickle

# Upload your dataset
from google.colab import files
uploaded = files.upload()

# Load the dataset
df = pd.read_csv('corrected_data.csv')  # Update with the actual file name

# Convert padded_sequences to numerical format if necessary
df['padded_sequences'] = df['padded_sequences'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
sequence_lengths = df['padded_sequences'].apply(len)
print(sequence_lengths.unique())

X = np.array(df['padded_sequences'].tolist(), dtype=np.int32)
print(f"Shape of X: {X.shape}")

# Create labels by shifting sequences
y = np.roll(X, -1, axis=1)
y = np.array([seq.tolist() for seq in y], dtype=np.int32)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build GRU model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=256, input_length=X.shape[1]))
model.add(GRU(128, return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(10000, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
model.fit(X_train, np.expand_dims(y_train, -1), 
          epochs=10, 
          batch_size=32, 
          validation_data=(X_val, np.expand_dims(y_val, -1)), 
          callbacks=[early_stopping, checkpoint])

# Define and save tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['processed_lyrics'])
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
