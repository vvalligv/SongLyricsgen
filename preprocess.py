import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('cleaned_song_lyrics.csv')

def clean_text(text):
    text = re.sub(r'<[^>]+>','',text)
    text = re.sub(r'[^a-zA\s]','',text)
    text = re.sub(r'\s+',' ',text).strip()
    return text

df['cleaned_lyrics'] = df['lyrics'].apply(clean_text)

df['tokenized_lyrics'] = df['cleaned_lyrics'].apply(word_tokenize)
df['tokenized_lyrics'] = df['tokenized_lyrics'].apply(lambda x:[word.lower() for word in x])


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

df['tokenized_lyrics'] = df['tokenized_lyrics'].apply(lambda x:[word for word in x if word not in stop_words])


df['tokenized_lyrics'] = df['tokenized_lyrics'].apply(lambda x:[lemmatizer.lemmatize(word) for word in x])
df['processed_lyrics'] = df['tokenized_lyrics'].apply(lambda x: ' '.join(x))

tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(df['processed_lyrics'].apply(lambda x: ' '.join(x)))
df['sequences'] = tokenizer.texts_to_sequences(df['processed_lyrics'])

max_length = 100
df['padded_sequences'] =list( pad_sequences(df['sequences'],maxlen = max_length,padding = 'post'))

df.to_csv('preprocessed.csv',index = False)
