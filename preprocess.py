import pandas as pd # Used for data manipulation and analysis.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences #Tools from Keras for tokenizing text and padding sequences,
#which are common steps in preparing text data for deep learning models.
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk #: Natural Language Toolkit, used for various text processing tasks like tokenization, stop words removal, and lemmatization.
import re  # Python's regular expression module, used for text cleaning.

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') #These include the tokenizer (punkt), a list of stopwords (stopwords), and the WordNet lexical database for lemmatization (wordnet).

df = pd.read_csv('cleaned_song_lyrics.csv')

def clean_text(text):
    text = re.sub(r'<[^>]+>','',text) #Removes any HTML tags from the lyrics (e.g., <div>Hello</div> becomes Hello).
    text = re.sub(r'[^a-zA\s]','',text) #Removes all characters that are not lowercase alphabets or spaces (e.g., Hello!123 becomes Hello).
    text = re.sub(r'\s+',' ',text).strip() #Removes any extra spaces and leading/trailing spaces (e.g., Hello world becomes Hello world).
    return text

df['cleaned_lyrics'] = df['lyrics'].apply(clean_text)

df['tokenized_lyrics'] = df['cleaned_lyrics'].apply(word_tokenize) #Splits the cleaned lyrics into individual words (tokens). 
#For example, "Hello world" becomes ["Hello", "world"].
df['tokenized_lyrics'] = df['tokenized_lyrics'].apply(lambda x:[word.lower() for word in x]) # Converts all tokens 
#to lowercase (e.g., ["Hello", "World"] becomes ["hello", "world"]).


stop_words = set(stopwords.words('english')) #A set of common English words (like "the," "is," "in") that are typically removed from text
#data because they don't carry much meaning.
lemmatizer = WordNetLemmatizer() #A tool that reduces words to their base or root form (e.g., "running" becomes "run").

df['tokenized_lyrics'] = df['tokenized_lyrics'].apply(lambda x:[word for word in x if word not in stop_words]) # Filters out the stop words from the tokenized lyrics.
#For example, ["this", "is", "a", "song"] might become ["song"].


df['tokenized_lyrics'] = df['tokenized_lyrics'].apply(lambda x:[lemmatizer.lemmatize(word) for word in x]) #Lemmatizes each word in the tokenized lyrics 
(e.g., ["running", "jumps"] becomes ["run", "jump"]).
df['processed_lyrics'] = df['tokenized_lyrics'].apply(lambda x: ' '.join(x)) #Joins the tokens back into a single string (e.g., ["run", "jump"]
#becomes "run jump"). The result is stored in a new column processed_lyrics.


tokenizer = Tokenizer(num_words = 10000)#Initializes a Keras tokenizer that will consider the top 10,000 most frequent words.
tokenizer.fit_on_texts(df['processed_lyrics'].apply(lambda x: ' '.join(x))) #Fits the tokenizer on the processed lyrics, building the word index based on word frequency.

df['sequences'] = tokenizer.texts_to_sequences(df['processed_lyrics']) #Converts each lyric into a sequence of integers, where each integer represents a word's index in the tokenizer's vocabulary. 
#For example, "run jump" might become [2, 5] depending on the word indices.

max_length = 100
df['padded_sequences'] =list( pad_sequences(df['sequences'],maxlen = max_length,padding = 'post'))#The pad_sequences function is used to ensure that all sequences in the dataset have the same length. 
#This is important because most machine learning models expect input data of uniform size.

#df['sequences']: This is the list of sequences, where each sequence is a list of integers representing words in the lyrics.
#maxlen=max_length: Specifies the maximum length of the sequences. If a sequence is shorter than this length, it will be padded with zeros; if it's longer, it will be truncated.
#padding='post': This specifies that padding should be added at the end (post-padding) of the sequences.

#Let's consider a simple example to see how this works:

#Assume you have three song lyrics:

#"Hello world"
#"This is a test"
#"Keras tokenizer example"
#After tokenization (assuming these are the token indices):

#"Hello world" becomes [5, 9]
#"This is a test" becomes [1, 7, 3, 4]
#"Keras tokenizer example" becomes [10, 8, 2]
#Original Sequences:

#sequence_1 = [5, 9]
#sequence_2 = [1, 7, 3, 4]
#sequence_3 = [10, 8, 2]
#After Padding with max_length = 5:


#padded_sequence_1 = [5, 9, 0, 0, 0]
#padded_sequence_2 = [1, 7, 3, 4, 0]
#padded_sequence_3 = [10, 8, 2, 0, 0]

df.to_csv('preprocessed.csv',index = False)
