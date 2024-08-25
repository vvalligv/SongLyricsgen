import pandas as pd

df = pd.read_csv('output3.csv')
df['lyrics_length'] = df['lyrics'].apply(lambda x: len(str(x).split()))

Q1 = df['lyrics_length'].quantile(0.25)
Q3 = df['lyrics_length'].quantile(0.75)
IQR = Q3-Q1

lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR

print(f'Lower Bound:{lower_bound}')
print(f'Upper Bound:{upper_bound}')

df_fil = df[(df['lyrics_length']>= lower_bound) &(df['lyrics_length']<=upper_bound)]#Filtering the DataFrame to Remove Outliers
df_fil = df_fil.drop(columns= ["lyrics_length"])#Drop the column of lyrics length which is used to calculate the outliers

print("Columns before saving:", df_fil.columns.tolist())



print(f'Original dataset shape: {df.shape}')
print(f'Filtered dataset shape: {df_fil.shape}')

# Save the cleaned and preprocessed dataset to a CSV file
df_fil.to_csv('cleaned_song_lyrics.csv', index=False)

df_check = pd.read_csv('cleaned_song_lyrics.csv')
print("Columns in saved file:", df_check.columns.tolist())
