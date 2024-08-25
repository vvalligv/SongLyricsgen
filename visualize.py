import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('output3.csv')

print("Head: ",df.head())
print("Describe: ",df.describe())
print("Isnull: ",df.isnull().sum())


#distribution of  song lengths --> histogram

df['lyrics_length'] = df['lyrics'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10,6))
sns.histplot(df['lyrics_length'],bins = 50,kde = True) #Kde --> kernel density estimate is a flowy line which helps to visualiza
plt.title('Distribution of Song Lyrics Length')
plt.xlabel('No. of words')
plt.ylabel('Frequency')
plt.show()

