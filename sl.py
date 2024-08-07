
import csv
import lyricsgenius
import requests
import os


def get_geniusdataset(artist_name,access_token):
    base_url="https://api.genius.com"
    
    headers = {
        'Authorization': f'Bearer {access_token}'  # Ensure correct format
    }
    search_url=f"{base_url}/search"
    params = {'q':artist_name}
    
    response = requests.get(search_url,headers = headers,params=params)
    #print(f"Request URL: {response.url}")
    #print(f"Request Headers: {headers}")

    if response.status_code==200:
        return response.json()
    else:
        print(f"Error:{response.status_code}")
        return None
    
def get_genius_lyrics(song_title, artist_name ,access_token):
    genius = lyricsgenius.Genius(access_token)
    song = genius.search_song(song_title, artist_name)
    if song:
        return song.lyrics
    else:
        return "Lyrics not found"

def savetocsv(songs, filename):
    file_exists = os.path.isfile(filename)
    fieldnames = ['artist','title', 'lyrics']
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for song in songs:
            writer.writerow(song)

def main( artist_name, access_token):
     data = get_geniusdataset(artist_name,access_token)

     if data and 'response' in data:
         hits = data['response']['hits']
         songs = []
         for hit in hits:
             
             song_title = hit['result']['title']
             song_id = hit['result']['id']
             print(f"Song:{song_title}")
             lyrics = get_genius_lyrics(song_title, artist_name, access_token)
             songs.append({'artist':artist_name,'title': song_title, 'lyrics': lyrics})
         savetocsv(songs, 'output3.csv')

# Replace with your actual access token and song details
access_token = "WCmPAc4bFy5jOUtlzw5qj9mVWwGs9JbNLfenlJsRbaXyxvG8jfp9prEZCwsm-Ot5"
#filename = 'output2.csv'
#song_title = "Love Yourself"
artists = ['Taylor Swift', 'Justin Bieber', 'Adele', 'Drake', 'Beyoncé', 'Ed Sheeran', 'Rihanna', 'Bruno Mars']

for artist in artists:
    main(artist, access_token)
#main( artist_name, access_token)
