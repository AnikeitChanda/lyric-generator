import pandas as pd

cleaned_lyrics_file_path = "cleaned_lyrics_new.csv"

def getValueInRangeWithString(str, length):
	while True:
		try:
			s = input(f"{str} (1-{length}): ")
			x = int(s)
			if x >= 1 and x <= length:
				return x
		except ValueError:
			pass

def getGenre(df):
	genres = list(df["genre"].unique())
	genres.sort()
	print("Genre")
	for index, genre in enumerate(genres):
		print(f"{index + 1}. {genre}")
# 	print(f"Select a genre (1-{len(genres)}): ", end=None)
	genreNumber = getValueInRangeWithString("Select a genre", len(genres)) - 1
	return genres[genreNumber]
	
def getArtist(df, selectedGenre):
	subDF = df[df["genre"] == selectedGenre]
	artists = list(subDF["artist"].unique())
	artists.sort()
	for index, artist in enumerate(artists):
		print(f"{index + 1}. {artist}")
	artistNumber = getValueInRangeWithString("Select an artist", len(artists)) - 1
	return artists[artistNumber]

def getSongAndLyrics(df, selectedGenre, selectedArtist):
	subDF = df[(df["genre"] == selectedGenre) & (df["artist"] == selectedArtist)]
	songs = list(subDF["title"].unique())
	songs.sort()
	for index, song in enumerate(songs):
		print(f"{index + 1}. {song}")
	songNumber = getValueInRangeWithString("Select a song", len(songs)) - 1
	song = songs[songNumber]
	lyrics = subDF[subDF["title"] == song]["lyrics"].iloc[0]
	return song, lyrics

def main():
	df = pd.read_csv(cleaned_lyrics_file_path)
	try:
		selectedGenre = getGenre(df)
		selectedArtist = getArtist(df, selectedGenre)
		selectedSong, selectedLyrics = getSongAndLyrics(df, selectedGenre, selectedArtist)
	except EOFError:
		return
	print(selectedSong)
	print(selectedLyrics)

if __name__ == "__main__":
	main()