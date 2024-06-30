import googleapiclient.discovery
import pandas as pd

# API information
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = 'AIzaSyAa90Sl4DTiizJdok2Wakx8sYugOO1o314'

# API client
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

# List of games
games = [
    "minecraft shorts",
    "valorant shorts",
    "PUBG shorts",
    "GTA shorts",
    "Free Fire shorts",
    "Battlegrounds shorts",
    "Counter-Strike shorts",
    "League of Legends shorts",
    "Wuthering Waves shorts",
    "Brawl Stars shorts",
    "Street Fighter shorts",
    "Starcraft shorts",
    "Pokemon shorts",
    "Clash of Clans shorts",
    "Among Us shorts",
    "Genshin Impact shorts"
]

# Dictionary to store video data
game_info = {game: {
    'id': [],
    'duration': [],
    'views': [],
    'likes': [],
    'comments': [],
    'top_comment': []
} for game in games}

# Function to get the top comment for a video
def get_top_comment(video_id):
    try:
        comments_response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            order="relevance",
            maxResults=1,
            fields="items(snippet(topLevelComment(snippet(textDisplay))))"
        ).execute()
        top_comment = comments_response['items'][0]['snippet']['topLevelComment']['snippet']['textDisplay']
        return top_comment
    except:
        return "No comments"

# Function to fetch video data
def fetch_video_data(game, query):
    video_ids = youtube.search().list(
        part="id",
        type='video',
        regionCode="US",
        order="relevance",
        q=query,
        maxResults=2,
        fields="items(id(videoId))"
    ).execute()

    for item in video_ids['items']:
        vidId = item['id']['videoId']
        r = youtube.videos().list(
            part="statistics,contentDetails",
            id=vidId,
            fields="items(statistics,contentDetails(duration))"
        ).execute()

        try:
            duration = r['items'][0]['contentDetails']['duration']
            views = r['items'][0]['statistics']['viewCount']
            likes = r['items'][0]['statistics']['likeCount']
            comments = r['items'][0]['statistics']['commentCount']
            top_comment = get_top_comment(vidId)

            game_info[game]['id'].append(vidId)
            game_info[game]['duration'].append(duration)
            game_info[game]['views'].append(views)
            game_info[game]['likes'].append(likes)
            game_info[game]['comments'].append(comments)
            game_info[game]['top_comment'].append(top_comment)
        except:
            pass

# Fetch data for each game
for game in games:
    fetch_video_data(game, game)

# Save data to CSV files
for game in games:
    pd.DataFrame(data=game_info[game]).to_csv(f"{game.replace(' ', '_').lower()}.csv", index=False)
