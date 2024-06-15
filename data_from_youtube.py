import googleapiclient.discovery
import pandas as pd
# API information
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = 'AIzaSyAa90Sl4DTiizJdok2Wakx8sYugOO1o314'
# API client
youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)
# Minecraft query
minecraft_videos_ids = youtube.search().list(
    part="id",
    type='video',
    regionCode="US",
    order="relevance",
    q="minecraft",
    maxResults=50,
    fields="items(id(videoId))"
).execute()
# Valorant query
valorant_videos_ids = youtube.search().list(
        part="id",
        type='video',
        regionCode="US",
        order="relevance",
        q="valorant",
        maxResults=50,
        fields="items(id(videoId))"
).execute()

# Dictionary to store minecraft video data
minecraft_info = {
    'id':[],
    'duration':[],
    'views':[],
    'likes':[],
    'favorites':[],
    'comments':[]
}
# Dictionary to store valorant video data
valorant_info = {
    'id':[],
    'duration':[],
    'views':[],
    'likes':[],
    'favorites':[],
    'comments':[]
}
# For loop to obtain the information of each minecraft video
for item in minecraft_videos_ids['items']:
    # Getting the id
    vidId = item['id']['videoId']
    # Getting stats of the video
    r = youtube.videos().list(
        part="statistics,contentDetails",
        id=vidId,
        fields="items(statistics," + \
                     "contentDetails(duration))"
    ).execute()
    # We will only consider videos which contains all properties we need.
    # If a property is missing, then it will not appear as dictionary key,
    # this is why we need a try/catch block
    try:
        duration = r['items'][0]['contentDetails']['duration']
        views = r['items'][0]['statistics']['viewCount']
        likes = r['items'][0]['statistics']['likeCount']
        favorites = r['items'][0]['statistics']['favoriteCount']
        comments = r['items'][0]['statistics']['commentCount']
        minecraft_info['id'].append(vidId)
        minecraft_info['duration'].append(duration)
        minecraft_info['views'].append(views)
        minecraft_info['likes'].append(likes)
        minecraft_info['favorites'].append(favorites)
        minecraft_info['comments'].append(comments)
    except:
        pass
# For loop to obtain the information of each valorant video
for item in valorant_videos_ids['items']:
    vidId = item['id']['videoId']
    r = youtube.videos().list(
        part="statistics,contentDetails",
        id=vidId,
        fields="items(statistics," + \
                     "contentDetails(duration))"
    ).execute()
    try:
        duration = r['items'][0]['contentDetails']['duration']
        views = r['items'][0]['statistics']['viewCount']
        likes = r['items'][0]['statistics']['likeCount']
        favorites = r['items'][0]['statistics']['favoriteCount']
        comments = r['items'][0]['statistics']['commentCount']
        valorant_info['id'].append(vidId)
        valorant_info['duration'].append(duration)
        valorant_info['views'].append(views)
        valorant_info['likes'].append(likes)
        valorant_info['favorites'].append(favorites)
        valorant_info['comments'].append(comments)
    except:
        pass
pd.DataFrame(data=minecraft_info).to_csv("minecraft.csv", index=False)
pd.DataFrame(data=valorant_info).to_csv("valorant.csv", index=False)