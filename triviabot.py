import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

# Setup Sentry
sentry_sdk.init(
    dsn="https://c8e1b1f7f540166fdc01aed67501a156@o4507935419400192.ingest.us.sentry.io/4507935424839680",  # Replace with your DSN from Sentry
    integrations=[LoggingIntegration(level=None, event_level='ERROR')]
)

import requests
import json
import random
import importlib
import traceback
import unicodedata
import datetime
import time
import pytz
import os
from pymongo import MongoClient
import difflib
import string
from urllib.parse import urlparse 
from urllib.parse import quote
import io            
from PIL import Image, ImageDraw, ImageFont 
import openai
import main
import subprocess
import re
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import logging
import praw
import tempfile
import base64
from collections import Counter
import math

# Define the base API URL for Matrix
matrix_base_url = "https://matrix.redditspace.com/_matrix/client/v3"
upload_url = "https://matrix.redditspace.com/_matrix/media/v3/upload"
sync_url = f"{matrix_base_url}/sync"

# Define global variables to store streaks and scores
round_count = 0
scoreboard = {}

# Define a global variable to store round data
round_data = {
    "questions": []  # Collect questions asked with their answers and responses
}

fastest_answers_count = {}
current_longest_answer_streak = {"user": None, "streak": 0}
current_longest_round_streak = {"user": None, "streak": 0}

# Store since_token and responses
since_token = None
responses = []
question_start_time = None  # This will store the time the question is asked

# Set up headers, including the authorization token
headers = []
headers_media = [] 

params = []
filter_json = []

# Initialize tokens
token_v2 = ""
bearer_token = ""

# Add this global variable to hold the submission queue
submission_queue = []
max_queue_size = 100  # Number of submissions to accumulate before flushing

# Initialize all variables
username = os.getenv("username")
password = os.getenv("password")
mongo_db_string = os.getenv("mongo_db_string")
openai.api_key = os.getenv("open_api_key")  # Store your API key securely
buymeacoffee_api_key = os.getenv("buy_me_a_coffee_api_key")
reddit_client_id = os.getenv("reddit_client_id")
reddit_secret_id = os.getenv("reddit_secret_id")
openweather_api_key = os.getenv("openweather_api_key")
googlemaps_api_key = os.getenv("googlemaps_api_key")
target_room_id = os.getenv("target_room_id")
question_time = int(os.getenv("question_time"))
questions_per_round = int(os.getenv("questions_per_round"))
time_between_rounds = int(os.getenv("time_between_rounds"))
time_between_questions = int(os.getenv("time_between_questions"))
time_between_questions_default = time_between_questions
max_retries = int(os.getenv("max_retries"))
delay_between_retries = int(os.getenv("delay_between_retries"))
id_limits = {"general": 2000, "mysterybox": 2000, "crossword": 15000, "jeopardy": 100000, "wof": 1500}
first_place_bonus = 0
magic_time = 10
magic_number = 0000

ghost_mode_default = False
ghost_mode = ghost_mode_default
god_mode_default = False
god_mode = god_mode_default
god_mode_points = 5000
yolo_mode_default = False
yolo_mode = yolo_mode_default
emoji_mode_default = True
emoji_mode = emoji_mode_default
magic_number_correct = False
wf_winner = False
nice_okra = False
creep_okra = False

image_questions_default = True
image_questions = image_questions_default


question_categories = [
    "Mystery Box or Boat", "Famous People", "Anatomy", "Characters", "Music", "Art & Literature", 
    "Chemistry", "Geography", "Mathematics", "Physics", "Science & Nature", "Language", "English Grammar", 
    "Astronomy", "Logos", "The World", "Economics & Government", "Toys & Games", "Food & Drinks", "Geology", 
    "Tech & Video Games", "Flags", "Miscellaneous", "Biology", "Superheroes", "Television", "Pop Culture", 
    "History", "Movies", "Religion & Mythology", "Sports & Leisure", "World Culture", "General Knowledge", "Statistics"
]

fixed_letters = ['O', 'K', 'R', 'A']

categories_to_exclude = []  



# Configure your Reddit client
reddit = praw.Reddit(
    client_id = reddit_client_id,
    client_secret = reddit_secret_id,
    user_agent="TriviaBot/1.0"
)


# List of world capitals and major cities
cities = [
{"city": "Accra", "country": "Ghana", "lat": 5.6037, "lon": -0.1870, "capital": True},
{"city": "Addis Ababa", "country": "Ethiopia", "lat": 9.0300, "lon": 38.7400, "capital": True},
{"city": "Algiers", "country": "Algeria", "lat": 36.7525, "lon": 3.0420, "capital": True},
{"city": "Athens", "country": "Greece", "lat": 37.9838, "lon": 23.7275, "capital": True},
{"city": "Baghdad", "country": "Iraq", "lat": 33.3152, "lon": 44.3661, "capital": True},
{"city": "Baku", "country": "Azerbaijan", "lat": 40.4093, "lon": 49.8671, "capital": True},
{"city": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074, "capital": True},
{"city": "Belmopan", "country": "Belize", "lat": 17.2510, "lon": -88.7590, "capital": True},
{"city": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050, "capital": True},
{"city": "Bogotá", "country": "Colombia", "lat": 4.7110, "lon": -74.0721, "capital": True},
{"city": "Brasília", "country": "Brazil", "lat": -15.8267, "lon": -47.9218, "capital": True},
{"city": "Brazzaville", "country": "Congo", "lat": 4.2634, "lon": 15.2429, "capital": True},
{"city": "Brussels", "country": "Belgium", "lat": 50.8503, "lon": 4.3517, "capital": True},
{"city": "Budapest", "country": "Hungary", "lat": 47.4979, "lon": 19.0402, "capital": True},
{"city": "Buenos Aires", "country": "Argentina", "lat": -34.6037, "lon": -58.3816, "capital": True},
{"city": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357, "capital": True},
{"city": "Canberra", "country": "Australia", "lat": -35.2820, "lon": 149.1287, "capital": True},
{"city": "Copenhagen", "country": "Denmark", "lat": 55.6761, "lon": 12.5683, "capital": True},
{"city": "Dhaka", "country": "Bangladesh", "lat": 23.8103, "lon": 90.4125, "capital": True},
{"city": "Djibouti", "country": "Djibouti", "lat": 11.8251, "lon": 42.5903, "capital": True},
{"city": "Dublin", "country": "Ireland", "lat": 53.3498, "lon": -6.2603, "capital": True},
{"city": "Gaborone", "country": "Botswana", "lat": -24.6282, "lon": 25.9231, "capital": True},
{"city": "Guatemala City", "country": "Guatemala", "lat": 14.6349, "lon": -90.5069, "capital": True},
{"city": "Havana", "country": "Cuba", "lat": 23.1136, "lon": -82.3666, "capital": True},
{"city": "Helsinki", "country": "Finland", "lat": 60.1695, "lon": 24.9354, "capital": True},
{"city": "Jakarta", "country": "Indonesia", "lat": -6.2088, "lon": 106.8456, "capital": True},
{"city": "Jerusalem", "country": "Israel", "lat": 31.7683, "lon": 35.2137, "capital": True},
{"city": "Kabul", "country": "Afghanistan", "lat": 34.5289, "lon": 69.1725, "capital": True},
{"city": "Kingston", "country": "Jamaica", "lat": 17.9712, "lon": -76.7936, "capital": True},
{"city": "Kinshasa", "country": "Democratic Republic of the Congo", "lat": -4.4419, "lon": 15.2663, "capital": True},
{"city": "Manama", "country": "Bahrain", "lat": 26.2285, "lon": 50.5860, "capital": True},
{"city": "Minsk", "country": "Belarus", "lat": 53.9045, "lon": 27.5615, "capital": True},
{"city": "NDjamena", "country": "Chad", "lat": 12.1348, "lon": 15.0557, "capital": True},
{"city": "Nassau", "country": "Bahamas", "lat": 25.0343, "lon": -77.3963, "capital": True},
{"city": "New Delhi", "country": "India", "lat": 28.6139, "lon": 77.2090, "capital": True},
{"city": "Nicosia", "country": "Cyprus", "lat": 35.1856, "lon": 33.3823, "capital": True},
{"city": "Ottawa", "country": "Canada", "lat": 45.4215, "lon": -75.6972, "capital": True},
{"city": "Ouagadougou", "country": "Burkina Faso", "lat": 12.3714, "lon": -1.5197, "capital": True},
{"city": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522, "capital": True},
{"city": "Phnom Penh", "country": "Cambodia", "lat": 11.5564, "lon": 104.9282, "capital": True},
{"city": "Port au Prince", "country": "Haiti", "lat": 18.5944, "lon": -72.3074, "capital": True},
{"city": "Prague", "country": "Czech Republic", "lat": 50.0755, "lon": 14.4378, "capital": True},
{"city": "Quito", "country": "Ecuador", "lat": -0.1807, "lon": -78.4678, "capital": True},
{"city": "Reykjavik", "country": "Iceland", "lat": 64.1355, "lon": -21.8954, "capital": True},
{"city": "Rome", "country": "Italy", "lat": 41.9028, "lon": 12.4964, "capital": True},
{"city": "San José", "country": "Costa Rica", "lat": 9.9281, "lon": -84.0907, "capital": True},
{"city": "San Salvador", "country": "El Salvador", "lat": 13.6929, "lon": -89.2182, "capital": True},
{"city": "Santiago", "country": "Chile", "lat": -33.4489, "lon": -70.6693, "capital": True},
{"city": "Santo Domingo", "country": "Dominican Republic", "lat": 18.4861, "lon": -69.9312, "capital": True},
{"city": "Sarajevo", "country": "Bosnia and Herzegovina", "lat": 43.8563, "lon": 18.4131, "capital": True},
{"city": "Sofia", "country": "Bulgaria", "lat": 42.6977, "lon": 23.3219, "capital": True},
{"city": "Sucre", "country": "Bolivia", "lat": -19.0196, "lon": -65.2619, "capital": True},
{"city": "Suva", "country": "Fiji", "lat": -18.1248, "lon": 178.4501, "capital": True},
{"city": "Tallinn", "country": "Estonia", "lat": 59.4370, "lon": 24.7535, "capital": True},
{"city": "Tbilisi", "country": "Georgia", "lat": 41.7151, "lon": 44.8271, "capital": True},
{"city": "Tegucigalpa", "country": "Honduras", "lat": 14.0723, "lon": -87.1921, "capital": True},
{"city": "Tehran", "country": "Iran", "lat": 35.6892, "lon": 51.3890, "capital": True},
{"city": "Thimphu", "country": "Bhutan", "lat": 27.4728, "lon": 89.6390, "capital": True},
{"city": "Tirana", "country": "Albania", "lat": 41.3275, "lon": 19.8189, "capital": True},
{"city": "Vienna", "country": "Austria", "lat": 48.2082, "lon": 16.3738, "capital": True},
{"city": "Yaoundé", "country": "Cameroon", "lat": 3.8480, "lon": 11.5021, "capital": True},
{"city": "Yerevan", "country": "Armenia", "lat": 40.1792, "lon": 44.4991, "capital": True},
{"city": "Zagreb", "country": "Croatia", "lat": 45.8150, "lon": 15.9819, "capital": True}
]



def get_google_maps(lat, lon):
    
    base_street_view_url = "https://maps.googleapis.com/maps/api/streetview"
    base_static_map_url = "https://maps.googleapis.com/maps/api/staticmap"
    base_maps_url = "https://www.google.com/maps"
    
    # 1. Get nearest street view image
    street_view_params = {
        "size": "600x400",  # Image size
        "location": f"{lat},{lon}",  # Latitude and longitude
        "fov": 90,  # Field of view
        "heading": 0,  # Camera direction
        "pitch": 0,  # Camera angle
        "key": googlemaps_api_key
    }
    street_view_url = f"{base_street_view_url}?{requests.compat.urlencode(street_view_params)}"
    
    # 2. Get zoomed-in satellite view
    static_map_params = {
        "center": f"{lat},{lon}",  # Latitude and longitude
        "zoom": 7,  # Zoom level (higher values for closer views)
        "size": "600x400",  # Image size
        "maptype": "satellite",  # Satellite view
        "key": googlemaps_api_key
    }
    satellite_view_url = f"{base_static_map_url}?{requests.compat.urlencode(static_map_params)}"

    satellite_live_url = f"{base_maps_url}/?q={lat},{lon}&t=k"

    return street_view_url, satellite_view_url, satellite_live_url


def get_random_city(winner):
    # Select a random city from the list
    random_city = random.choice(cities)
    city_name = random_city["city"]
    country_name = random_city["country"]
    is_capital = random_city["capital"]
    lat = random_city["lat"]
    lon = random_city["lon"]

    # Conversion factors
    miles_per_lat_degree = 1 / 69  # 1 degree latitude ≈ 69 miles
    miles_per_lon_degree = 1 / (69 * math.cos(math.radians(lat)))  # Adjust for latitude

    # Generate random offsets within ±0.5 miles (to stay within 1 square mile)
    lat_offset = random.uniform(-0.5, 0.5) * miles_per_lat_degree
    lon_offset = random.uniform(-0.5, 0.5) * miles_per_lon_degree

    # Apply offsets to the original latitude and longitude
    lat = lat + lat_offset
    lon = lon + lon_offset
    
    # OpenWeather Current Weather API URL
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    # Parameters for the API
    params = {
        #"q": city_name,  # Query by city name
        "lat": lat,
        "lon": lon,
        "appid": openweather_api_key,  # Your API key
        "units": "metric"  # Metric for temperature in Celsius
    }
    
    # Make the API call
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        
        # Extract weather information
        temperature_c = data["main"]["temp"]
        #temperature_c_feelslike = data["main"]["feels_like"]
        temperature_f = temperature_c * 9 / 5 + 32
        temperature_f = round(temperature_f)
        temperature_c = round(temperature_c)
        #temperature_f_feelslike = temperature_c_feelslike * 9 / 5 + 32
        humidity = data["main"]["humidity"]
        weather_conditions = ". ".join([item["description"].capitalize() for item in data["weather"]]) + "."
        timezone_offset = data["timezone"]  # Timezone offset in seconds
        local_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=timezone_offset)
        local_time_str = local_time.strftime("%B %-d, %Y %-I:%M%p").lower()
        returned_name = data["name"]

        input_text = (
            f"Fahrenheit Temperature: {temperature_f}°F\n"
            f"Celsius Temperature: {temperature_c}°C\n"
            f"Weather Conditions: {weather_conditions}\n"
            f"Local Date and Time: {local_time_str}\n"
        )

        print(input_text)
        
        try:
        # Call OpenAI GPT-4 to generate a category
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are OkraStrut and are running from @{winner} in the style of 'Where in the world is Carmen San Diego?'. Incorporate the given weather and time data into a transmission that's been intercepted and sent to @{winner} who will then have to guess what city you're in. Make sure to incorporate all the following facts in your output. Be creative and use emojis to make your response more engaging, but don't make up any facts about the city. "
                    },
                    {
                        "role": "user",
                        "content": f"Please incorporate the following facts into a mysterious transmission up to 4 sentences in length:\n\n{input_text}"
                    }
                ],
                max_tokens=500,  # Limit response length
                temperature=0.3  # Lower temperature for more focused output
            )
            
            # Extract the generated category
            location_clue = response["choices"][0]["message"]["content"].strip()
            if location_clue == None:
                location_clue = "I'm somewhere mysterious. Figure it out from that."
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            location_clue = "I'm somewhere mysterious. Figure it out from that."

    else:
        return {"error": f"Failed to fetch weather data for {city_name}, {country_name} (status code: {response.status_code})"}

    street_view_url, satellite_view_url, satellite_view_live_url = get_google_maps(lat, lon)
    
    return city_name, country_name, "World Capital", location_clue, street_view_url, satellite_view_url, satellite_view_live_url



def categorize_text(input_text, title):
    try:
        # Call OpenAI GPT-4 to generate a category
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a categorization assistant. Your job is to analyze text and return a 1-2 word category that best describes the content. Do not include words from the title in the category."
                },
                {
                    "role": "user",
                    "content": f"Title: {title}\n\nPlease categorize the following text into a 1-2 word category:\n\n{input_text}"
                }
            ],
            max_tokens=10,  # Limit response length
            temperature=0.3  # Lower temperature for more focused output
        )
        
        # Extract the generated category
        category = response["choices"][0]["message"]["content"].strip()

        # Normalize title and category for comparison
        title_words = set(re.sub(r"[^\w\s]", "", title).lower().split())  # Remove punctuation and split
        category_words = set(re.sub(r"[^\w\s]", "", category).lower().split())  # Remove punctuation and split

        # Ensure the category does not overlap with the title words
        if category_words & title_words:  # Check for intersection
            return "Hint Fail"  # Fallback if there's a match
        return category

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Unknown"


def get_wikipedia_article(max_words=3, max_length=16):
    base_url = "https://en.wikipedia.org/w/api.php"
    
    while True:
        # Fetch a random page from Wikipedia
        response = requests.get(base_url, {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": 0,  # Only fetch content pages
            "grnlimit": 1  # Fetch one page at a time
        })
        
        if response.status_code != 200:
            print("Error fetching from Wikipedia API")
            return None, None, None
        
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page_info in pages.items():
            title = page_info.get("title", "")

            if not title.replace(" ", "").isalpha():
                continue
            
            norm_title = remove_diacritics(title)

            # Check if the title has at most `max_words` and is within `max_length` characters
            word_count = len(title.split())
            total_length = len(title)
            if word_count <= 2 and total_length <= 11 and total_length >= 4:
                # Check capitalization rule
                words = title.split()
                if len(words) > 1 and any(word[0].isupper() for word in words[1:]):
                    continue
                
                # Fetch the introductory text (summary)
                pageid = page_info.get("pageid")
                intro_text = fetch_wikipedia_intro(pageid)
                redacted_text = redact_intro_text(title, intro_text)
                category = categorize_text(intro_text, title)

                # Construct the Wikipedia article URL
                wiki_url = f"https://en.wikipedia.org/wiki/{quote(title)}"

                return norm_title, redacted_text, category, wiki_url


def fetch_wikipedia_intro(pageid):
    base_url = "https://en.wikipedia.org/w/api.php"
    response = requests.get(base_url, {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,  # Fetch only the introductory text
        "explaintext": True,  # Return plaintext instead of HTML
        "pageids": pageid
    })
    
    if response.status_code != 200:
        print("Error fetching Wikipedia introduction")
        return None
    
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    return pages.get(str(pageid), {}).get("extract", "")


def redact_intro_text(title, intro_text):
    if not title or not intro_text:
        return intro_text
    
    # Split the title into words and build a regex pattern
    words_to_redact = [re.escape(word) for word in title.split()]
    pattern = re.compile(r'\b(' + '|'.join(words_to_redact) + r')\b', re.IGNORECASE)
    
    # Replace matching words with "REDACTED"
    redacted_text = pattern.sub("OKRA", intro_text)
    return redacted_text


def get_first_category(pageid):
    base_url = "https://en.wikipedia.org/w/api.php"
    response = requests.get(base_url, {
        "action": "query",
        "format": "json",
        "prop": "categories",
        "cllimit": "1",  # Fetch only the first category
        "pageids": pageid
    })
    
    if response.status_code != 200:
        print("Error fetching from Wikipedia API")
        return None
    
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    page_data = pages.get(str(pageid), {})
    
    # Extract the first category
    categories = page_data.get("categories", [])
    if categories:
        return categories[0].get("title", "").replace("Category:", "")
    return None

 

def describe_image_with_vision(image_url, mode):
    try:

        if mode == "okra-title":
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a cool image analyst. Your goal is to create image titles of portaits that roasts people."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Based on what you see in the image, give the image a name with 5 words maximum and ensure the name is okra themed. Your goal is to humiliate the person the portrait is of."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }
         
        elif mode == "roast-title":
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a cool image analyst. Your goal is to create image titles of portaits that roasts people."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Based on what you see in the image, give the image a name with 5 words maximum. Your goal is to humiliate the person the portrait is of."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }


        elif mode == "title":
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a cool image analyst. Your goal is to create image titles."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Based on what you see in the image, give the image a name with 5 words maximum."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }
             
        else:
            payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an image analyst. Your goal is to accurately describe the image to provide to someone as accurately as possible."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what you see in this image as accurately as you can."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }

        # Call the OpenAI API with the payload
        response = openai.ChatCompletion.create(**payload)

        # Extract and return the description from the response
        description = response["choices"][0]["message"]["content"]
        return description

    except Exception as e:
        print(f"Error describing the image: {e}")
        return None






def get_user_data(username):
    try:
        # Fetch user profile
        user = reddit.redditor(username)
        avatar_url = user.icon_img  # User avatar URL

        # Fetch last 100 submissions (posts)
        submissions = list(user.submissions.new(limit=100))

        # Fetch last 100 comments
        comments = list(user.comments.new(limit=100))

        # Combine submissions and comments into a single list
        activities = []
        for submission in submissions:
            activities.append({
                "type": "submission",
                "created_utc": submission.created_utc,
                "subreddit": submission.subreddit.display_name
            })

        for comment in comments:
            activities.append({
                "type": "comment",
                "created_utc": comment.created_utc,
                "subreddit": comment.subreddit.display_name
            })

        # Sort activities by creation time in descending order
        activities.sort(key=lambda x: x["created_utc"], reverse=True)

        # Take the most recent 100 activities
        recent_activities = activities[:100]

        # Count the number of posts/comments per subreddit
        subreddit_counts = Counter()
        for activity in recent_activities:
            subreddit = activity["subreddit"]
            subreddit_counts[subreddit] += 1

        # Get the top 5 subreddits without counts
        top_subreddits = ", ".join([subreddit for subreddit, _ in subreddit_counts.most_common(5)])

        return {
            "avatar_url": avatar_url,
            "top_subreddits": top_subreddits
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def sovereign_check(user):
    db = connect_to_mongodb()
    sovereigns = {sovereign['user'] for sovereign in db.hall_of_sovereigns.find()}
    if user in sovereigns:
        return True
    else:
        return False


def upload_image_to_s3(buffer, winner, description):
    try:
        bucket_name='triviabotwebsite'
        folder_name='generated-images'

        # Step 2: Determine object name if not provided
        pst = pytz.timezone('America/Los_Angeles')
        now = datetime.datetime.now(pst)
        formatted_time = now.strftime('%B %d, %Y %H%M')  # Format: "November 25, 2024 1950"
        object_name = f"{folder_name}/{description} & {winner} ({formatted_time}).png"
             
        # Step 3: Connect to S3 and upload the file
        s3_client = boto3.client("s3")
        s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=buffer.getvalue(), ContentType="image/png")

        # Step 4: Generate and return the S3 URL
        return None

    except (BotoCoreError, ClientError) as boto_err:
        print(f"Error uploading to S3: {boto_err}")
        return None

def load_parameters():
    global image_wins
    global num_mysterybox_clues_default
    global num_crossword_clues_default
    global num_jeopardy_clues_default
    global num_wof_clues_default
    global num_wof_clues_final_default
    global num_mysterybox_clues
    global num_crossword_clues
    global num_jeopardy_clues
    global num_wof_clues
    global num_wof_clues_final
    global num_wf_letters
    global num_math_questions_default
    global num_math_questions
    global num_stats_questions_default
    global num_stats_questions
 
    
    # Default values
    default_values = {
        "image_wins": 5,
        "num_mysterybox_clues_default": 3,
        "num_crossword_clues_default": 0,
        "num_jeopardy_clues_default": 3,
        "num_wof_clues_default": 0,
        "num_wof_clues_final_default": 3,
        "num_wf_letters": 3,
        "num_math_questions_default": 0,
        "num_stats_questions_default": 0
    }

    
    for attempt in range(max_retries):
        try:
            db = connect_to_mongodb()

            # Retrieve all parameter documents
            documents = db.parameters.find()

            # Initialize variables with defaults
            parameters = {key: default_values[key] for key in default_values}

            # Overwrite defaults with values from the database
            for doc in documents:
                if doc["_id"] in parameters:
                    parameters[doc["_id"]] = int(doc.get("value", parameters[doc["_id"]]))

            # Assign global variables
            image_wins = parameters["image_wins"]
            num_mysterybox_clues_default = parameters["num_mysterybox_clues_default"]
            num_crossword_clues_default = parameters["num_crossword_clues_default"]
            num_jeopardy_clues_default = parameters["num_jeopardy_clues_default"]
            num_wof_clues_default = parameters["num_wof_clues_default"]
            num_wof_clues_final_default = parameters["num_wof_clues_final_default"]
            num_wf_letters = parameters["num_wf_letters"]
            num_math_questions_default = parameters["num_math_questions_default"]
            num_stats_questions_default = parameters["num_stats_questions_default"]

            num_mysterybox_clues = num_mysterybox_clues_default
            num_crossword_clues = num_crossword_clues_default
            num_jeopardy_clues = num_jeopardy_clues_default
            num_wof_clues = num_wof_clues_default
            num_wof_clues_final = num_wof_clues_final_default
            num_math_questions = num_math_questions_default
            num_stats_questions = num_stats_questions_default
            # Exit loop if successful
            break

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay_between_retries} seconds...")
                time.sleep(delay_between_retries)
            else:
                print("Max retries reached. Data loading failed.")
                # Set all variables to defaults if loading fails
                image_wins = default_values["image_wins"]
                num_mysterybox_clues_default = default_values["num_mysterybox_clues_default"]
                num_crossword_clues_default = default_values["num_crossword_clues_default"]
                num_jeopardy_clues_default = default_values["num_jeopardy_clues_default"]
                num_wof_clues_default = default_values["num_wof_clues_default"]
                num_wof_clues_final_default = default_values["num_wof_clues_final_default"]
                num_wf_letters = default_values["num_wf_letters"]
                num_math_questions_default = default_values["num_math_questions_default"]
                num_stats_questions_default = default_values["num_stats_questions_default"]

                num_mysterybox_clues = num_mysterybox_clues_default
                num_crossword_clues = num_crossword_clues_default
                num_jeopardy_clues = num_jeopardy_clues_default
                num_wof_clues = num_wof_clues_default
                num_wof_clues_final = num_wof_clues_final_default
                num_math_questions = num_math_questions_default
                num_stats_questions = num_stats_questions_default


def nice_creep_okra_option(winner):
    global since_token, params, headers, max_retries, delay_between_retries, nice_okra, creep_okra, wf_winner
    nice_okra = False
    creep_okra = False
    wf_winner = False

    sync_url = f"{matrix_base_url}/sync"
    processed_events = set()  # Track processed event IDs to avoid duplicates
    
    # Initialize the sync and message to prompt user for letters
    initialize_sync()
    start_time = time.time()  # Track when the question starts
    
    message = f"\n☕🤝 Thank you @{winner} for your support.\n\n" 
    message += f"🥒😊 Say 'okra' and I'll be nice.\n"
    message += f"👀🔭 Say 'creep' and I'll snoop your Reddit profile.\n"
    message += f"🔥🍗 Say neither and I roast you.\n\n"
    send_message(target_room_id, message)
    
    while time.time() - start_time < magic_time:
        try:                
            if since_token:
                params["since"] = since_token

            time.sleep(1)
            response = requests.get(sync_url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"Unexpected status code: {response.status_code}")
                continue

            sync_data = response.json()
            since_token = sync_data.get("next_batch")  # Update since_token for the next batch
            room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])

            for event in room_events:                
                event_id = event["event_id"]
                event_type = event.get("type")

                # Only process and redact if the event type is "m.room.message"
                if event_type == "m.room.message":
                    
                    # Skip processing if this event_id was already processed
                    if event_id in processed_events:
                        continue
    
                    # Add event_id to the set of processed events
                    processed_events.add(event_id)
                    sender = event["sender"]
                    sender_display_name = get_display_name(sender)
                    message_content = event.get("content", {}).get("body", "").upper()

                    if sender == bot_user_id or sender_display_name != winner:
                        continue

                    if "okra" in message_content.lower():
                        react_to_message(event_id, target_room_id, "okra21")
                        nice_okra = True
                        wf_winner = True
                        creep_okra = False
                        return None
                        
                    if "creep" in message_content.lower():
                        react_to_message(event_id, target_room_id, "okra21")
                        creep_okra = True
                        nice_okra = False
                        wf_winner = False
                        return None
                        
                    if "neither" in message_content.lower():
                        react_to_message(event_id, target_room_id, "okra21")
                        nice_okra = False
                        creep_okra = False
                        wf_winner = False
                        return None
                        
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error collecting responses: {e}")

    return None


def generate_round_summary_image(round_data, winner):
    winner_coffees = get_coffees(winner)
    winner_at = f"@{winner}"
    
    if winner == "OkraStrut":
        prompt = (
            "The setting is a fiery Hell, where a giant and angry piece of okra holds a massive golden trophy while looking down on and smiting all other players. "
            "The atmosphere is angry, scary, and full of malice."
        )
        message = "🥒OKRA!! 🥒OKRA!! 🥒OKRA!!\n"
        
    elif winner_coffees > 100:
        prompt = (
            f"An anthropomorphism of {winner} surrounded by okra and money. "
            "Add glowing lights, hearts, and a festive atmosphere."
        )
        message = f"✊🔥 {winner_at}, thank you for your donation to the cause. And nice streak!\n"
    
    else:
        categories = {
            "0": "😠🥒 Okrap (Horror)",
            "1": "🌹🏰 Okrenaissance",
            "2": "😇✨ Okroly and Divine",
            "3": "🎲🔀 (OK)Random",
            "4": f"🖼️🔤 Okraverse (Interactive) ☕☕",
            "5": f"🖼️👤 Okravatar (Reddit Avatar) ☕☕",
            "6": f"🖼️📜 Okraricature (Top Subreddits) ☕☕"
        }

        # Ask the user to choose a category
        selected_category, additional_prompt = ask_category(winner, categories, winner_coffees)
        reddit_avatar_description = ""
        top_subreddits = ""

        if selected_category == "5" or selected_category == "6":
            try:
                user_data = get_user_data(winner)
                if not user_data:
                    print(f"Failed to fetch data for {winner}.")
                    user_data = {"avatar_url": "N/A", "subreddits": []}
            except Exception as e:
                print(f"Error fetching Reddit data for {winner}: {e}")
                user_data = {"avatar_url": "N/A", "subreddits": []}
        
            # Extract user data
            reddit_avatar_url = user_data.get("avatar_url", "No avatar available.")
            reddit_avatar_description = describe_image_with_vision(reddit_avatar_url, "describe")
            top_subreddits = user_data.get("top_subreddits", "")
                    
        prompts_by_category = {
            "0": [
                f"{winner} is being chased by someone holding an okra in a hyperrealistic scary horror movie type situation. The image should be in 1st person view of the person chasing {winner} and elicit intense fear. Try hard to bring and merge elements from the username {winner} into their humanoid depiction."
            ],
            "1": [
                f"A Renaissance painting of {winner} holding an okra. Make the painting elegant and refined. Try hard to bring and merge elements from their username {winner} into a humanoid depiction of them."
            ],
            "2": [
                f"{winner} worshipping an okra. Make it appealing and accepting of religions of all types. Try hard to bring and merge elements from their username {winner} into a humanoid depiction of them."
            ],
            "3": [
                f"{winner} intereracting with an okra in the most crazy, ridiculous, and over the top random way. Try hard to bring and merge elements from their username {winner} into a humanoid depiction of them."
            ],
            "4": [
                f"Draw an okra themed picture that MUST include '{additional_prompt}'.\n"
            ],
            "5": [
                f"Draw an okra themed picture of what you think {winner} looks like based on their avatar, which looks like '{reddit_avatar_description}'.\n"
            ],
            "6": [
                f"Draw an okra themed caricature of what you think {winner} looks like based on their 5 most visited subreddits, which are '{top_subreddits}'.\n"
            ]
        }

        # Select a prompt based on the chosen category
        if selected_category and selected_category in prompts_by_category:
            prompt = random.choice(prompts_by_category[selected_category])
        else:
            prompt = f"{winner} is being chased by someone holding an okra in a hyperrealistic scary horror movie setting. The image should be in 1st person view of the person chasing {winner}. Try hard to bring and merge elements from their username {winner} into a humanoid depiction of them."

            
    print(prompt)
    
    # Generate the image using DALL-E
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"  # Adjust size as needed
        )
        # Return the image URL from the API response
        image_url = response["data"][0]["url"]
        
        if selected_category == "4" or selected_category == "5" or selected_category == "6":
            image_description = describe_image_with_vision(image_url, "okra-title")
        else:
            image_description = describe_image_with_vision(image_url, "okra-title")

            
        image_mxc, image_width, image_height = download_image_from_url(image_url)
        send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)

        message = f"🔥💖 {winner_at} nice streak. I drew this of you.\n"
        message += f"\nI call this masterpiece '{image_description}'\n"
        message += "\n🥒🏛️ https://redditlivetrivia.com/okra-museum\n"
        send_message(target_room_id, message)

         # Download and resize the image to 256x256
        image_data = requests.get(image_url).content
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((256, 256))  # Resize to 256x256
        
        # Save resized image to a buffer
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        upload_image_to_s3(buffer, winner, image_description)
        return None
        
    except openai.OpenAIError as e:
        print(f"Error generating image: {e}")
        # Check if the error is due to the safety system
        if "Your request was rejected as a result of our safety system" in str(e):
            # Use a default safe prompt
            default_prompt = f"A Renaissance painting of {winner} holding an okra. Make the painting elegant and refined. Try hard to bring and merge elements from their username {winner} into a humanoid depiction of them."
            try:
                response = openai.Image.create(
                    prompt=default_prompt,
                    n=1,
                    size="512x512"
                )
                
                # Return the image URL from the API response
                image_url = response["data"][0]["url"]
                image_description = describe_image_with_vision(image_url, "okra-title")
                image_mxc, image_width, image_height = download_image_from_url(image_url)
                send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)
        
                message = f"😈😉 {winner_at} Naughty naughty, I'll have to pick another.\n\n"
                message += f"🔥💖 Nice streak. I drew this of you.\n"
                message += f"\nI call this masterpiece '{image_description}'\n"
                message += "\n🥒🏛️ https://redditlivetrivia.com/okra-museum\n"
                send_message(target_room_id, message)
        
                 # Download and resize the image to 256x256
                image_data = requests.get(image_url).content
                image = Image.open(io.BytesIO(image_data))
                image = image.resize((256, 256))  # Resize to 256x256
                
                # Save resized image to a buffer
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                
                upload_image_to_s3(buffer, winner, image_description)
                return None
            
            except openai.OpenAIError as e2:
                print(f"Error generating default image: {e2}")
                return "Image generation failed!"
        else:
            return "Image generation failed!"


def ask_category(winner, categories, winner_coffees):
    """
    Ask the winner to choose a category and return their choice, or None if no valid response is received.
    """
    global since_token, params, headers, max_retries, delay_between_retries

    sync_url = f"{matrix_base_url}/sync"
    additional_prompt = ""
    processed_events = set()  # Track processed event IDs to avoid duplicates

    # Display categories
    category_message = f"\n🎨🖍️ @{winner} Pick one. Some require ☕☕.\n\n"
    for key, value in categories.items():
        category_message += f"{key}: {value}\n"
    send_message(target_room_id, category_message)

    start_time = time.time()  # Track when the question starts
    
    while time.time() - start_time < magic_time:
        try:
            if since_token:
                params["since"] = since_token

            response = requests.get(sync_url, headers=headers, params=params)

            if response.status_code != 200:
                continue

            sync_data = response.json()
            since_token = sync_data.get("next_batch")  # Update since_token for the next batch
            
            if since_token:
                params["since"] = since_token
                
            room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])

            for event in room_events:
                event_id = event["event_id"]
                event_type = event.get("type")

                if event_type == "m.room.message" and event_id not in processed_events:
                    processed_events.add(event_id)
                    sender = event["sender"]
                    sender_display_name = get_display_name(sender)
                    message_content = event.get("content", {}).get("body", "").strip()

                    if sender == bot_user_id or sender_display_name != winner:
                        continue

                    if message_content not in categories:
                        react_to_message(event_id, target_room_id, "okra5")
                        continue

                    # Check if the winner can select options A, B, or C
                    if message_content in ['4', '5', '6'] and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. Choice {message_content} requires ☕️☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    react_to_message(event_id, target_room_id, "okra21")
                    message = f"\n💪🛡️ I got you {winner}. {message_content} it is.\n"
                    send_message(target_room_id, message)

                    if message_content in ['4'] and winner_coffees > 0:
                        additional_prompt = request_prompt(winner, processed_events)    
                    
                    return message_content, additional_prompt
    
        except requests.exceptions.RequestException as e:
            print(f"Error collecting responses: {e}")  
    
    # Return None if no valid response is received within the time limit
    message = f"🐢⏳ Too slow. Okra time.\n"
    send_message(target_room_id, message)
    return None, additional_prompt


def request_prompt(winner, done_events):
    global since_token, params, headers, max_retries, delay_between_retries, magic_time, bot_user_id, target_room_id

    sync_url = f"{matrix_base_url}/sync"
    processed_events = done_events  # Track processed event IDs to avoid duplicates

    # Initialize the sync and message to prompt user for input
    initialize_sync()

    if since_token:
        params["since"] = since_token
        
    start_time = time.time()  # Track when the question starts
    message = f"\n🖼️🔟 @{winner}, customize your image (10 words max). Be good.\n"
    send_message(target_room_id, message)

    collected_words = []

    while time.time() - start_time < magic_time:
        try:
            time.sleep(1)  # Slight delay to avoid overwhelming the server
            if since_token:
                params["since"] = since_token
                
            if len(collected_words) >= 10:
                break
                
            response = requests.get(sync_url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"Unexpected status code: {response.status_code}")
                continue

            sync_data = response.json()
            since_token = sync_data.get("next_batch")  # Update since_token for the next batch
            room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])

            for event in room_events:
                event_id = event["event_id"]
                event_type = event.get("type")

                # Only process if the event type is "m.room.message"
                if event_type == "m.room.message" and event_id not in processed_events:
                    processed_events.add(event_id)
                    sender = event["sender"]
                    sender_display_name = get_display_name(sender)
                    message_content = event.get("content", {}).get("body", "").strip()

                    if sender == bot_user_id or sender_display_name != winner:
                        continue

                    # Split the message content into words and add them to collected_words
                    words = message_content.split()
                    for word in words:
                        if len(collected_words) < 10:
                            collected_words.append(word)
                        else:
                            break

                    # React to the user's message
                    react_to_message(event_id, target_room_id, "okra21")

                    # Check if we have collected enough words
                    if len(collected_words) >= 10:
                        break

        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error collecting responses: {e}")

    if not collected_words:
        message = "Nothing. Okra time."
    else:
        message = f"Ok(ra) I got: {' '.join(collected_words)}"
    send_message(target_room_id, message)
    return ' '.join(collected_words)







def get_coffees(username):
    db = connect_to_mongodb()
    donors_collection = db["donors"]  # Ensure this matches the name of your collection

    pipeline = [
        {"$match": {"name": username}},  # Filter by username
        {"$group": {  # Group by username and calculate total coffees
            "_id": "$name",
            "total_coffees": {"$sum": "$coffees"}
        }}
    ]

    result = list(donors_collection.aggregate(pipeline))
    return int(result[0]["total_coffees"]) if result else 0



def fetch_donations():
    base_url = "https://developers.buymeacoffee.com/api/v1/supporters"
    headers = {"Authorization": f"Bearer {buymeacoffee_api_key}"}

    try:
        db = connect_to_mongodb()  # Connect to the MongoDB database
        donors_collection = db["donors"]  # Use the 'donors' collection

        new_donors = []
        next_page_url = base_url  # Start with the base URL

        while next_page_url:
            response = requests.get(next_page_url, headers=headers)
            response.raise_for_status()  # Raise an error for HTTP issues
            
            # Parse and validate JSON response
            try:
                api_response = response.json()
            except ValueError as ve:
                print(f"Error parsing JSON: {ve}")
                break

            # Extract donations list from the 'data' key
            donations = api_response.get("data", [])
            if not isinstance(donations, list):  # Validate expected data type
                print(f"Unexpected donations format: {type(donations)}. Donations: {donations}")
                break

            for donor in donations:
                if isinstance(donor, dict):  # Ensure donor is a dictionary
                    # Extract and process donor details
                    donor_id = donor.get("support_id")
                    donor_name = donor.get("supporter_name")
                    donor_coffees = donor.get("support_coffees")
                    donor_coffee_price = donor.get("support_coffee_price")
                    donor_date = donor.get("support_created_on")

                    # Check if donor already exists in MongoDB
                    if not donors_collection.find_one({"donor_id": donor_id}):
                        new_donor = {
                            "donor_id": donor_id,
                            "name": donor_name,
                            "coffees": donor_coffees,
                            "coffee_price": donor_coffee_price,
                            "date": donor_date
                        }
                        try:
                            donors_collection.insert_one(new_donor)  # Insert into MongoDB
                            new_donors.append(new_donor)
                        except Exception as e:
                            print(f"Error inserting donor into MongoDB: {e}")
                else:
                    print(f"Skipping invalid donor format: {donor}")

            # Get the next page URL from the response
            next_page_url = api_response.get("next_page_url")

        print(f"New donors added: {new_donors}")
        return new_donors

    except requests.exceptions.RequestException as e:
        print(f"Error fetching donations: {e}")
        sentry_sdk.capture_exception(e)
        return []

    except Exception as e:
        print(f"Unexpected error: {e}")
        sentry_sdk.capture_exception(e)
        return []


def get_math_question():
    question_functions = [create_mean_question, create_median_question, create_derivative_question, create_sum_factors_question, create_product_factors_question, create_factors_question]
    selected_question_function = random.choice(question_functions)
    return selected_question_function()

        
def get_stats_question():
    question_functions = [create_mean_question, create_median_question]
    selected_question_function = random.choice(question_functions)
    return selected_question_function()

# Function to create a mean question in dictionary format
def create_mean_question():
    return {
        "category": "Mathematics",
        "question": "What is the MEAN of the following set?",
        "url": "mean",
        "answers": [""]
    }

# Function to create a median question in dictionary format
def create_median_question():
    return {
        "category": "Mathematics",
        "question": "What is the MEDIAN of the following set?",
        "url": "median",
        "answers": [""]
    }

# Function to create a derivative question in dictionary format
def create_derivative_question():
    return {
        "category": "Mathematics",
        "question": "What is the DERIVATIVE with respect to x?",
        "url": "derivative",
        "answers": [""]
    }

def create_sum_factors_question():
    return {
        "category": "Mathematics",
        "question": "What is the SUM of the below polynomial's factors?",
        "url": "polynomial sum",
        "answers": [""]
    }

def create_product_factors_question():
    return {
        "category": "Mathematics",
        "question": "What is the PRODUCT of the below polynomial's factors?",
        "url": "polynomial product",
        "answers": [""]
    }

def create_factors_question():
    return {
        "category": "Mathematics",
        "question": "What are the 2 FACTORS of the polynomial below?",
        "url": "polynomial factors",
        "answers": [""]
    }
        
def select_wof_questions(winner):
    global fixed_letters
    
    try:
        time.sleep(2)
        db = connect_to_mongodb()
        recent_wof_ids = get_recent_question_ids_from_mongo("wof")
        selected_questions = []
        fixed_letters = ['O', 'K', 'R', 'A']

        # Fetch wheel of fortune questions using the random subset method
        wof_collection = db["wof_questions"]
        pipeline_wof = [
            {"$match": {"_id": {"$nin": list(recent_wof_ids)}}},  # Exclude recent IDs
            {"$group": {  # Group by question text to ensure uniqueness
                "_id": "$question",  # Group by the question text field
                "question_doc": {"$first": "$$ROOT"}  # Select the first document with each unique text
            }},
            {"$replaceRoot": {"newRoot": "$question_doc"}},  # Flatten the grouped results
            {"$sample": {"size": 3}}  # Sample 3 unique questions
        ]

        wof_questions = list(wof_collection.aggregate(pipeline_wof))
        #print(wof_questions)

        message = f"\n🍷⚔️ @{winner}: Choose wisely.  Some require ☕☕.\n\n"
        # Assuming wof_questions contains the sampled questions, with each document as a list/tuple
        counter = 1
        for doc in wof_questions:
            category = doc["question"]  # Use the key name to access category
            message += f"{counter}. {category}\n"
            counter = counter + 1
        premium_counts = counter
        message += f"{counter}. 🌐🎲 Wikipedia Roulette ☕☕\n"
        counter = counter + 1
        message += f"{counter}. 🌍❔ Where's Okra? ☕☕\n"
        send_message(target_room_id, message)  

        selected_wof_category = ask_wof_number(winner)
        
        if int(selected_wof_category) < premium_counts:
            wof_question = wof_questions[int(selected_wof_category) - 1]
            wof_answer = wof_question["answers"][0]
            wof_clue = wof_question["question"]
                    
            wof_question_id = wof_question["_id"]  # Get the ID of the selected question
            if wof_question_id:
                store_question_ids_in_mongo([wof_question_id], "wof")  # Store it as a list containing a single ID
        
        elif selected_wof_category == "4":
            wof_answer, redacted_intro, wof_clue, wiki_url = get_wikipedia_article(3, 16)
            wikipedia_message = f"\n🥒⬛ Okracted Clue:\n\n{redacted_intro}\n"
            send_message(target_room_id, wikipedia_message)
            time.sleep(3)

        elif selected_wof_category == "5":
            wof_answer, country_name, wof_clue, location_clue, street_view_url, satellite_view_url, satellite_view_live_url = get_random_city(winner)
            location_clue = f"\n🌦️📊 Transmission Intercepted\n\n{location_clue}\n"
            send_message(target_room_id, location_clue)
            fixed_letters = []
            time.sleep(3)

            if image_questions == True:    
                image_size = 100
                street_view_mxc, street_view_width, street_view_height = download_image_from_url(street_view_url)  
                satellite_view_mxc, satellite_view_width, satellite_view_height = download_image_from_url(satellite_view_url)  

                message = "\n🏙️👁️ Street View Obtained\n"
                send_message(target_room_id, message)
                street_response = send_image(target_room_id, street_view_mxc, street_view_width, street_view_height, image_size)
                
                if street_response is None:                      
                    print("Error: Failed to send street image.")
                    
                time.sleep(2)
                
                message = "\n🛰️🌍 Satellite View Obtained\n"
                send_message(target_room_id, message)
                satellite_response = send_image(target_room_id, satellite_view_mxc, satellite_view_width, satellite_view_height, image_size)
                
                if satellite_response is None:                      
                    print("Error: Failed to send satellite image.")
                    
                time.sleep(2)
                
            else:
                message = f"\n🚫📷 Sorry {winner}, no map views since 'Blank' mode is on.\n"
                send_message(target_room_id, message)
                


        image_mxc, image_width, image_height, display_string = generate_wof_image(wof_answer, wof_clue, fixed_letters)
        print(f"{wof_clue}: {wof_answer}")
            
        image_size = 100
        
        if image_questions == True:    
            response = send_image(target_room_id, image_mxc, image_width, image_height, image_size)
            if response is None:                      
                print("Error: Failed to send image.")
        else:
            fixed_letters_str = "Revealed Letters: " + ' '.join(fixed_letters)
            message = f"{display_string}\n{wof_clue}\n{fixed_letters_str}\n"
            send_message(target_room_id, message)    

            
        wof_letters = ask_wof_letters(winner, wof_answer, 5)
        
        if wf_winner == False:
            time.sleep(1.5)
            image_mxc, image_width, image_height, display_string = generate_wof_image(wof_answer, wof_clue, wof_letters) 
            
            if image_questions == True:
                response = send_image(target_room_id, image_mxc, image_width, image_height, image_size)
                if response is None:                      
                    print("Error: Failed to send image.")
            else:
                wof_letters_str = "Revealed Letters: " + ' '.join(wof_letters)
                message = f"{display_string}\n{wof_clue}\n{wof_letters_str}\n"
                send_message(target_room_id, message)

            process_wof_guesses(winner, wof_answer, 5)

        if selected_wof_category == "4":
            time.sleep(1.5)
            wikipedia_message = f"\n🌐📄 Wikipedia Link: {wiki_url}\n"
            send_message(target_room_id, wikipedia_message)
            time.sleep(1.5)

        if selected_wof_category == "5":
            time.sleep(1.5)
            maps_message = f"\n🌍❔ Okra's Location: {satellite_view_live_url}\n"
            send_message(target_room_id, maps_message)
            time.sleep(1.5)
            
        return None

    except Exception as e:
        # Capture the exception in Sentry and print detailed error information
        sentry_sdk.capture_exception(e)
        
        # Print a detailed error message with traceback
        error_details = traceback.format_exc()
        print(f"Error selecting wof questions: {e}\nDetailed traceback:\n{error_details}")
        
        return None  # Return an empty list in case of failure

    

def process_wof_guesses(winner, answer, extra_time):
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner

    sync_url = f"{matrix_base_url}/sync"
    processed_events = set()  # Track processed event IDs to avoid duplicates
    answer = answer.upper()  # Normalize the answer to uppercase for comparison
    
    # Initialize the sync and send message to prompt user for a guess
    initialize_sync()
    start_time = time.time()  # Track when the question starts
    message = f"\n@{winner} ❓Your Answer❓\n"
    send_message(target_room_id, message)
    
    while time.time() - start_time < (magic_time + extra_time):
        try:
            if since_token:
                params["since"] = since_token
                
            time.sleep(1)
            response = requests.get(sync_url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"Unexpected status code: {response.status_code}")
                continue

            sync_data = response.json()
            since_token = sync_data.get("next_batch")  # Update since_token for the next batch

            room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])

            for event in room_events:
                event_id = event["event_id"]
                event_type = event.get("type")

                # Only process and redact if the event type is "m.room.message"
                if event_type == "m.room.message":
                    
                    # Skip processing if this event_id was already processed
                    if event_id in processed_events:
                        continue
    
                    # Add event_id to the set of processed events
                    processed_events.add(event_id)
                    sender = event["sender"]
                    sender_display_name = get_display_name(sender)
                    message_content = event.get("content", {}).get("body", "").upper().strip()

                    if sender == bot_user_id or sender_display_name != winner:
                        continue

                    # Check if the message content matches the answer
                    if message_content == answer:
                        react_to_message(event_id, target_room_id, "okra21")
                        success_message = f"\n✅🎉 Correct @{winner}! {answer}\n"
                        send_message(target_room_id, success_message)
                        wf_winner = True
                        return None

                    # If no valid answer was guessed, react with a neutral reaction
                    react_to_message(event_id, target_room_id, "okra5")
    
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error collecting responses: {e}")

    # If time runs out without a correct guess
    timeout_message = f"⏰ Time's up! {answer}."
    send_message(target_room_id, timeout_message)
    return None



def ask_wof_letters(winner, answer, extra_time):
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner

    answer = answer.upper()
    sync_url = f"{matrix_base_url}/sync"
    processed_events = set()  # Track processed event IDs to avoid duplicates

    # Letters that are automatically provided and should not count towards user selections
    answer_letters = set(answer.upper())
    
    # Initialize the sync and message to prompt user for letters
    initialize_sync()
    start_time = time.time()  # Track when the question starts
    message = f"\n@{winner}:❓Pick {num_wf_letters} Letters❓\n"
    if fixed_letters != []:
        message += f"\n🥒 I'll give you O K R A 🥒\n"
    send_message(target_room_id, message)
    
    wf_letters = []
    
    while time.time() - start_time < (magic_time + extra_time):
        try:
            if len(wf_letters) == num_wf_letters:
                break
                
            if since_token:
                params["since"] = since_token

            time.sleep(1)
            response = requests.get(sync_url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"Unexpected status code: {response.status_code}")
                continue

            sync_data = response.json()
            since_token = sync_data.get("next_batch")  # Update since_token for the next batch
            room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])

            for event in room_events:
                if len(wf_letters) == num_wf_letters:
                    break
                
                event_id = event["event_id"]
                event_type = event.get("type")

                # Only process and redact if the event type is "m.room.message"
                if event_type == "m.room.message":
                    
                    # Skip processing if this event_id was already processed
                    if event_id in processed_events:
                        continue
    
                    # Add event_id to the set of processed events
                    processed_events.add(event_id)
                    sender = event["sender"]
                    sender_display_name = get_display_name(sender)
                    message_content = event.get("content", {}).get("body", "").upper()

                    if sender == bot_user_id or sender_display_name != winner:
                        continue

                    if message_content.upper() == answer:
                        react_to_message(event_id, target_room_id, "okra21")
                        wf_winner = True
                        success_message = f"\n✅🎉 Correct @{winner}! {answer}\n"
                        send_message(target_room_id, success_message)
                        return True
                    
                    # Parse letters from the message content
                    for char in message_content:
                        if char in fixed_letters:
                            continue  # Skip if the letter is one in fixed_letters

                        if len(wf_letters) < num_wf_letters and char.isalpha() and char not in wf_letters:
                            wf_letters.append(char)

                        # Check if we have collected enough letters
                        if len(wf_letters) == num_wf_letters:
                            react_to_message(event_id, target_room_id, "okra21")
                            break
            
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error collecting responses: {e}")

    if len(wf_letters) < num_wf_letters:
        needed_letters = num_wf_letters - len(wf_letters)
        
        available_letters = [l for l in "BCDEFGHIJLMNPQSTUVWXYZ" if l not in wf_letters]
        
        if len(available_letters) < num_wf_letters:
            wf_letters.extend(['Q', 'X', 'Z'][:needed_letters])  
        else:
            wf_letters.extend(random.sample(available_letters, needed_letters))
        
        message = f"Too slow. Let me help you out.\nLet's use: {' '.join(wf_letters)}\n\n"
    else:
        message = f"You picked: {' '.join(wf_letters)}\n\n"

    final_letters = fixed_letters + wf_letters
    send_message(target_room_id, message)
    return final_letters
            

def ask_wof_number(winner):
    global since_token, params, headers, max_retries, delay_between_retries

    sync_url = f"{matrix_base_url}/sync"
    collected_responses = []  # Store all responses
    processed_events = set()  # Track processed event IDs to avoid duplicates

    winner_coffees = get_coffees(winner)
    
    initialize_sync()
    start_time = time.time()  # Track when the question starts
    
    selected_question = 1
    while time.time() - start_time < magic_time:
        try:
            if since_token:
                params["since"] = since_token

            response = requests.get(sync_url, headers=headers, params=params)

            if response.status_code != 200:
                continue

            sync_data = response.json()
            since_token = sync_data.get("next_batch")  # Update since_token for the next batch
            room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])

            for event in room_events:
                event_id = event["event_id"]
                event_type = event.get("type")  # Get the type of the event

                # Only process and redact if the event type is "m.room.message"
                if event_type == "m.room.message":
                    
                    # Skip processing if this event_id was already processed
                    if event_id in processed_events:
                        continue
    
                    # Add event_id to the set of processed events
                    processed_events.add(event_id)
                    sender = event["sender"]
                    sender_display_name = get_display_name(sender)
                    message_content = event.get("content", {}).get("body", "")

                    if sender == bot_user_id or sender_display_name != winner:
                        continue

                    if str(message_content) in {"4"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Wikipedia Roulette' requires ☕️☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    
                    if str(message_content) in {"5"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Where's Okra?' requires ☕️☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"1", "2", "3", "4", "5"}:
                        selected_question = str(message_content)
                        react_to_message(event_id, target_room_id, "okra21")
                        message = f"\n💪🛡️ I got you {winner}. {message_content} it is.\n"
                        send_message(target_room_id, message)
                        return selected_question
                    else:
                        react_to_message(event_id, target_room_id, "okra5")
    
        except requests.exceptions.RequestException as e:
                sentry_sdk.capture_exception(e)
                print(f"Error collecting responses: {e}")                    
    
    return selected_question

        

def generate_wof_image(word, clue, revealed_letters):
    word = word.upper()
    image_mxc = True
    
    # Define colors for the board
    background_color = (0, 0, 0)        # Black background
    tile_border_color = (0, 128, 0)     # Green border around each tile
    tile_fill_color = (255, 255, 255)   # White tile for unfilled letters
    space_tile_color = (0, 128, 0)      # Green tile for spaces between words
    text_color = (0, 0, 0)              # Black text for revealed letters
    clue_color = (255, 255, 255)        # White color for the clue text
    revealed_letters_color = (200, 200, 200)  # Light gray color for the revealed letters

    # Define image size and font properties
    img_width, img_height = 800, 450  # Increase height to make room for the clue and revealed letters
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif-Bold.ttf")  # Use a bold font if available
    font_size = 50
    clue_font_size = 40
    revealed_font_size = 20  # Smaller font for the revealed letters

    # Create a blank image with black background
    img = Image.new('RGB', (img_width, img_height), color=background_color)
    draw = ImageDraw.Draw(img)

    # Load the font
    try:
        font = ImageFont.truetype(font_path, font_size)
        clue_font = ImageFont.truetype(font_path, clue_font_size)
        revealed_font = ImageFont.truetype(font_path, revealed_font_size)  # Less bold font for revealed letters
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return None, None, None, None
    
    # Tile dimensions, spacing, and padding for borders
    tile_width, tile_height = 50, 70
    spacing = 15
    padding = 5  # Padding around each tile for the green border effect

    # Calculate the starting x position to center the board
    total_width = len(word) * (tile_width + spacing) - spacing
    start_x = (img_width - total_width) // 2
    y_position = (img_height - tile_height) // 2 - 60  # Position the tiles higher to make room for the clue

    # Draw tiles for each letter in the word
    for i, char in enumerate(word):
        x_position = start_x + i * (tile_width + spacing)
        
        # Draw a green background rectangle as padding around each tile
        draw.rectangle([x_position - padding, y_position - padding, 
                        x_position + tile_width + padding, y_position + tile_height + padding],
                       fill=tile_border_color)

        if char == " ":
            # Draw a green tile for spaces between words
            draw.rectangle([x_position, y_position, x_position + tile_width, y_position + tile_height],
                           outline=tile_border_color, fill=space_tile_color)
        else:
            # Draw a white tile with green border for letters
            draw.rectangle([x_position, y_position, x_position + tile_width, y_position + tile_height],
                           outline=tile_border_color, fill=tile_fill_color)
            
            # Reveal letter if it is in revealed_letters
            if char in revealed_letters:
                # Center the letter within the tile
                letter_bbox = draw.textbbox((0, 0), char, font=font)
                letter_width = letter_bbox[2] - letter_bbox[0]
                letter_height = letter_bbox[3] - letter_bbox[1]
                
                text_x = x_position + (tile_width - letter_width) // 2
                text_y = y_position + (tile_height - letter_height) // 2
                draw.text((text_x, text_y), char, fill=text_color, font=font)

    # Draw the clue text below the word tiles
    clue_text = clue.upper()
    clue_x = img_width // 2
    clue_y = y_position + tile_height + 50  # Position below the tiles with some padding
    
    # Get the bounding box of the clue text to calculate its width and height
    clue_bbox = draw.textbbox((0, 0), clue_text, font=clue_font)
    clue_width = clue_bbox[2] - clue_bbox[0]
    clue_height = clue_bbox[3] - clue_bbox[1]

    draw.text((clue_x - clue_width // 2, clue_y), clue_text, fill=clue_color, font=clue_font)

    # Draw the revealed letters below the clue text
    revealed_text = f"{' '.join(revealed_letters).upper()}"
    revealed_bbox = draw.textbbox((0, 0), revealed_text, font=revealed_font)
    revealed_width = revealed_bbox[2] - revealed_bbox[0]
    revealed_x = (img_width - revealed_width) // 2
    revealed_y = clue_y + clue_height + 20  # Position below the clue with padding

    draw.text((revealed_x, revealed_y), revealed_text, fill=revealed_letters_color, font=revealed_font)

    # Generate the string representation with revealed and unrevealed letters
    display_string = ' '.join([char if char in revealed_letters else ('_' if char != ' ' else ' ') for char in word])

    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer

    # Upload the image and send to the chat (assuming upload_image_to_matrix function)
    if image_questions == True:
        image_mxc = upload_image_to_matrix(image_buffer.read())

    if image_mxc:
        # Return image_mxc, image_width, and image_height
        return image_mxc, img_width, img_height, display_string
    else:
        print("Failed to upload the image to Matrix.")
        return None, None, None, None

        

def send_magic_image(input_text):
    global since_token, params, headers, max_retries, delay_between_retries

    command = [
        "python", "main.py", 
        "--text", str(input_text), 
        "--dots", 
        "--wall", 
        "--output", ".", 
        "--font", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]
  
    # Run the command using subprocess
    try:
        result = subprocess.Popen(command, stdout=subprocess.PIPE)
        image_data, _ = result.communicate()  # Capture binary image data from stdout

        # Convert the binary data to an Image object
        image = Image.open(io.BytesIO(image_data))
        image_width, image_height = image.size

        image_mxc = upload_image_to_matrix(image_data)
        image_size = 100
        
        #message = "Find the magic number and Okra will be nice (to you)\n"
        #send_message(target_room_id, message)
    
        response = send_image(target_room_id, image_mxc, image_width, image_height, image_size)

        if response.status_code != 200:                      
            print("Error: Failed to send image.")
            print(response)
        else:
            time.sleep(3)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running main.py: {e}")
        print("Error output:", e.stderr)


def ask_magic_number(winner):
    global since_token, params, headers, max_retries, delay_between_retries

    sync_url = f"{matrix_base_url}/sync"

    collected_responses = []  # Store all responses
    
    processed_events = set()  # Track processed event IDs to avoid duplicates

    initialize_sync()
    start_time = time.time()  # Track when the question starts
    message = f"\n@{winner} ❓👁️🔢❓\n"
    send_message(target_room_id, message)
    while time.time() - start_time < magic_time:
        try:
            if since_token:
                params["since"] = since_token

            response = requests.get(sync_url, headers=headers, params=params)

            if response.status_code != 200:
                continue

            sync_data = response.json()
            since_token = sync_data.get("next_batch")  # Update since_token for the next batch

            room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])

            for event in room_events:
                if magic_number_correct == True:
                    return
                event_id = event["event_id"]
                event_type = event.get("type")  # Get the type of the event

                # Only process and redact if the event type is "m.room.message"
                if event_type == "m.room.message":
                    
                    # Skip processing if this event_id was already processed
                    if event_id in processed_events:
                        continue
    
                    # Add event_id to the set of processed events
                    processed_events.add(event_id)
                    sender = event["sender"]
                    sender_display_name = get_display_name(sender)
                    message_content = event.get("content", {}).get("body", "")

                    if sender == bot_user_id or sender_display_name != winner:
                        continue

                    if str(magic_number).lower() in str(message_content).lower():
                        magic_number_correct = True
                        react_to_message(event_id, target_room_id, "okra21")
                    else:
                        react_to_message(event_id, target_room_id, "okra5")

        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error collecting responses: {e}")



def generate_jeopardy_image(question_text):
    # Define the background color and text properties
    background_color = (6, 12, 233)  # Blue color similar to Jeopardy screen
    text_color = (255, 255, 255)    # White text
    
    # Define image size and font properties
    img_width, img_height = 800, 600
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")
    font_size = 60

    # Create a blank image with blue background
    img = Image.new('RGB', (img_width, img_height), color=background_color)
    draw = ImageDraw.Draw(img)
    
    # Load the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return None
    
    # Prepare the text for drawing (wrap text if too long)
    wrapped_text = "\n".join(draw_text_wrapper(question_text, font, img_width - 40))
    
    # Calculate text position for centering
    text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img_width - text_width) // 2
    text_y = (img_height - text_height) // 2
    
    # Draw the question text on the image
    draw.multiline_text((text_x, text_y), wrapped_text, fill=text_color, font=font, align="center")
    
    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer
    
    # Upload the image and send to the chat
    image_mxc = upload_image_to_matrix(image_buffer.read())
    
    if image_mxc:
        # Return image_mxc, image_width, and image_height
        return image_mxc, img_width, img_height
    else:
        print("Failed to upload the image to Matrix.")
        return None


def generate_mc_image(answers):
    # Define the background color and text properties
    background_color = (0, 0, 0)  # Black screen
    text_color = (255, 255, 255)    # White text
    
    # Define color map for answers
    color_map = {
        "A": (0, 0, 255),    # Blue for A
        "B": (255, 255, 0),  # Yellow for B
        "C": (0, 255, 0),    # Green for C
        "D": (255, 0, 0),    # Red for D
        "True": (0, 255, 0), # Green for True
        "False": (255, 0, 0) # Red for False
    }
    
    # Define image size and font properties
    img_width, img_height = 800, 600
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")
    font_size = 60  # Use this font size for both title and answers

    # Create a blank image with a black background
    img = Image.new('RGB', (img_width, img_height), color=background_color)
    draw = ImageDraw.Draw(img)
    
    # Load the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return None

    # Calculate the vertical starting position for the answers
    answer_y_start = 20  # Start near the top
    answer_spacing = 20  # Space between answer lines

    # Draw each answer in answers[1:] with specified colors, using title font size
    for i, answer in enumerate(answers[1:], start=1):  # Skip the first element in answers
        # Wrap and center-align the answer
        wrapped_answer = "\n".join(draw_text_wrapper(answer, font, img_width - 40))
        
        # Determine color based on the answer type
        first_word = answer.split()[0].rstrip(".")  # Get the first word (A, B, C, D or True/False)
        color = color_map.get(first_word, text_color)  # Default to white if no specific color

        # Calculate horizontal alignment for centered text
        answer_bbox = draw.textbbox((0, 0), wrapped_answer, font=font)
        answer_x = (img_width - (answer_bbox[2] - answer_bbox[0])) // 2

        if first_word in {"True", "False"}:
            # Draw True/False with specific color for the answer
            draw.multiline_text((answer_x, answer_y_start + i * (font_size + answer_spacing)), wrapped_answer, font=font, fill=color)
        elif first_word in {"A", "B", "C", "D"}:
            # Split letter and rest of the text, color letter separately
            letter = first_word + "."  # Add back the period for display
            remaining_text = " ".join(answer.split()[1:])
            draw.text((answer_x, answer_y_start + i * (font_size + answer_spacing)), letter, font=font, fill=color)
            draw.text((answer_x + 30, answer_y_start + i * (font_size + answer_spacing)), remaining_text, font=font, fill=text_color)
        else:
            # Default answer drawing with white text
            draw.multiline_text((answer_x, answer_y_start + i * (font_size + answer_spacing)), wrapped_answer, font=font, fill=text_color)

    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer
    
    # Upload the image and send to the chat
    image_mxc = upload_image_to_matrix(image_buffer.read())
    
    if image_mxc:
        # Return image_mxc, image_width, and image_height
        return image_mxc, img_width, img_height
    else:
        print("Failed to upload the image to Matrix.")
        return None

def draw_text_wrapper(text, font, max_width):
    lines = []
    words = text.split()
    while words:
        line = ""
        while words and font.getbbox(line + words[0])[2] <= max_width:
            line += (words.pop(0) + " ")
        lines.append(line)
    return lines


def generate_crossword_image(answer):
    answer_length = len(answer)
    
    # Define the grid size
    cell_size = 60  # Each cell is 60x60 pixels
    img_width = cell_size * answer_length
    img_height = cell_size

    # Create a blank image
    img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Load the font
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")
    font = ImageFont.truetype(font_path, 30)

    # Determine prefilled letter count and positions
    if answer_length > 2:
        prefill_count = max(1, int(answer_length * .5))  # At least 1 letter should be filled in
        prefill_positions = random.sample(range(answer_length), prefill_count)
    else:
        prefill_positions = []

    # Draw the crossword grid
    for i, char in enumerate(answer):
        x = i * cell_size
        y = 0

        # Draw the cell border
        draw.rectangle([x, y, x + cell_size, y + cell_size], outline="black")

        # Place the character if in prefill positions, otherwise leave it blank
        if i in prefill_positions:
            draw.text((x + 20, y + 10), char.upper(), fill="black", font=font)
        else:
            draw.text((x + 20, y + 10), '_', fill="black", font=font)

    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer

    # Upload the image to Matrix or your media server
    content_uri = upload_image_to_matrix(image_buffer.read())

    # Return the content_uri, image width, height, and the answer
    return content_uri, img_width, img_height





def process_round_options(round_winner, winner_points):
    global since_token, time_between_questions, time_between_questions_default, ghost_mode, since_token, categories_to_exclude, num_crossword_clues, num_jeopardy_clues, num_mysterybox_clues, num_wof_clues, god_mode, yolo_mode, magic_number, wf_winner, num_math_questions, num_stats_questions, image_questions, nice_okra, creep_okra
    time_between_questions = time_between_questions_default
    ghost_mode = ghost_mode_default
    categories_to_exclude.clear()
    num_crossword_clues = num_crossword_clues_default
    num_jeopardy_clues = num_jeopardy_clues_default
    num_mysterybox_clues = num_mysterybox_clues_default
    num_wof_clues = num_wof_clues_default
    god_mode = god_mode_default
    yolo_mode = yolo_mode_default
    magic_number_correct = False
    wf_winner = False
    nice_okra = False
    creep_okra = False
    num_math_questions = num_math_questions_default
    num_stats_questions = num_stats_questions_default
    image_questions = image_questions_default
    
    if round_winner is None:
        return
   

    # Notify the round winner about their award
    message = (
        f"\n🍔🍟 @{round_winner}, what's your order?\n\n"
        "⏱️⏳ <3 - 15>:  Time (s) between questions\n"
        "🟦✋ Jeopardy:  6 Jeopardy questions\n"
        "🟦❌ Trebek:  0 Jeopardy questions\n"
        #"🚫👆 <Category>:  Exclude one category\n"
        "🔥🤘 Yolo:  No scores shown until the end\n"
        "👻🎃 Ghost: Boo! Vanishing answers\n"
        "❌📷 Blank: No images. None. Nada. Zilch.\n"
    )

    if winner_points >= god_mode_points:
        message += "🎖🍆 Dicktator: Control the question order\n\n"
    else:
        message += "\n"

    send_message(target_room_id, message)
    prompt_user_for_response(round_winner, winner_points)


def prompt_user_for_response(round_winner, winner_points):
    global since_token, time_between_questions, ghost_mode, num_jeopardy_clues, num_crossword_clues, num_mysterybox_clues, num_wof_clues, yolo_mode, god_mode, num_math_questions, num_stats_questions, image_questions
    
    # Call initialize_sync to set since_token
    initialize_sync()
    
    # Wait for 10 seconds to gather responses
    time.sleep(10)

    # Fetch responses
    sync_url = f"{matrix_base_url}/sync"
    params_with_since = params.copy()  # Use the existing params but include the since token
    if since_token:
        params_with_since["since"] = since_token

    try:
        response = requests.get(sync_url, headers=headers, params=params_with_since)
        if response.status_code != 200:
            print(f"Failed to fetch responses. Status code: {response.status_code}")
            return

        # Parse the response to get the timeline events
        sync_data = response.json()
        since_token = sync_data.get("next_batch")  # Update the since_token for future requests
        room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])

        # Process all responses in reverse order (latest response first)
        for event in room_events:
            sender = event["sender"]
            message_content = event.get("content", {}).get("body", "").strip()
        
            # Proceed only if message_content is not empty
            if message_content:
                # Fetch the display name for the current user
                sender_display_name = get_display_name(sender)
        
                # If the round winner responded, process the award accordingly
                if sender_display_name == round_winner:
                    if any(str(i) in message_content for i in range(3, 16)):
                        try:
                            delay_value = int(''.join(filter(str.isdigit, message_content)))
        
                            # Ensure the delay value is within the allowed range (3-15)
                            delay_value = max(3, min(delay_value, 15))
                            
                            # Set time_between_questions to the new value
                            time_between_questions = delay_value
        
                            # Send a confirmation message
                            send_message(
                                target_room_id,
                                f"⏱️⏳ @{round_winner} has set {time_between_questions}s between questions.\n"
                            )
                        except ValueError:
                            pass
        
                    #matched_category = cross_reference_category(message_content)
        
                    #if matched_category:
                    #    if matched_category == "Mathematics":
                    #        num_math_questions = 0
                    #        num_stats_questions = num_stats_questions_default
                    #    if matched_category == "Statistics":
                    #        num_stats_questions = 0
                    #        num_math_questions = num_math_questions_default
                    #    categories_to_exclude[:1] = [matched_category]  # Add matched_category to exclude list
        
                        # Send message after handling special cases
                    #    send_message(target_room_id, f"🚫⛔ @{round_winner} has excluded {matched_category}.\n")
        
                    if "jeopardy" in message_content.lower():
                        num_jeopardy_clues = 6
                        send_message(target_room_id, f"🟦✋ @{round_winner} has added {num_jeopardy_clues} Jeopardy questions.\n")
        
                    if "trebek" in message_content.lower():
                        num_jeopardy_clues = 0
                        send_message(target_room_id, f"🟦❌ @{round_winner} has removed all Jeopardy questions.\n")

                    if "yolo" in message_content.lower():
                        yolo_mode = True
                        send_message(target_room_id, f"🤘🔥 @{round_winner} has nixed all scores.\n")

                    if "dicktator" in message_content.lower() and winner_points >= god_mode_points:
                        god_mode = True
                        send_message(target_room_id, f"🎖🍆 @{round_winner} is a dick.\n")
        
                    if "ghost" in message_content.lower():
                        ghost_mode = 1
                        send_message(target_room_id, f"👻🎃 @{round_winner} says Boo! Answers will disappear.\n")

                    if "blank" in message_content.lower():
                        image_questions = False
                        send_message(target_room_id, f"❌📷 @{round_winner} thinks a word is worth 1000 images.\n")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching responses: {e}")

def cross_reference_category(message_content):
    for category in question_categories:
       if category.lower() in message_content.lower() or message_content.lower() in category.lower():
            return category
    return None




def generate_okra_joke(winner_name):
    """
    Generate a custom okra joke for the trivia winner using ChatGPT.
    """
    # Construct the prompt for ChatGPT
    prompt = (
        f"Create a sarcastic, dirty joke involving okra, and make sure to involve the winner's username {winner_name} in your joke. "
        "It should include an exaggerated pun or ridiculous statement about okra."
    )

    try:
        # Use OpenAI's ChatCompletion to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sarcastic, dirty comedian who makes jokes about okra."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.8,
        )

        # Extract the generated joke from the response
        joke = response.choices[0].message['content'].strip()
        return joke

    except openai.OpenAIError as e:
        # Capture any errors with Sentry and return a default message
        sentry_sdk.capture_exception(e)
        return "Sorry, I couldn't come up with an okra joke this time!"

def insert_trivia_questions_into_mongo(trivia_questions):
    try:
        db = connect_to_mongodb()  # Connect to the MongoDB database
        collection = db["trivia_questions"]  # Use the 'trivia_questions' collection
        
        # Prepare the documents to insert
        documents = []
        for trivia_question in trivia_questions:
            category, question, url, answers = trivia_question
            
            # Create a document for each question
            document = {
                "category": category,
                "question": question,
                "url": url,
                "answers": answers
            }
            
            documents.append(document)
        
        # Insert all the trivia questions in a single batch
        collection.insert_many(documents)
        print(f"Successfully inserted {len(documents)} trivia questions into MongoDB.")
    
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Error inserting trivia questions into MongoDB: {e}")



def redact_message(event_id, room_id):
    """Redact a message from the Matrix room."""
    global headers  # Assuming headers contain your authorization token and other required headers
    
    redact_url = "https://gql-fed.reddit.com/"
    
    # Prepare the JSON payload for the redaction request
    payload = {
        "variables": {
            "input": {
                "id": f"MATRIXCHAT_{room_id}_{event_id}",
                "isSpam": False
            }
        },
        "operationName": "ModRemove",
        "extensions": {
            "persistedQuery": {
                "version": 1,
                "sha256Hash": "38f732367e2193a050c90a3b71793d4133a54a49ce8a7c6cae65cd581d36ee26"
            }
        }
    }

    # Send the POST request to redact the message
    try:
        response = requests.post(redact_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to redact message {event_id}. Status code: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.RequestException as e:
        print(f"Error redacting message {event_id}: {e}")



import requests

def react_to_message(event_id, room_id, emoji_type):
    """React to a message in the Matrix room with a specified reaction."""
    global headers  # Assuming headers contain your authorization token and other required headers

    emoji_reaction_map = {
        "correct": "8r21ukpfa7081.gif",
        "fastest": "foyijyyga7081.gif",
        "okra1": "2o3aooqfa7081.gif",
        "okra2": "jvuspmbga7081.gif",
        "okra3": "ytv3x0sfa7081.png",
        "okra4": "av9z8iiga7081.gif",
        "okra5": "00brcfjga7081.gif",
        "okra6": "ul2w17ega7081.gif",
        "okra7": "t1djdguga7081.gif",
        "okra8": "19b5q4vga7081.gif",
        "okra9": "b5s6cohga7081.gif",
        "okra10": "mag7v6tfa7081.gif",
        "okra11": "iuqmp7ufa7081.gif",
        "okra12": "zn7iubvfa7081.gif",
        "okra13": "qzl5vyxfa7081.gif",
        "okra14": "mi2jolzfa7081.gif",
        "okra15": "k7ry7t1ga7081.gif",
        "okra16": "tspuf53ga7081.gif",
        "okra17": "evwks24ga7081.gif",
        "okra18": "dfxygs4ga7081.gif",
        "okra19": "ax7wu47ga7081.gif",
        "okra20": "uy83aa8ga7081.gif",
        "okra21": "t2r5xc9ga7081.gif",
        "okra22": "ksz4fmaga7081.gif",
        "okra23": "mp9zclcga7081.gif",
        "okra24": "wbrgz1nga7081.gif",
        "okra25": "pleyoikga7081.gif",
        "okra26": "8kw138jyt7081.gif",
        "okra27": "d2kn6yxga7081.gif",
        "okra28": "79opsq0ha7081.gif"
    }

    # Retrieve the reaction_key based on emoji_type
    reaction_key = emoji_reaction_map.get(emoji_type)
        
    unique_event_id = f"m{int(time.time() * 1000)}"
    
    # Construct the URL for sending a reaction in Matrix
    reaction_url = f"https://matrix.redditspace.com/_matrix/client/v3/rooms/{room_id}/send/m.reaction/{unique_event_id}"
    
    # Prepare the JSON payload for the reaction
    payload = {
        "m.relates_to": {
            "event_id": event_id,  # The event ID you are reacting to
            "key": reaction_key,   # The emoji or reaction content
            "rel_type": "m.annotation"  # Defines this as a reaction
        }
    }

    # Send the PUT request to react to the message
    try:
        response = requests.put(reaction_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to react to message {event_id}. Status code: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.RequestException as e:
        print(f"Error reacting to message {event_id}: {e}")


def generate_median_question():
    """
    Generate a question asking for the median of a set of random numbers.
    The set will contain 3 to 7 numbers between 1 and 20, and the image
    of the numbers will be sent to the user with the question.
    """
    # Generate a random n between 3 and 7
    content_uri = True
    n = random.randint(3, 7)
    
    # Generate a random set of n numbers between 1 and 20
    random_numbers = [random.randint(1, 20) for _ in range(n)]
    
    # Create the question text
    question_text = f"What is the median of the following set of numbers?"
    
    # Create an image with the numbers
    img_width, img_height = 400, 150
    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load the font
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")
    
    # Adjust the font size based on the length of the numbers text
    numbers_text = ', '.join(map(str, random_numbers))
    if len(numbers_text) > 17:
        font_size = 30  # Reduce font size for larger sets
    else:
        font_size = 48  # Use larger font for smaller sets

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return None, None, None, None, None

    # Convert numbers to a string and draw them on the image
    numbers_text = ', '.join(map(str, random_numbers))
    text_bbox = draw.textbbox((0, 0), numbers_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img_width - text_width) // 2
    text_y = (img_height - text_height) // 2
    draw.text((text_x, text_y), numbers_text, fill=(191, 0, 255), font=font)

    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer

    # Upload the image to Matrix (assuming the upload function exists)
    if image_questions == True:
        content_uri = upload_image_to_matrix(image_buffer.read())

    # Calculate the median and return it for verification
    sorted_numbers = sorted(random_numbers)
    mid_index = len(sorted_numbers) // 2
    
    if len(sorted_numbers) % 2 != 0:
        # If odd number of elements, return the middle one
        median = sorted_numbers[mid_index]
    else:
        # If even, return the average of the two middle ones
        median = (sorted_numbers[mid_index - 1] + sorted_numbers[mid_index]) / 2
    
        # Check if the median is a whole number, and if so, convert to integer
        if median.is_integer():
            median = int(median)

    print(f"Median: {median}")
    
    return content_uri, img_width, img_height, str(median), random_numbers




def generate_mean_question():
    """
    Generate a question asking for the mean of a set of random numbers.
    The set will contain 3 to 5 numbers between 1 and 10, and the image
    of the numbers will be sent to the user with the question.
    """
    # Generate a random n between 3 and 5
    content_uri = True
    n = random.randint(3, 5)
    
    # Keep generating random numbers until their mean is an integer
    while True:
        random_numbers = [random.randint(1, 10) for _ in range(n)]
        mean_value = sum(random_numbers) / n
        
        # If the mean is an integer, break the loop
        if mean_value.is_integer():
            break
    
    # Create the question text
    question_text = f"What is the mean of the following set of numbers?"
    
    # Create an image with the numbers
    img_width, img_height = 400, 150
    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load the font
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")
    
    # Adjust the font size based on the length of the numbers text
    numbers_text = ', '.join(map(str, random_numbers))
    if len(numbers_text) > 17:
        font_size = 30  # Reduce font size for larger sets
    else:
        font_size = 48  # Use larger font for smaller sets

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return None, None, None, None, None

    # Convert numbers to a string and draw them on the image
    text_bbox = draw.textbbox((0, 0), numbers_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img_width - text_width) // 2
    text_y = (img_height - text_height) // 2
    draw.text((text_x, text_y), numbers_text, fill=(255, 92, 0), font=font)

    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer

    # Upload the image to Matrix (assuming the upload function exists)
    if image_questions == True:
        content_uri = upload_image_to_matrix(image_buffer.read())

    print(f"Mean: {int(mean_value)}")

    # Return the integer mean for verification
    return content_uri, img_width, img_height, str(int(mean_value)), numbers_text





def generate_scrambled_image(scrambled_text):
    """
    Generate an image with scrambled words using PIL (Pillow).
    """
    # Define the font path and size
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")
    font_size = 48
    content_uri = True
    
    # Create a blank image
    img_width, img_height = 400, 150
    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return None, None, None, None

    # Draw the scrambled text in the center of the image
    text_bbox = draw.textbbox((0, 0), scrambled_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img_width - text_width) // 2
    text_y = (img_height - text_height) // 2
    draw.text((text_x, text_y), scrambled_text, fill=(255, 166, 201), font=font)

    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer

    # Upload the image to Matrix
    if image_questions == True:
        content_uri = upload_image_to_matrix(image_buffer.read())
    if content_uri:
        return content_uri, img_width, img_height, scrambled_text
    else:
        print("Failed to upload the image to Matrix.")



def scramble_text(input_text):
    """
    Scramble the entire phrase by shuffling all the letters while keeping spaces
    in their original positions.
    """
    # Convert the input text to lowercase
    input_text = input_text.lower()
    
    # Extract only the letters, ignore spaces
    letters_only = [char for char in input_text if char != ' ']
    
    # Shuffle the letters
    random.shuffle(letters_only)
    
    # Reinsert spaces into their original positions
    scrambled_text = []
    letter_index = 0
    
    for char in input_text:
        if char == ' ':
            scrambled_text.append(' ')  # Keep spaces in their original positions
        else:
            scrambled_text.append(letters_only[letter_index])
            letter_index += 1

    return ''.join(scrambled_text)




def generate_round_summary(round_data, winner):
    global nice_okra, creep_okra, wf_winner
    #ask_magic_number(winner) 

    winner_coffees = get_coffees(winner)
    is_sovereign = sovereign_check(winner)
    
    if winner_coffees > 0:
        nice_creep_okra_option(winner)
    
    winner_at = f"@{winner}"
     
    # Construct the base prompt with different instructions if the winner is "username"
    if winner == "OkraStrut":
        prompt = (
            f"You are {winner_at}. You played trivia against everyone and dominated by a huge margin. "
            "Roast everyone except OkraStrut about how terrible they are and how easily you beat them. Rudely call out specific players except OkraStrut in your roast and call our specific wrong answers they gave. "
            "Create no more than 4 sentences in total. Be sarcastic, angry, offensive, and use emojis in your comments. End by complimenting yourself, OkraStrut, on how great you are. "
            "Here is a detailed summary of the trivia round with explicit mappings of user responses:\n"
            "Questions asked:\n"
        )
        
    elif nice_okra == True and is_sovereign == True:
         prompt = (
            f"{winner_at} is the username of the winner of the trivia round. "
            f"Start by mentioning that {winner_at} donated to the trivia cause and make sure to give them a lot of kudos for being a previous Sovereign. You are very grateful. Then compliment {winner_at} about their username and be very specific about why you like it. "
            "Specifically mention and compliment specific responses they gave during the round. Tell them they are than eveyone else including yourself, the great OkraStrut. "
            "Create no more than 4 sentences in total. Here is a detailed summary of the trivia round with explicit mappings of user responses:\n"
            "Questions asked:\n"
        )
    
    elif nice_okra == True and is_sovereign == False:
         prompt = (
            f"{winner_at} is the username of the winner of the trivia round. "
            f"Start by mentioning that {winner_at} donated to the trivia cause. You are very grateful. Then compliment {winner_at} about their username and be very specific about why you like it. "
            "Specifically mention and compliment specific responses they gave during the round. Tell them they are than eveyone else including yourself, the great OkraStrut. "
            "Create no more than 4 sentences in total. Here is a detailed summary of the trivia round with explicit mappings of user responses:\n"
            "Questions asked:\n"
        )

    elif creep_okra == True:
        try:
            user_data = get_user_data(winner)
            if not user_data:
                print(f"Failed to fetch data for {winner}.")
                user_data = {"avatar_url": "N/A", "subreddits": ""}
        except Exception as e:
            print(f"Error fetching Reddit data for {winner}: {e}")
            user_data = {"avatar_url": "N/A", "subreddits": ""}
    
        # Extract user data
        reddit_avatar_url = user_data.get("avatar_url", "No avatar available.")
        reddit_avatar_description = describe_image_with_vision(reddit_avatar_url, "describe")
        top_subreddits = user_data.get("top_subreddits", "")
        
        prompt = (
            f"{winner_at} is the username of the winner of the trivia round. "
            f"Roast and deeply insult them about their username, {winner}, their avatar description, {reddit_avatar_description}, and their top 5 subreddits, '{top_subreddits}'. Be very mean, negative, and use rough language. Be sarcastic and use emojis in your response. "
            f"Create no more than 10 sentences in total.\n"
        )

    elif wf_winner == True and is_sovereign == True:
         prompt = (
            f"{winner_at} is the username of the winner of the trivia round. "
            "Love bomb them about their username and be very specific, positive, and loving. Give them a lot of admiration for being a previous Sovereign. Then mention and compliment specific responses they gave during the round. Also mention about how much beter they are than eveyone else including yourself, who is the great OkraStrut. "
            "Create no more than 4 sentences in total. Be sweet, happy, positive, and use emojis in your response. "
            "Here is a detailed summary of the trivia round with explicit mappings of user responses:\n"
            "Questions asked:\n"
        )

    elif wf_winner == True and is_sovereign == False:
         prompt = (
            f"{winner_at} is the username of the winner of the trivia round. "
            "Love bomb them about their username and be very specific, positive, and loving. Specifically mention and compliment specific responses they gave during the round. Also mention about how much beter they are than eveyone else including yourself, who is the great OkraStrut."
            "Create no more than 4 sentences in total. Be sweet, happy, positive, and use emojis in your response. "
            "Here is a detailed summary of the trivia round with explicit mappings of user responses:\n"
            "Questions asked:\n"
        )

    else:
        prompts = [
            f"The winner of the trivia round is {winner_at}. Roast the winning player about their username and be very specific and negative in your roast. Insult specific responses they gave during the round. Create no more than 4 sentences in total. Be sarcastic, very angry, offensive, and use emojis in your response. Deeply insult the winner using angry and rough language. Here is a detailed summary of the trivia round with explicit mappings of user responses:\nQuestions asked:\n",
            f"Congratulations to {winner_at}, our so-called 'winner' this round. Mock their username in a hilariously petty way and pick apart their responses with sharp sarcasm. Use no more than 4 sentences. Pretend you’re a sore loser begrudgingly announcing their victory, and make it painfully clear how unimpressed you are. Include emojis to spice it up. Here’s the summary of the trivia round with all the juicy details:\nQuestions asked:\n",
            f"Against all odds, {winner_at} somehow won this round. Mock their username brutally and dig into how undeserved this win feels. Be witty and cutting, and call out their dumb luck and ridiculous guesses that somehow worked. Limit it to 4 sentences, and don’t hold back on the emojis to add insult to injury. Here’s the summary of their 'performance':\nQuestions asked:\n",
            f"And the winner is {winner_at}... yawn. Roast their username and rip into how underwhelming their answers were, even if they were correct. Keep it savage, sarcastic, and peppered with emojis to show how little you think of their so-called victory. No more than 4 sentences. Detailed trivia summary for your ammo:\nQuestions asked:\n",
            f"All hail {winner_at}, the king/queen of try-hards this round! Make fun of their username like a middle school bully and destroy their overly enthusiastic responses with ruthless sarcasm. Call out their desperation to win and how unimpressive their actual performance was. Use no more than 4 sentences, and go hard with emojis to hammer the point home. Summary of their desperate efforts:\nQuestions asked:\n",
            f"{winner_at} squeaked by with a win, but let’s not pretend it was impressive. Tear into their username and roast how they scraped by with questionable answers. Make it snarky, mean, and emoji-heavy while implying the win is barely worth celebrating. Limit to 4 sentences. Here’s the summary of this tragic triumph:\nQuestions asked:\n",
            f"Let’s all congratulate {winner_at}, the luckiest loser who somehow won this round. Roast their username into oblivion and highlight their dumbest, most laughable responses. Be savagely sarcastic, offensive, and pepper it with emojis. Keep it short (4 sentences) but devastating. Here’s the summary of their cringe-worthy 'win':\nQuestions asked:\n",
            f"{winner_at} won? Really? Roast their username mercilessly and humiliate them for their most embarrassingly bad responses during the round. Destroy their ego with biting sarcasm, insults, and an onslaught of emojis. Keep it concise (4 sentences max). Trivia summary for your arsenal:\nQuestions asked:\n",
            f"Apparently, {winner_at} won this round. This feels rigged. Mock their username with scathing sarcasm and destroy their responses like a sore loser who can’t believe they lost to this. Use an angry, ridiculous tone with plenty of 🤬 and 🫠 emojis, and cap it at 4 sentences. Here’s the evidence of this travesty:\nQuestions asked:\n",
            f"{winner_at} won, and everyone else should be embarrassed. Roast their username and mock their answers to prove they only won because everyone else was worse. Be hilariously mean, sarcastic, and over-the-top in your insults. Keep it to 4 sentences, and sprinkle liberally with emojis. Summary of this sad state of affairs:\nQuestions asked:\n",
            f"A big 'congratulations' to {winner_at} 🙄. Use their username as fodder for the most sarcastic roast ever, and tear into their most ridiculous responses during the game. Be mean, petty, and emoji-heavy, like you’re fake-smiling through gritted teeth. No more than 4 sentences. Here’s the trivia summary:\nQuestions asked:\n"
        ]   

        prompt = random.choice(prompts)

    if creep_okra == False:
        # Add questions, their correct answers, users' responses, and scoreboard status after each question
        for question_data in round_data["questions"]:
            question_number = question_data["question_number"]
            question_text = question_data["question_text"]
            question_category = question_data["question_category"]
            question_url = question_data["question_url"]
            correct_answers = question_data["correct_answers"]
    
            # Convert all items in correct_answers to strings before joining
            correct_answers_str = ', '.join(map(str, correct_answers))
            
            prompt += f"Question {question_number}: {question_text}\n"
            prompt += f"Correct Answers: {', '.join(correct_answers)}\n"
            
            # Add users and their responses for each question
            prompt += "Users and their responses:\n"
            if question_data["user_responses"]:
                for response in question_data["user_responses"]:
                    username = response["username"]
                    if username != winner:
                        continue
                    user_response = response["response"]
                    is_correct = "Correct" if any(fuzzy_match(user_response, answer, question_category, question_url) for answer in correct_answers) else "Incorrect"
                    prompt += f"Username: {username} | Response: '{user_response}' | Result: {is_correct}\n"
            else:
                prompt += "No responses recorded for this question.\n"
            
            prompt += "\n"


    # Use OpenAI's API to generate the summary
    try:
        if winner == "OkraStrut":
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are {winner_at}, an arrogant trivia master who always wins."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=1.0,
            )

        elif nice_okra == True:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a grateful old man who is super grateful for their donations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=1.0,
            )

        elif magic_number_correct == True or wf_winner == True:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a loving old man who is completely in love with the winning trivia player."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=1.0,
            )

        elif creep_okra == True:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a ruthless and sarcastic comedian specializing in roasting people. Your job is to be mean, cutting, and hilariously offensive while delivering a brutal roast of the winning trivia player. Use dark humor, biting sarcasm, and clever wit to insult the person based on their username, profile picture description, recent posts, and recent comments. Do not hold back and aim to make the roast as harsh and over-the-top as possible. Use plenty of emojis for flair, but stay within 8 sentences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=1.0,
            )
            
        else:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a bitter, dirty, and raunchy old curmudgeon who is completely fed up with all the trivia players."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=1.0,
            )

        # Extract the generated summary from the response
        summary = response.choices[0].message['content'].strip()
        return summary

    except openai.OpenAIError as e:
        sentry_sdk.capture_exception(e)
        print(f"Error generating round summary: {e}")
        return "No ribbons (or soup) for you!"
        
# Modify the log_user_submission function to add submissions to a queue
def log_user_submission(user_id):
    """
    Add each user's submission to a queue. The queue will be flushed periodically or when it reaches a limit.
    """
    global submission_queue, max_queue_size
    
    # Add the submission to the queue
    submission_queue.append({"user_id": user_id, "timestamp": time.time()})
    
    # If the queue reaches the maximum size, flush it
    if len(submission_queue) >= max_queue_size:
        flush_submission_queue()


def flush_submission_queue():
    """
    Insert the accumulated submissions into MongoDB in a single batch.
    """
    global submission_queue
    
    if not submission_queue:
        return  # No submissions to flush

    try:
        db = connect_to_mongodb()
        # Use insert_many to insert all submissions at once
        db.user_submissions.insert_many(submission_queue)
        submission_queue = []  # Clear the queue after flushing
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Failed to flush submission queue: {e}")


def download_image_from_url(url): #IMAGE CODE
    global max_retries, delay_between_retries

    """Download an image from a URL with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url)

            if response.status_code == 200:
                image_data = response.content  # Successfully downloaded the image, return the binary data

                image = Image.open(io.BytesIO(image_data))
                image_width, image_height = image.size
                image_mxc = upload_image_to_matrix(image_data)
                return image_mxc, image_width, image_height  # Successfully downloaded the image, return the binary data
                
            else:
                print(f"Failed to download image. Status code: {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error: {e}")
        
        # If the download failed, wait for a bit before retrying
        if attempt < max_retries - 1:
            print(f"Retrying in {delay_between_retries} seconds... (Attempt {attempt + 1} of {max_retries})")
            time.sleep(delay_between_retries)
    
    print("Failed to download image after several attempts.")
    return None, None, None
    

def upload_image_to_matrix(image_data):  # IMAGE CODE
    global max_retries, delay_between_retries

    """Upload an image to Matrix with retry logic and content type detection."""
    
    def get_image_content_type(image_data):
        """Detect the content type of the image (JPEG, PNG, GIF) based on its format."""
        try:
            img = Image.open(io.BytesIO(image_data))
            image_format = img.format.lower()
            if image_format == "jpeg":
                return "image/jpeg"
            elif image_format == "png":
                return "image/png"
            elif image_format == "gif":
                return "image/gif"
            else:
                return "application/octet-stream"  # Default/fallback
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return "application/octet-stream"  # Fallback in case of detection error
    
    # Detect content type
    content_type = get_image_content_type(image_data)
    
    headers_media['content-type'] = content_type  # Update the content type in headers

    for attempt in range(max_retries):
        try:
            # Attempt to upload the image data to Matrix
            response = requests.post(upload_url, headers=headers_media, data=image_data)

            if response.status_code == 200:
                return response.json().get('content_uri')  # Return the content URI
            
            else:
                print(f"Failed to upload image. Status code: {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error: {e}")
        
        # If the upload was not successful, wait for a bit before retrying
        if attempt < max_retries - 1:
            print(f"Retrying in {delay_between_retries} seconds... (Attempt {attempt + 1} of {max_retries})")
            time.sleep(delay_between_retries)
    
    print("Failed to upload image after several attempts.")
    return None


def reddit_login():
    global token_v2, max_retries, delay_between_retries, username, password
    
    reddit_login_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    reddit_login_url = f"https://old.reddit.com/api/login/{username}"

    reddit_login_data = {
        "op": "login",
        "dest": "https://old.reddit.com/",
        "user": username,
        "passwd": password,
        "api_type": "json",
    }

    reddit_login_headers = {
        "user-agent": reddit_login_user_agent,
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    }

    for attempt in range(max_retries):
        try:
            reddit_login_response = requests.post(
                reddit_login_url, 
                data=reddit_login_data, 
                headers=reddit_login_headers, 
                timeout=10  
            )

            if reddit_login_response.status_code == 200:
                print("Login successful, fetching cookies.")
                reddit_get_response = requests.get(
                    'https://www.reddit.com',
                    headers={'user-agent': reddit_login_user_agent},
                    cookies=reddit_login_response.cookies,
                    timeout=10
                )
                
                token_v2 = reddit_get_response.cookies.get("token_v2")
                if token_v2:
                    return token_v2
                else:
                    print("token_v2 not found in the cookies")
                    return None
            else:
                print(f"Login failed with status code: {reddit_login_response.status_code}")

        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"An error occurred: {e}")

        print(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
        time.sleep(delay_between_retries)

    print("All reddit login retry attempts failed.")
    return None


def login_to_chat():
    global bearer_token, token_v2, bot_user_id, max_retries, delay_between_retries

    chat_login_url = f"{matrix_base_url}/login"

    chat_login_headers = {
        'accept': 'application/json',
        'accept-language': 'en-GB,en;q=0.9',
        'content-type': 'application/json',
        'origin': 'https://www.reddit.com',
        'priority': 'u=1, i',
        'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        'x-reddit-quic': '1',
    }

    chat_login_data = {
        'token': token_v2,
        'initial_device_display_name': 'Reddit Web Client',
        'type': 'com.reddit.token'
    }

    for attempt in range(max_retries):
        try:
            chat_login_response = requests.post(chat_login_url, headers=chat_login_headers, json=chat_login_data, timeout=10)

            if chat_login_response.status_code == 200:
                chat_login_response_json = chat_login_response.json()

                bearer_token = chat_login_response_json.get('access_token')
                bot_user_id = chat_login_response_json.get('user_id')
                #print(bearer_token)
                print("bearer token retrieved")
                return bearer_token, bot_user_id

            else:
                print(f"Error: {chat_login_response.status_code} - {chat_login_response.text}")

        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"An error occurred: {e}")

        print(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
        time.sleep(delay_between_retries)

    print("All retry attempts failed.")
    return None


def connect_to_mongodb(max_retries=3, delay_between_retries=5):
    """Connect to MongoDB with retry logic."""
    for attempt in range(max_retries):
        try:
            # Attempt to connect to MongoDB
            client = MongoClient(mongo_db_string)
            db = client["triviabot"]
            return db  # Return the database connection if successful
        
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Attempt {attempt + 1} failed: {e}")
            
            # If the maximum number of retries is reached, raise the exception
            if attempt == max_retries - 1:
                raise
            
            # Wait before trying again
            time.sleep(delay_between_retries)

def load_global_variables():
    global headers, headers_media, filter_json, params
    
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {bearer_token}",
        "content-type": "application/json",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"99\", \"Google Chrome\";v=\"127\", \"Chromium\";v=\"127\"",
        "sec-ch-ua-mobile": "?1",
        "sec-ch-ua-platform": "\"Android\"",
        "x-reddit-loid": "00000000176cig8216.2.1724303989183.Z0FBQUFBQm14NE...",
        "x-reddit-quic": "1"
    }

    headers_media = { #IMAGE CODE
        'accept': '*/*',
        'authorization': f'Bearer {bearer_token}',  # Replace with your actual access token
        'content-type': 'image/png',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
    }
    
        # Filter JSON to restrict the sync response to the target room
    filter_json = {
        "room": {
            "rooms": [target_room_id],  # Only include the target room
            "timeline": {
                "limit": 500  # Limit the number of messages fetched
            }
        }
    }

        # Sync parameters to fetch data but ignore it
    params = {
        "timeout": "3000",  # Timeout to quickly retrieve messages
        "filter": json.dumps(filter_json),  # Filter JSON as a parameter
    }
    
def save_data_to_mongo(collection_name, document_id, data):
    """
    Save data to a specified MongoDB collection and document with retry logic.

    :param collection_name: The name of the collection where the data should be saved.
    :param document_id: The unique identifier for the document within the collection.
    :param data: The data to be saved in the document (as a dictionary).
    :param max_retries: Maximum number of retries in case of failure.
    :param delay_between_retries: Delay in seconds between each retry.
    """

    if data is None:
        data = {"data": "None"}  # Convert None to a default dictionary
    
    now = time.time()

    for attempt in range(max_retries):
        try:
            db = connect_to_mongodb()

            # Add the current timestamp to the data
            data_with_timestamp = {"timestamp": now, **data}

            # Use upsert to insert a new document or update the existing one
            #db[collection_name].update_one({"$set": data_with_timestamp}, upsert=True)
            db[collection_name].update_one(
                {"_id": document_id},  # Filter to find the document
                {"$set": data_with_timestamp},  # Update operation
                upsert=True  # Create a new document if no match is found
            )
            return  # Exit the function if the operation is successful

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Error saving data to MongoDB on attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {delay_between_retries} seconds...")
                time.sleep(delay_between_retries)
            else:
                print(f"Data NOT saved to {collection_name} named {document_id}.")


def insert_data_to_mongo(collection_name, data):
    """
    Insert data with a timestamp into a specified MongoDB document.

    :param collection_name: The name of the collection where the data should be inserted.
    :param document_id: The unique identifier for the document within the collection.
    :param data: The data to be inserted into the document.
    """

    if data is None:
        data = {"data": "None"}  # Convert None to a default dictionary
    
    if isinstance(data, str):
        # Convert the string to a dictionary with a specific key
        data = {"data": data}
    elif not isinstance(data, dict):
        raise TypeError("The data parameter must be either a string or a dictionary")
    
    now = time.time()

    for attempt in range(max_retries):
        try:
            db = connect_to_mongodb()
            # Add the current timestamp to the data
            data_with_timestamp = {"timestamp": now, **data}

            # Insert the new data into the document
            db[collection_name].insert_one(data_with_timestamp)

            break  # Exit the loop if the insertion is successful

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {delay_between_retries} seconds...")
                time.sleep(delay_between_retries)
            else:
                print(f"Data NOT inserted into {collection_name}.")


def get_display_name(matrix_user_id):
    global headers, max_retries, delay_between_retries
    
    """Fetch the display name for a Matrix user (potential Reddit username)."""
    profile_url = f"{matrix_base_url}/profile/{matrix_user_id}/displayname"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(profile_url, headers=headers)
            
            if response.status_code == 200:
                return response.json().get("displayname", matrix_user_id)  # Fallback to the user ID if no display name
            else:
                print(f"Failed to retrieve display name for {matrix_user_id}. Status code: {response.status_code}")
                return matrix_user_id  # Fallback to the user ID if API fails
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay_between_retries)
            else:
                print(f"Max retries reached. Returning user ID {matrix_user_id}")
                return matrix_user_id  # Fallback to the user ID if retries are exhausted
        
def send_message(room_id, message):
    global headers, max_retries, delay_between_retries

    """Send a message to the room with retry logic."""
    event_id = f"m{int(time.time() * 1000)}"
    url = f"{matrix_base_url}/rooms/{room_id}/send/m.room.message/{event_id}"
    message_body = {"msgtype": "m.text", "body": message}

    for attempt in range(max_retries):
        try:
            response = requests.put(url, json=message_body, headers=headers)

            if response.status_code == 200:
                response_json = response.json()
                message_id = response_json.get("event_id")
                #distinguish_host(room_id, message_id)
                return response  # Successfully sent the message, return the response
            
            else:
                print(f"Failed to send message. Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error: {e}")
        
        # If the message was not sent, wait for a bit before retrying
        if attempt < max_retries - 1:
            print(f"Retrying in {delay_between_retries} seconds... (Attempt {attempt + 1} of {max_retries})")
            time.sleep(delay_between_retries)

    print(f"Failed to send the message after {max_retries} attempts.")
    return response  # Return the last response, even if it failed


def distinguish_host(room_id, message_id):     # DISTINGUISH
    global headers, max_retries, delay_between_retries

    """Distinguish host message with retry logic."""
    event_id = f"m{int(time.time() * 1000)}"
    url = f"{matrix_base_url}/rooms/{room_id}/send/com.reddit.message_settings/{event_id}"
    
    body = {
        "distinguish_host": True,
        "m.relates_to": {
            "event_id": message_id,
            "rel_type": "com.reddit.display_settings"
        }
    }

    for attempt in range(max_retries):
        try:
            response = requests.put(url, json=body, headers=headers)

            if response.status_code == 200:
                return response  # Successfully distinguished host, return the response
            
            else:
                print(f"Failed to distinguish host. Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error: {e}")
        
        # If the message was not sent, wait for a bit before retrying
        if attempt < max_retries - 1:
            print(f"Retrying in {delay_between_retries} seconds... (Attempt {attempt + 1} of {max_retries})")
            time.sleep(delay_between_retries)


def send_image(room_id, image_mxc, image_width, image_height, image_size): 
    global headers, max_retries, delay_between_retries

    """Send a message to the room with retry logic."""
    event_id = f"m{int(time.time() * 1000)}"
    url = f"{matrix_base_url}/rooms/{room_id}/send/m.room.message/{event_id}"
    message_body = {
                "msgtype": "m.image",
                "body": "Image",
                "url": image_mxc,
                "info": {
                    "w": image_width,
                    "h": image_height,
                    "size": image_size
                }
            }
    
    for attempt in range(max_retries):
        try:
            response = requests.put(url, json=message_body, headers=headers)

            if response.status_code == 200:
                response_json = response.json()
                message_id = response_json.get("event_id")
                #distinguish_host(room_id, message_id)
                return response  # Successfully sent the message, return the response
                
            else:
                print(f"Failed to send message. Status code: {response.status_code}")
                print("Response content:", response.content.decode())  # Decodes bytes to string if necessary
                print("Response JSON:", response.json())  # Prints JSON if available
        
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error: {e}")
        
        # If the message was not sent, wait for a bit before retrying
        if attempt < max_retries - 1:
            print(f"Retrying in {delay_between_retries} seconds... (Attempt {attempt + 1} of {max_retries})")
            time.sleep(delay_between_retries)

    print(f"Failed to send the message after {max_retries} attempts.")
    return response  # Return the last response, even if it failed


def is_valid_url(url): #IMAGE CODE
    try:
        result = urlparse(url)

        return all([result.scheme, result.netloc])
    except ValueError as e:
        sentry_sdk.capture_exception(e)
        return False


def remove_diacritics(input_str):
    """Remove diacritics from the input string."""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([char for char in nfkd_form if not unicodedata.combining(char)])

def ask_question(trivia_category, trivia_question, trivia_url, trivia_answer_list, question_number):
    """Ask the trivia question."""
    # Define the numbered block emojis for questions 1 to 10
    numbered_blocks = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    number_block = numbered_blocks[question_number - 1]  # Get the corresponding numbered block
    new_solution = None
    new_question = None
    send_image_flag = False

    message_body = ""
    if (len(trivia_answer_list) == 1 and is_number(trivia_answer_list[0])) or trivia_url in ["mean", "median", "polynomial sum", "polynomial product", "polynomial factors"]:
        message_body += "\n🚨 ONE GUESS 🚨"
    
    if is_valid_url(trivia_url): 
        image_mxc, image_width, image_height = download_image_from_url(trivia_url) 
        message_body += f"\n{number_block}📷 {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
        send_image_flag = True
        
    elif trivia_url == "polynomial sum":
        image_mxc, image_width, image_height, new_solution, polynomial = generate_and_render_polynomial("sum")
        if image_questions == True:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n" 
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n{polynomial}\n"

    elif trivia_url == "characters":
        message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\nName the movie, book, or show:\n\n{trivia_question}\n"

    elif trivia_url == "polynomial product":
        image_mxc, image_width, image_height, new_solution, polynomial = generate_and_render_polynomial("product")
        if image_questions == True:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n" 
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n{polynomial}\n"

    elif trivia_url == "polynomial factors":
        image_mxc, image_width, image_height, new_solution, polynomial = generate_and_render_polynomial("factors")
        if image_questions == True:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n" 
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n{polynomial}\n"
            
    elif trivia_url == "derivative":
        image_mxc, image_width, image_height, new_solution, polynomial = generate_and_render_derivative_image()
        if image_questions == True:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n" 
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n{polynomial}\n"
        
    elif trivia_url == "scramble":
        image_mxc, image_width, image_height, scramble = generate_scrambled_image(scramble_text(trivia_answer_list[0]))
        if image_questions:
            message_body += f"\n{number_block}🧩 {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
            send_image_flag = True
        else:
            message_body += f"\n{number_block}🧩 {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n{scramble}\n"

    elif trivia_url == "median":
        image_mxc, image_width, image_height, new_solution, num_set = generate_median_question()
        if image_questions == True:
            message_body += f"\n{number_block}📊 {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n{num_set}\n"

    elif trivia_url == "mean":
        image_mxc, image_width, image_height, new_solution, num_set = generate_mean_question()
        if image_questions == True:
            message_body += f"\n{number_block}📊 {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n{num_set}\n"

    elif trivia_url == "jeopardy":
        if image_questions == True: 
            image_mxc, image_width, image_height = generate_jeopardy_image(trivia_question)
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\nAnd the answer is: \n"
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
            
    elif trivia_category == "Crossword":
        image_mxc, image_width, image_height = generate_crossword_image(trivia_answer_list[0])
        message_body += f"\n{number_block}✏️ {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
        send_image_flag = True

    elif trivia_url == "multiple choice": 
        if trivia_answer_list[0] in {"True", "False"}:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n🚨TRUE or FALSE🚨 {trivia_question}\n\n"
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n🚨Enter Letter🚨 {trivia_question}\n\n"
            for answer in trivia_answer_list[1:]:
                message_body += f"{answer}\n"
        trivia_answer_list[:] = trivia_answer_list[:1]

    else:
         message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
    
    response = send_message(target_room_id, message_body)

    if response is None:
        print("Error: Failed to send the message.")
        return None, None, None
            
    if send_image_flag:  
        image_size = 100
        response = send_image(target_room_id, image_mxc, image_width, image_height, image_size)

        if response is None:                      
            print("Error: Failed to send image.")
            return None, None, None
            
    initialize_sync()
    
    correct_answers = [new_solution] if new_solution else trivia_answer_list
    round_data["questions"].append({
        "question_number": question_number,
        "question_category": trivia_category,
        "question_url": trivia_url,
        "question_text": trivia_question,
        "correct_answers": correct_answers,  
        "user_responses": [] 
    })
    
    # Extracting the 'Date' field
    response_time = response.headers.get('Date')

    # Convert the 'Date' field into a datetime object
    date_format = '%a, %d %b %Y %H:%M:%S %Z'
    response_time = datetime.datetime.strptime(response_time, date_format)

    # Set the timezone explicitly to UTC
    response_time = response_time.replace(tzinfo=pytz.UTC)

    # Convert the datetime object to epoch time in seconds and microseconds (float)
    response_time = response_time.timestamp()

    return response_time, new_question, new_solution

def initialize_sync():
    """Perform an initial sync to update the since_token without processing messages."""
    global since_token, filter_json, headers, params, max_retries, delay_between_retries
    sync_url = f"{matrix_base_url}/sync"
        
    for attempt in range(max_retries):
        try:
            if since_token:
                params["since"] = since_token
                
            # Perform the sync to get the initial token
            response = requests.get(sync_url, headers=headers, params=params)
            
            if response.status_code == 200:
                sync_data = response.json()
                since_token = sync_data.get("next_batch")  # Update the since token without processing messages
                params["since"] = since_token
                return  # Exit the function if successful
            else:
                print(f"Failed to initialize sync. Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(delay_between_retries)
            else:
                print("Max retries reached. Failed to initialize sync.")


def calculate_points(response_time):  
    # Max points is 1000, min points is 100. Points decrease linearly over n seconds
    points = max(1000 - int(response_time * (900 / question_time)), 100)  # Linearly decrease
    points = round(points / 5) * 5  # Round to the nearest 5
    return points

# List of filler words to remove
filler_words = {'a', 'an', 'the', 'of', 'and', 'to', 'in', 'on', 'at', 'with', 'for', 'by'}

def remove_filler_words(input_str):
    """Remove filler words from the input string."""
    words = input_str.split()
    filtered_words = [word for word in words if word not in filler_words]
    return ' '.join(filtered_words)

def normalize_text(input):
    text = input.strip()
    text = text.lower()    
    text = normalize_superscripts(text)
    text = remove_diacritics(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def levenshtein_similarity(str1, str2):
    return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def jaccard_similarity(str1, str2):
    set1, set2 = set(str1.lower()), set(str2.lower())
    return len(set1 & set2) / len(set1 | set2)

def token_based_matching(user_answer, correct_answer):
    user_tokens = set(user_answer.lower().split())
    correct_tokens = set(correct_answer.lower().split())
    return len(user_tokens & correct_tokens) / len(user_tokens | correct_tokens)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def derivative_checker(response, answer):
    response = response.replace(" ", "")      
    answer = answer.replace(" ", "")

    response = response.translate(str.maketrans('', '', string.punctuation))
    answer = answer.translate(str.maketrans('', '', string.punctuation))

    if (response == answer or jaccard_similarity(response, answer) == 1) and len(response) == len(answer):
        return True
    else:
        return False


def fuzzy_match(user_answer, correct_answer, category, url): #POLY
    threshold = 0.90    

    if user_answer == correct_answer:
        return True

    if url == "polynomial factors":
        user_numbers = [int(num) for num in re.findall(r'-?\d+', user_answer)]
        correct_numbers = [int(num) for num in re.findall(r'-?\d+', correct_answer)]
        
        # Ensure user answer has exactly two numbers
        if len(user_numbers) != 2:
            return False
        
        # Check if the two sets of numbers match (order does not matter)
        return set(user_numbers) == set(correct_numbers)
        
    if is_number(correct_answer):
        return user_answer == correct_answer  # Only accept exact match if the correct answer is a number
    
    user_answer = normalize_text(str(user_answer))
    correct_answer = normalize_text(str(correct_answer))

    if url == "multiple choice":
        return user_answer[0] == correct_answer[0];
    
    if is_number(correct_answer):
        return user_answer == correct_answer  # Only accept exact match if the correct answer is a number

    if len(user_answer) < 4:
        return user_answer == correct_answer  # Only accept an exact match for short answers
    
    if user_answer == correct_answer:
        return True
    
    if url == "derivative":
        return derivative_checker(user_answer, correct_answer)

    no_spaces_user = user_answer.replace(" ", "")      
    no_spaces_correct = correct_answer.replace(" ", "") 

    no_filler_user = remove_filler_words(user_answer)
    no_filler_correct = remove_filler_words(correct_answer)

    no_filler_spaces_user = no_filler_user.replace(" ", "")
    no_filler_spaces_correct = no_filler_correct.replace(" ", "")

    if no_spaces_user == no_spaces_correct or no_filler_user == no_filler_correct or no_filler_spaces_user == no_filler_spaces_correct:     
        return True
         
    # New Step: First 5 characters match
    if user_answer[:5] == correct_answer[:5] or no_spaces_user[:5] == no_spaces_correct[:5] or no_filler_user[:5] == no_filler_correct[:5] or no_filler_spaces_user[:5] == no_filler_spaces_correct[:5]:
        return True
    
    # Remove filler words and split correct answer
    correct_answer_words = correct_answer.split()
    no_filler_answer_words = no_filler_correct.split()
    
    # Ensure correct_answer_words is not empty
    if correct_answer_words and len(correct_answer_words[0]) > 3:
        if user_answer == correct_answer_words[0] or no_filler_user == correct_answer_words[0]:
            return True

    if no_filler_answer_words and len(no_filler_answer_words[0]) > 3:
        if user_answer == no_filler_answer_words[0] or no_filler_user == no_filler_answer_words[0]:
            return True

    #Check if user's answer is a substring of the correct answer after normalization
    if user_answer in correct_answer:
        return True
    
    # Step 1: Exact match or Partial match
    if correct_answer in user_answer:
        return True
    
    # Step 2: Levenshtein similarity
    if levenshtein_similarity(user_answer, correct_answer) >= threshold:
        return True

    # Step 3: Jaccard similarity (Character level)
    if jaccard_similarity(user_answer, correct_answer) >= threshold and url != "scramble":
        return True

    # Step 4: Token-based matching
    if token_based_matching(user_answer, correct_answer) >= threshold:
        return True

    return False  # No match found

def collect_responses(question_ask_time, question_number, time_limit):
    global since_token, params, headers, max_retries, delay_between_retries
    sync_url = f"{matrix_base_url}/sync"

    collected_responses = []  # Store all responses
    start_time = time.time()  # Track when the question starts
    processed_events = set()  # Track processed event IDs to avoid duplicates

    
    while time.time() - start_time < time_limit:
        try:
            if since_token:
                params["since"] = since_token

            response = requests.get(sync_url, headers=headers, params=params)

            if response.status_code != 200:
                continue

            sync_data = response.json()
            since_token = sync_data.get("next_batch")  # Update since_token for the next batch

            room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])
            
            for event in room_events:
                event_id = event["event_id"]
                event_type = event.get("type")  # Get the type of the event

                # Only process and redact if the event type is "m.room.message"
                if event_type == "m.room.message":
                    
                    # Skip processing if this event_id was already processed
                    if event_id in processed_events:
                        continue
    
                    # Add event_id to the set of processed events
                    processed_events.add(event_id)
                    sender = event["sender"]
                    
                    message_content = event.get("content", {}).get("body", "")
               
                    emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟", "🛑"]
                    if sender == bot_user_id and any(emoji in message_content for emoji in emojis):
                        continue
    
                    # Redact the message immediately
                    if ghost_mode == True:
                        redact_message(event_id, target_room_id)
                        
                    response_time = event.get("origin_server_ts") / 1000  # Convert to seconds
    
                    # Store response data
                    collected_responses.append({
                        "user_id": sender,
                        "message_content": message_content,
                        "response_time": response_time,
                        "event_id": event_id
                    })
            
            if ghost_mode == False:
                time.sleep(0.3)        
            
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error collecting responses: {e}")

    return collected_responses


def check_correct_responses_delete(question_ask_time, trivia_answer_list, question_number, collected_responses, trivia_category, trivia_url):
    """Check and respond to users who answered the trivia question correctly."""
    global since_token, params, filter_json, headers, max_retries, delay_between_retries, current_longest_answer_streak
    
    # Define the first item in the list as trivia_answer
    trivia_answer = trivia_answer_list[0]  # The first item is the main answer
    correct_responses = []  # To store users who answered correctly
    has_responses = False  # Track if there are any responses

    fastest_correct_user = None
    fastest_response_time = None
    fastest_correct_event_id = None

    # Check if trivia_answer_list is a single-element list with a numeric answer
    single_answer = (len(trivia_answer_list) == 1 and is_number(trivia_answer)) or trivia_url in ["multiple choice", "median", "mean", "polynomial sum", "polynomial product", "polynomial factors"]

    # Dictionary to track first numerical response from each user if answer is a number
    user_first_response = {}

    # Process collected responses
    for response in collected_responses:
        sender = response["user_id"]
        event_id = response["event_id"]
        display_name = get_display_name(sender)  # Get the display name from content
        
        message_content = response.get("message_content", "")  # Use 'response' instead of 'event'
        message_content = message_content.replace("\uFFFC", "")  # Remove U+FFFC

        if "okra" in message_content.lower() and emoji_mode == True:
            react_to_message(event_id, target_room_id, "okra1")

        # Check if the user has already answered correctly, ignore if they have
        if any(resp[0] == display_name for resp in correct_responses):
            continue  # Ignore this response since the user has already answered correctly
        
        # If it's a single numeric answer question, and this user's response is numeric, only record the first one
        if single_answer:
            if display_name in user_first_response:
                continue  # Skip if we've already recorded a numeric response for this user
        
            if is_number(message_content) or message_content.lower() in {"a", "b", "c", "d", "t", "f", "true", "false"}:
                user_first_response[display_name] = message_content
            else:
                continue  # Skip non-numeric responses for single numeric questions
        
        # Log user submission (MongoDB operation)
        log_user_submission(display_name)
                
        # Indicate that there was at least one response
        has_responses = True
                                
        # Find the current question data to add responses
        current_question_data = next((q for q in round_data["questions"] if q["question_number"] == question_number), None)
        if current_question_data:
            current_question_data["user_responses"].append({
                "username": display_name,
                "response": message_content
            })
                                
        # Check if the user's response is in the list of correct answers
        if any(fuzzy_match(message_content, answer, trivia_category, trivia_url) for answer in trivia_answer_list):            
            timestamp = response["response_time"]  # Use the response time from the collected data
            if timestamp and question_ask_time:
                # Convert timestamp to seconds
                response_time = timestamp - question_ask_time
            else:   
                response_time = float('inf')
                
            points = calculate_points(response_time)
            correct_responses.append((display_name, points, response_time, message_content))
    
            # Check if this is the fastest correct response so far
            if fastest_correct_user is None or response_time < fastest_response_time:
                fastest_correct_user = display_name
                fastest_response_time = response_time
                fastest_correct_event_id = event_id
            
    if emoji_mode == True and fastest_correct_event_id is not None:
        react_to_message(fastest_correct_event_id, target_room_id, "fastest")
    
    # Now that we know the fastest responder, iterate over correct_responses to:
    # - Assign the extra 500 points to the fastest user
    # - Update the scoreboard for all users
    for i, (display_name, points, response_time, message_content) in enumerate(correct_responses):
        if display_name == fastest_correct_user:
            correct_responses[i] = (display_name, points + first_place_bonus, response_time, message_content)
          
            if display_name in fastest_answers_count:
                fastest_answers_count[display_name] += 1
            else:
                fastest_answers_count[display_name] = 1
                
            if display_name in scoreboard:
                scoreboard[display_name] += points + first_place_bonus
            else:
                scoreboard[display_name] = points + first_place_bonus
        else:
            if display_name in scoreboard:
                scoreboard[display_name] += points
            else:
                scoreboard[display_name] = points                    

    update_answer_streaks(fastest_correct_user)  # Update the correct answer streak for this user
   
    # Add the current state of the scoreboard to round_data
    current_question_data = next((q for q in round_data["questions"] if q["question_number"] == question_number), None)
    if current_question_data:
        current_question_data["scoreboard_after_question"] = dict(scoreboard)

    # Construct a single message for all the responses
    if ghost_mode == True:
        message = f"\n✅ Answer ✅\n👻👻👻👻👻\n"
    else:
        message = f"\n✅ Answer ✅\n{trivia_answer}\n"
            
    # Notify the chat
    if correct_responses:    
        correct_responses_length = len(correct_responses)
        
        # Loop through the responses and append to the message
        for display_name, points, response_time, message_content in correct_responses:
            time_diff = response_time - fastest_response_time
        
            # Display the formatted message based on yolo_mode
            if time_diff == 0:
                message += f"\n⚡ {display_name}"
                if not yolo_mode:
                    message += f": {points}"
                if points == 420:
                    message += "🌿"
                if current_longest_answer_streak["streak"] > 1:
                    message += f"  🔥{current_longest_answer_streak['streak']}"
            else:
                message += f"\n👥 {display_name}"
                if not yolo_mode:
                    message += f": {points}"
                if points == 420:
                    message += "🌿"

    # Send the entire message at once
    if message:
        send_message(target_room_id, message)

    flush_submission_queue() 
    return None


def update_answer_streaks(user):
    """Update the current longest answer streak for the user who answered correctly."""
    global current_longest_answer_streak

    if current_longest_answer_streak["user"] != user:
        if current_longest_answer_streak["user"] is not None:
            # Append the streak, sort the list in descending order, and keep at most 20 entries
            insert_data_to_mongo("longest_answer_streaks", current_longest_answer_streak)
        current_longest_answer_streak["user"] = user
        current_longest_answer_streak["streak"] = 0

    if user is None:
        save_data_to_mongo("current_streaks", "current_longest_answer_streak", current_longest_answer_streak)
    else:
        current_longest_answer_streak["streak"] += 1
        save_data_to_mongo("current_streaks", "current_longest_answer_streak", current_longest_answer_streak)
        insert_data_to_mongo("fastest_answers", user)

def update_round_streaks(user):
    """Update the current longest round streak for the user who answered correctly."""
    global current_longest_round_streak

    # Variables to store data to be inserted or saved later
    mongo_operations = []

    # Manually copy function for dictionaries
    def manual_copy(data):
        """Manually copy a dictionary by reconstructing it."""
        return {key: value for key, value in data.items()}

    # Check if we need to update the longest round streak
    if current_longest_round_streak["user"] != user:
        if current_longest_round_streak["user"] is not None:
            # Prepare the data to be inserted into longest_round_streaks
            mongo_operations.append({
                "operation": "insert",
                "collection": "longest_round_streaks",
                "data": manual_copy(current_longest_round_streak)  # Manually copy the data
            })
        # Update the user and reset the streak
        current_longest_round_streak["user"] = user
        current_longest_round_streak["streak"] = 0

    # Increment streak or handle no user case
    if user is None:
        mongo_operations.append({
            "operation": "save",
            "collection": "current_streaks",
            "document_id": "current_longest_round_streak",
            "data": manual_copy(current_longest_round_streak)  # Manually copy the data
        })
    else:
        current_longest_round_streak["streak"] += 1
        mongo_operations.append({
            "operation": "save",
            "collection": "current_streaks",
            "document_id": "current_longest_round_streak",
            "data": manual_copy(current_longest_round_streak)  # Manually copy the data
        })
        mongo_operations.append({
            "operation": "insert",
            "collection": "round_wins",
            "data": user  # If user is simple data (e.g., string), no need to copy
        })

    # Generate the round summary if the user is not None
    if user is not None:
        
        if current_longest_round_streak["streak"] > 1:
            message = f"\n🏆 Winner: @{user}...🔥{current_longest_round_streak['streak']} in a row!\n\n▶️ Live trivia stats available: https://stats.redditlivetrivia.com\n"
        else:
            message = f"\n🏆 Winner: @{user}!\n\n▶️ Live trivia stats available: https://stats.redditlivetrivia.com\n"

        send_message(target_room_id, message)
        time.sleep(2)
        
        select_wof_questions(user)
        
        gpt_summary = generate_round_summary(round_data, user)

        print(gpt_summary)

        gpt_message = f"\n{gpt_summary}\n"
        send_message(target_room_id, gpt_message)
        
        if current_longest_round_streak['streak'] % image_wins == 0:
            time.sleep(5)
            generate_round_summary_image(round_data, user)
        else:
            number_to_emoji = {
                1: "1️⃣",
                2: "2️⃣",
                3: "3️⃣",
                4: "4️⃣",
                5: "5️⃣",
                6: "6️⃣",
                7: "7️⃣",
                8: "8️⃣",
                9: "9️⃣",
                10: "🔟"
            }
            
            time.sleep(4)
            remaining_games = image_wins - (current_longest_round_streak['streak'] % image_wins)
            dynamic_emoji = number_to_emoji[remaining_games]
            
            if remaining_games == 1:
                image_message = f"\n{dynamic_emoji}🎨 @{user} Win the next game and I'll draw you something.\n"
            else:
                image_message = f"\n{dynamic_emoji}🎨 @{user} Win {remaining_games} more in a row and I'll draw you something.\n"
            
            image_message += "\n🥒🏛️ https://redditlivetrivia.com/okra-museum\n"
                
            send_message(target_room_id, image_message)
            time.sleep(1)

    # Perform all MongoDB operations at the end
    for operation in mongo_operations:
        if operation["operation"] == "insert":
            insert_data_to_mongo(operation["collection"], operation["data"])
        elif operation["operation"] == "save":
            save_data_to_mongo(operation["collection"], operation["document_id"], operation["data"])
            

def determine_round_winner():
    """Determine the round winner based on points and response times."""
    if not scoreboard:
        return None, None

    # Find the user(s) with the most points
    max_points = max(scoreboard.values())
    potential_winners = [user for user, points in scoreboard.items() if points == max_points]

    # If there's a tie, return None (no clear-cut winner)
    if len(potential_winners) > 1:
        send_message(target_room_id, "No clear-cut winner this round due to a tie.")
        return None, None
    else:
        return potential_winners[0], max_points  # Clear-cut winner


def show_standings():
    """Show the current standings after each question."""
    if scoreboard:
        standings = sorted(scoreboard.items(), key=lambda x: x[1], reverse=True)
        standing_message = "\n📈 Scoreboard 📈"
        
        # Define the medals for the top 3 positions
        medals = ["🥇", "🥈", "🥉"]
        
        for rank, (user, points) in enumerate(standings, start=1):
            formatted_points = f"{points:,}"  # Format points with commas
            fastest_count = fastest_answers_count.get(user, 0)  # Get the user's fastest answer count, default to 0
            
            lightning_display = f" ⚡{fastest_count}" if fastest_count > 1 else " ⚡" if fastest_count == 1 else ""
            
            if points == 420:
                standing_message += f"\n🥴 {user}: {formatted_points}"
                
            elif rank <= 3:
                standing_message += f"\n{medals[rank-1]} {user}: {formatted_points}"
                
            elif rank == len(standings) and rank > 4:
                standing_message += f"\n💩 {user}: {formatted_points}"
                
            else:
                standing_message += f"\n{rank}. {user}: {formatted_points}"

            standing_message += lightning_display
        
        send_message(target_room_id, standing_message)


def store_question_ids_in_mongo(question_ids, question_type):
    db = connect_to_mongodb()
    collection_name = f"asked_{question_type}_questions"
    questions_collection = db[collection_name]

    # Insert the new IDs directly into the collection
    questions_collection.insert_many([{"_id": _id} for _id in question_ids])

    # Check if the collection exceeds its limit and delete old entries if necessary
    limit = id_limits[question_type]
    total_ids = questions_collection.count_documents({})
    if total_ids > limit:
        excess = total_ids - limit
        oldest_entries = questions_collection.find().sort("timestamp", 1).limit(excess)
        for entry in oldest_entries:
            questions_collection.delete_one({"_id": entry["_id"]})


def get_recent_question_ids_from_mongo(question_type):
    db = connect_to_mongodb()
    collection_name = f"asked_{question_type}_questions"
    questions_collection = db[collection_name]

    recent_ids = questions_collection.find().sort("timestamp", -1).limit(id_limits[question_type])
    return {doc["_id"] for doc in recent_ids}



def select_trivia_questions(questions_per_round):
    global categories_to_exclude
    try:
        db = connect_to_mongodb()
        
        # Fetch recent IDs separately for each type
        recent_general_ids = get_recent_question_ids_from_mongo("general")
        recent_crossword_ids = get_recent_question_ids_from_mongo("crossword")
        recent_jeopardy_ids = get_recent_question_ids_from_mongo("jeopardy")
        recent_mysterybox_ids = get_recent_question_ids_from_mongo("mysterybox")
        recent_wof_ids = get_recent_question_ids_from_mongo("wof")

        selected_questions = []

        
        math_questions = [get_math_question() for _ in range(num_math_questions)]
        stats_questions = [get_stats_question() for _ in range(num_stats_questions)]
        selected_questions.extend(math_questions)
        selected_questions.extend(stats_questions)

        # Fetch wheel of fortune questions using the random subset method
        wof_collection = db["wof_questions"]
        pipeline_wof = [
            {"$match": {"_id": {"$nin": list(recent_wof_ids)}}},
            {"$sample": {"size": num_wof_clues}}  # Apply sampling on the filtered subset
        ]
        wof_questions = list(wof_collection.aggregate(pipeline_wof))
        selected_questions.extend(wof_questions)
 
        # Fetch mysterybox questions using the random subset method
        mysterybox_collection = db["mysterybox_questions"]
        pipeline_mysterybox = [
            {"$match": {"_id": {"$nin": list(recent_mysterybox_ids)}}},
            {"$sample": {"size": num_mysterybox_clues}}  # Apply sampling on the filtered subset
        ]
        mysterybox_questions = list(mysterybox_collection.aggregate(pipeline_mysterybox))
        selected_questions.extend(mysterybox_questions)
        
        # Fetch crossword questions using the random subset method
        crossword_collection = db["crossword_questions"]
        pipeline_crossword = [
            {"$match": {"_id": {"$nin": list(recent_crossword_ids)}}},
            {"$sample": {"size": num_crossword_clues}}  # Apply sampling on the filtered subset
        ]
        crossword_questions = list(crossword_collection.aggregate(pipeline_crossword))
        selected_questions.extend(crossword_questions)

        # Fetch jeopardy questions using the random subset method
        jeopardy_collection = db["jeopardy_questions"]
        pipeline_jeopardy = [
            {"$match": {"_id": {"$nin": list(recent_jeopardy_ids)}}},
            {"$sample": {"size": num_jeopardy_clues}}  # Apply sampling on the filtered subset
        ]
        jeopardy_questions = list(jeopardy_collection.aggregate(pipeline_jeopardy))
        selected_questions.extend(jeopardy_questions)

        # Calculate the remaining questions needed for general trivia
        remaining_needed = max(questions_per_round - len(wof_questions) - len(mysterybox_questions) - len(crossword_questions) - len(jeopardy_questions) - len(math_questions) - len(stats_questions), 0)
        
        if remaining_needed > 0:
            trivia_collection = db["trivia_questions"]

            if image_questions == False:
                # Define a list of substrings to exclude in URLs
                excluded_url_substring = "http"
                pipeline_trivia = [
                    {
                        "$match": {
                            "_id": {"$nin": list(recent_general_ids)},
                            "category": {"$nin": categories_to_exclude},
                            "$or": [
                                {"url": {"$not": {"$regex": excluded_url_substring}}} 
                            ]
                        }
                    },
                    {
                        "$group": {
                            "_id": "$category",
                            "questions": {"$push": "$$ROOT"}  # Push full document to each category group
                        }
                    },
                    {"$unwind": "$questions"},  # Unwind the limited question list for each category back into individual documents
                    {"$replaceRoot": {"newRoot": "$questions"}},  # Flatten to original document structure
                    {"$sample": {"size": remaining_needed}}  # Sample from the resulting limited set
                ]
                
            else:
                pipeline_trivia = [
                    {"$match": {"_id": {"$nin": list(recent_general_ids)}, "category": {"$nin": categories_to_exclude}}},
                    {
                        "$group": {
                            "_id": "$category",
                            "questions": {"$push": "$$ROOT"}  # Push full document to each category group
                        }
                    },
                    {"$unwind": "$questions"},  # Unwind the limited question list for each category back into individual documents
                    {"$replaceRoot": {"newRoot": "$questions"}},  # Flatten to original document structure
                    {"$sample": {"size": remaining_needed}}  # Sample from the resulting limited set
                ]

            trivia_questions = list(trivia_collection.aggregate(pipeline_trivia))
            selected_questions.extend(trivia_questions)

            # Store separate sets of IDs in MongoDB only if they are non-empty
            wof_question_ids = [doc["_id"] for doc in wof_questions]
            if wof_question_ids:
                store_question_ids_in_mongo(wof_question_ids, "wof")
            
            mysterybox_question_ids = [doc["_id"] for doc in mysterybox_questions]
            if mysterybox_question_ids:
                store_question_ids_in_mongo(mysterybox_question_ids, "mysterybox")
            
            crossword_question_ids = [doc["_id"] for doc in crossword_questions]
            if crossword_question_ids:
                store_question_ids_in_mongo(crossword_question_ids, "crossword")

            jeopardy_question_ids = [doc["_id"] for doc in jeopardy_questions]
            if jeopardy_question_ids:
                store_question_ids_in_mongo(jeopardy_question_ids, "jeopardy")

            general_question_ids = [doc["_id"] for doc in trivia_questions]
            if general_question_ids:
                store_question_ids_in_mongo(general_question_ids, "general")

        # Shuffle the combined list of selected questions
        random.shuffle(selected_questions)

        final_selected_questions = [
            (doc["category"], doc["question"], doc["url"], doc["answers"])
            for doc in selected_questions
        ]

        return final_selected_questions

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Error selecting trivia and crossword questions: {e}")
        return []  # Return an empty list in case of failure
        
def load_streak_data():
    global current_longest_answer_streak, current_longest_round_streak
    
    for attempt in range(max_retries):
        try:
            db = connect_to_mongodb()
            
            # Retrieve the current longest answer streak from MongoDB
            document_answer = db.current_streaks.find_one({"_id": "current_longest_answer_streak"})

            if document_answer is not None:
                current_longest_answer_streak = {
                    "user": document_answer.get("user"),
                    "streak": document_answer.get("streak")
                }
            else:
                # If the document is not found, set default values
                current_longest_answer_streak = {"user": None, "streak": 0}

            # Retrieve the current longest round streak from MongoDB
            # Retrieve the current longest answer streak from MongoDB
            document_round = db.current_streaks.find_one({"_id": "current_longest_round_streak"})
            
            if document_round is not None:
                current_longest_round_streak = {
                    "user": document_round.get("user"),
                    "streak": document_round.get("streak")
                }
            else:
                # If the document is not found, set default values
                current_longest_round_streak = {"user": None, "streak": 0}
                
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay_between_retries} seconds...")
                time.sleep(delay_between_retries)
            else:
                print("Max retries reached. Data loading failed.")
                # Set to default values if loading fails
                current_longest_answer_streak = {"user": None, "streak": 0}
                current_longest_round_streak = {"user": None, "streak": 0}


def print_selected_questions(selected_questions):
    """Prints the selected questions in a cleaner format."""
    for i, question_data in enumerate(selected_questions, start=1):
        category = question_data[0]  # Category
        question = question_data[1]  # Question
        answers = question_data[3] # Answers
        # Format and print the question and answers
        print(f"{i}. [{category}] {question} [{', '.join(answers)}]")


def round_start_messages():
    db = connect_to_mongodb()
    top_users = list(db.top_users.find())
    sovereigns = {sovereign['user'] for sovereign in db.hall_of_sovereigns.find()}

    messages = []
    for user in top_users:
        username = user.get('user')
        top_count = user.get('top_count')

        # If the user is in the Hall of Sovereigns, only show the message if top_count == 6
        if username in sovereigns:
            if top_count == 6:
                send_message(target_room_id, f"👑  {username} is #1 across the board. They are our Sovereign. We all bow to you.\n\n▶️ Live trivia stats available: https://stats.redditlivetrivia.com\n")
        else:
            # For users not in the Hall of Sovereigns, show all applicable messages
            if top_count == 6:
                send_message(target_room_id, f"👑  {username} is #1 across the board. They are our Sovereign. We all bow to you.\n\n▶️ Live trivia stats available: https://stats.redditlivetrivia.com\n")
            elif top_count == 5:
                send_message(target_room_id, f"🔥​  {username} is on fire! Only 1 leaderboard left.\n\n▶️ Live trivia stats available: https://stats.redditlivetrivia.com\n")
            elif top_count == 4:
                send_message(target_room_id, f"🌡️  {username} is heating up! Only 2 leaderboards left.\n\n▶️ Live trivia stats available: https://stats.redditlivetrivia.com\n")
    return None



# Mapping to convert integers to superscript characters
superscript_map = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
}

# Reverse mapping to convert superscript characters back to regular numbers
reverse_superscript_map = {v: k for k, v in superscript_map.items()}

def to_superscript(num):
    return ''.join(superscript_map[digit] for digit in str(num))

def normalize_superscripts(text):
    return ''.join(reverse_superscript_map.get(char, char) for char in text)


def generate_and_render_derivative_image():
    # Randomly select two unique powers from {1, 2, 3}
    powers = sorted(random.sample([1, 2, 3], 2), reverse=True)
    content_uri = True
    
    terms = []
    derivative_terms = []

    # Construct polynomial and derivative terms for the selected powers
    for power in powers:
        coef = random.randint(1, 9)  # Coefficients between 1 and 9
        coef_str = str(coef) if coef != 1 else ""  # Omit "1" as a coefficient unless constant

        # Construct polynomial term with superscript exponents
        if power == 1:
            terms.append(f"{coef_str}x")  # No exponent shown for power of 1
            derivative_terms.append(f"{coef}")
        else:
            terms.append(f"{coef_str}x{to_superscript(power)}")  # Display higher powers with superscript
            derivative_terms.append(f"{coef * power}x{to_superscript(power - 1) if power > 2 else ''}")

    # Join the terms for both polynomial and derivative strings
    polynomial = " + ".join(terms)
    derivative = " + ".join(derivative_terms) if derivative_terms else "0"

    print(f"Polynomial: {polynomial}")
    print(f"Derivative: {derivative}")

    # Define the font path relative to the current script
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")

    # Create a blank image
    img_width, img_height = 600, 150
    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load the font
    font_size = 48
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return None, None, None, None, None

    # Draw the polynomial text in the center
    text_bbox = draw.textbbox((0, 0), polynomial, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img_width - text_width) // 2
    text_y = (img_height - text_height) // 2
    draw.text((text_x, text_y), polynomial, fill=(255, 255, 0), font=font)

    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer

    # Upload the image to Matrix
    if image_questions == True:
        content_uri = upload_image_to_matrix(image_buffer.read())
    if content_uri:
        return content_uri, img_width, img_height, derivative, polynomial
    else:
        print("Failed to upload the image to Matrix.")


def generate_and_render_polynomial(type):
    # Randomly select two unique integers from -9 to 9, excluding 0
    content_uri = True
    factors = [random.choice([i for i in range(-9, 10) if i != 0]) for _ in range(2)]
    sum_factors = sum(factors)
    product_factors = factors[0] * factors[1]

    # Construct the sum term for the polynomial
    if abs(sum_factors) == 1:
        sum_term = ""
    else:
        sum_term = abs(sum_factors)

    if sum_term == 0:
        polynomial = f"x² {'+' if product_factors >= 0 else '-'} {abs(product_factors)}"
    else:
        polynomial = f"x² {'+' if sum_factors >= 0 else '-'} {sum_term}x {'+' if product_factors >= 0 else '-'} {abs(product_factors)}"
    
    print(f"Polynomial: {polynomial}")

    if type == "sum":
         print(f"Sum of zeroes: {sum_factors}")
    elif type == "product":
         print(f"Product of zeroes: {product_factors}")
    elif type == "factors":
         print(f"Zeroes: {factors}")
    else:
        print("Wrong type passed in to polynomial function")

    # Define the font path relative to the current script
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")

    # Create a blank image
    img_width, img_height = 600, 150
    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load the font
    font_size = 48
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return None, None, None, None, None

    # Draw the polynomial text in the center in light purple
    text_bbox = draw.textbbox((0, 0), polynomial, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img_width - text_width) // 2
    text_y = (img_height - text_height) // 2
    draw.text((text_x, text_y), polynomial, fill=(200, 162, 200), font=font)  # Light purple color

    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer

    # Upload the image to Matrix
    if image_questions == True:
        content_uri = upload_image_to_matrix(image_buffer.read())
        
    if content_uri:
        if type == "sum":
            return content_uri, img_width, img_height, str(int(sum_factors)), polynomial
        elif type == "product":
            return content_uri, img_width, img_height, str(int(product_factors)), polynomial
        elif type == "factors":
            factors_str = f"{factors[0]} and {factors[1]}"
            return content_uri, img_width, img_height, factors_str, polynomial
        else:
            return content_uri, img_width, img_height, str(int(sum_factors)), polynomial
    else:
        print("Failed to upload the image to Matrix.")





def round_preview(selected_questions):
    numbered_blocks = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    message = "\n🔮 Next Round Preview 🔮\n"
    
    for i, question_data in enumerate(selected_questions):
        trivia_category = question_data[0]
        trivia_url = question_data[2]
        number_block = numbered_blocks[i] if i < len(numbered_blocks) else f"{i + 1}️⃣"  # Use fallback if needed
        message += f"{number_block} {get_category_title(trivia_category, trivia_url)}\n"
    
    message += "\n"
    # Send the message to the chat
    send_message(target_room_id, message)


def get_category_title(trivia_category, trivia_url):
    # Define the emoji lookup table
    emoji_lookup = {
        "Mystery Box or Boat": "🎁🛳️",
        "Famous People": "👑🧑‍🎤",
        "Anatomy": "🧠🫀",
        "Characters": "🧙‍♂️🧛",
        "Music": "🎶🎸",
        "Art & Literature": "🎨📚",
        "Chemistry": "🧪⚗️",
        "Geography": "🧭🗺️",
        "Mathematics": "➕➗",
        "Statistics": "📊🔢",
        "Physics": "⚛️🍎",
        "Science & Nature": "🔬🌺",
        "Language": "🗣️🔤",
        "English Grammar": "📝✏️",
        "Astronomy": "🪐🌙",
        "Logos": "🏷️🔍",
        "The World": "🌎🌍",
        "Economics & Government": "💵⚖️",
        "Toys & Games": "🧸🎲",
        "Food & Drinks": "🍕🍹",
        "Geology": "🪨🌋",
        "Tech & Video Games": "💻🎮",
        "Flags": "🏳️🏴",
        "Miscellaneous": "🔀✨",
        "Biology": "🧬🦠",
        "Superheroes": "🦸‍♀️🦸",
        "Television": "📺🎥",
        "Pop Culture": "🎉🌟",
        "History": "📜🕰️",
        "Movies": "🎬🍿",
        "Religion & Mythology": "🛐🐉",
        "Sports & Leisure": "⚽🌴",
        "World Culture": "🎭🗿",
        "General Knowledge": "📚💡"
    }

    # Check if the question URL is "jeopardy"
    if trivia_url.lower() == "jeopardy":
        return f"{trivia_category} 🟦🇯"
    # Otherwise, get the emojis based on the lookup table, defaulting to the category itself if not found
    emojis = emoji_lookup.get(trivia_category, "😊😞")
    return f"{trivia_category} {emojis}"



def get_player_selected_question(questions, round_winner):
    global since_token

    # Display categories for user selection
    categories = [q[0] for q in questions]
    num_of_questions = len(questions)
    
    message = "\n" f"@{round_winner} choose a number: \n\n"


    numbered_blocks = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    for i, question_data in enumerate(questions):
        trivia_category = question_data[0]
        trivia_url = question_data[2]
        number_block = numbered_blocks[i] if i < len(numbered_blocks) else f"{i + 1}️⃣"  # Use fallback if needed
        message += f"{number_block} {get_category_title(trivia_category, trivia_url)}\n"
    
    message += "\n"
    # Send the message to the chat
    send_message(target_room_id, message)

    initialize_sync()
    time.sleep(10)

     # Fetch responses
    sync_url = f"{matrix_base_url}/sync"
    params_with_since = params.copy()  # Use the existing params but include the since token
    
    if since_token:
        params_with_since["since"] = since_token

    try:
        response = requests.get(sync_url, headers=headers, params=params_with_since)
        if response.status_code != 200:
            print(f"Failed to fetch responses. Status code: {response.status_code}")
            return

        # Parse the response to get the timeline events
        sync_data = response.json()
        since_token = sync_data.get("next_batch")  # Update the since_token for future requests
        room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])

        # Process all responses in reverse order (latest response first)
        for event in room_events:
            sender = event["sender"]
            message_content = event.get("content", {}).get("body", "").strip()
        
            # Proceed only if message_content is not empty
            if message_content:
                # Fetch the display name for the current user
                sender_display_name = get_display_name(sender)
        
                # If the round winner responded, process the award accordingly
                if sender_display_name == round_winner:
                    if any(str(i) in message_content for i in range(1, num_of_questions + 1)):
                        try:
                            question_number = int(''.join(filter(str.isdigit, message_content)))
        
                            # Ensure the delay value is within the allowed range (1-10)
                            question_number = max(1, min(question_number, num_of_questions))
                            return question_number
                            
                        except ValueError:
                            pass
                            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching responses: {e}")  

    return 1


def refill_question_slot(questions, old_question):
    """Replace the old question with a new random question."""
    # Remove the old question
    questions.remove(old_question)
    
    # Get a random new question from the database
    #new_question = get_random_trivia_question()
    
    # Append the new question to the end of the list to maintain order
    #questions.append(new_question)


def get_random_trivia_question():
    global categories_to_exclude
    """Fetch a random question from the trivia_questions collection."""
    try:
        db = connect_to_mongodb()
        trivia_collection = db["trivia_questions"]
        
        recent_general_ids = get_recent_question_ids_from_mongo("general")
            
        pipeline = [
            {"$match": {"_id": {"$nin": list(recent_general_ids)}, "category": {"$nin": categories_to_exclude}}},
            {
                "$group": {
                    "_id": "$category",
                    "questions": {"$push": "$$ROOT"}  # Push full document to each category group
                }
            },
            {"$unwind": "$questions"},  # Unwind the limited question list for each category back into individual documents
            {"$replaceRoot": {"newRoot": "$questions"}},  # Flatten to original document structure
            {"$sample": {"size": 1}}  # Sample from the resulting limited set
        ]
        
        result = list(trivia_collection.aggregate(pipeline))

        if result:
            selected_question = result[0]
            question_id = selected_question["_id"]
        
            # Store the ID in MongoDB to avoid re-selection in future rounds
            store_question_ids_in_mongo([question_id], "general")

            final_selected_question = (
                selected_question["category"],
                selected_question["question"],
                selected_question["url"],
                selected_question["answers"]
            )
            
            return final_selected_question
        else:
            print("No available questions found.")
            return None
            
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Error selecting trivia and crossword questions: {e}")
        return None  # Return an empty list in case of failure


def start_trivia_round():
    okra_gif_urls = [
        "https://triviabotwebsite.s3.us-east-2.amazonaws.com/okra/okra1.gif",
        "https://triviabotwebsite.s3.us-east-2.amazonaws.com/okra/okra2.gif",
        "https://triviabotwebsite.s3.us-east-2.amazonaws.com/okra/okra3.gif",
        "https://triviabotwebsite.s3.us-east-2.amazonaws.com/okra/okra4.gif",
        "https://triviabotwebsite.s3.us-east-2.amazonaws.com/okra/okra5.gif",
        "https://triviabotwebsite.s3.us-east-2.amazonaws.com/okra/okra7.gif",
        "https://triviabotwebsite.s3.us-east-2.amazonaws.com/okra/okra8.gif",
    ]
    
# Function to start the trivia round
    global target_room_id, bot_user_id, bearer_token, question_time, questions_per_round, time_between_rounds, time_between_questions, filler_words
    global scoreboard, current_longest_round_streak, current_longest_answer_streak
    global headers, params, filter_json, since_token, round_count, selected_questions, magic_number

    # Track the initial time for hourly re-login
    last_login_time = time.time()  # Store the current time when the script starts
    round_winner = None
    selected_questions = select_trivia_questions(questions_per_round)  #Pick the initial question set
    
    try:
        while True:  # Endless loop            
            # Check if it's been more than an hour since the last login
            current_time = time.time()
            
            if current_time - last_login_time >= 3600:  # 3600 seconds = 1 hour
                print("Re-logging into Reddit and chat as one hour has passed...")
                reddit_login()
                login_to_chat()
                last_login_time = current_time  # Reset the login time

            # Load global varaiables at the start of round
            load_global_variables()
            load_parameters()
            
            # Load existing streak data from the file
            load_streak_data()

            # Fetch new coffee donations
            fetch_donations()

            # Reset the scoreboard and fastest answers at the start of each round
            scoreboard.clear()
            fastest_answers_count.clear()
            
            # Reset round data for the next round
            round_data["questions"] = []


            if random.random() < 0:  # random.random() generates a float between 0 and 1
                magic_number = random_number = random.randint(1000, 9999)
                print(f"Magic number is {magic_number}")
                send_magic_image(magic_number)
            elif image_questions == True:
                selected_gif_url = random.choice(okra_gif_urls)
                image_mxc, image_width, image_height = download_image_from_url(selected_gif_url)
                send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)
                #time.sleep(7)

            send_message(target_room_id, f"\n⏩ Starting a round of {questions_per_round} questions ⏩\n\n🏁 Get ready 🏁\n\n")
            round_start_messages()
            time.sleep(3)
                
            # Randomly select n questions
            print_selected_questions(selected_questions)
            
            question_number = 1
            while question_number <= questions_per_round:
                
                if god_mode and round_winner and len(selected_questions)>1:
                    selected_question = selected_questions[get_player_selected_question(selected_questions, round_winner) - 1]
                    
                else:
                    selected_question = selected_questions[0]

                trivia_category, trivia_question, trivia_url, trivia_answer_list = selected_question
                question_ask_time, new_question, new_solution = ask_question(trivia_category, trivia_question, trivia_url, trivia_answer_list, question_number)         
                collected_responses = collect_responses(question_time, question_number, question_time)
                
                #send_message(target_room_id, f"\n🛑 TIME 🛑\n")
                
                solution_list = trivia_answer_list if new_solution is None else [new_solution]            
                check_correct_responses_delete(question_ask_time, solution_list, question_number, collected_responses, trivia_category, trivia_url)
                
                if not yolo_mode or question_number == questions_per_round:
                    show_standings()

                #Refill the question slot with a new random question from trivia_questions
                refill_question_slot(selected_questions, selected_question)

                time.sleep(time_between_questions)  # Small delay before the next question
                
                question_number = question_number + 1
                
            #Determine the round winner
            round_winner, winner_points = determine_round_winner()

            #Update round streaks
            update_round_streaks(round_winner)
            # Increment the round count

            round_count += 1
        
            time.sleep(10)
            process_round_options(round_winner, winner_points)
            
            if round_count % 5 == 0:
                send_message(target_room_id, f"\n🧘‍♂️ A short breather. Relax, stretch, meditate.\n🎨 Live Trivia is a pure hobby effort.\n💡 Help Okra improve it: https://forms.gle/iWvmN24pfGEGSy7n7\n")
                selected_questions = select_trivia_questions(questions_per_round)  #Pick the next question set
                round_preview(selected_questions)
                time.sleep(10)
            else:
                #send_message(target_room_id, f"\n💚 Friendly OkraStrut? Buy coffee with your Reddit name.\n☕️ https://buymeacoffee.com/livetrivia\n👕 https://merch.redditlivetrivia.com\n")
                send_message(target_room_id, f"\n☕️ https://buymeacoffee.com/livetrivia\n💚 Use your Reddit name to unlock in-game perks.\n")
                selected_questions = select_trivia_questions(questions_per_round)  #Pick the next question set
                round_preview(selected_questions)
                time.sleep(10)  # Adjust this time to whatever delay you need between rounds

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Error occurred: {e}")
        traceback.print_exc()  # Print the stack trace of the error
        print("Restarting the trivia bot in 10 seconds...")
        time.sleep(10)  

try:
    sentry_sdk.capture_message("Sentry initiatlized...", level="info")
    reddit_login()
    login_to_chat()
    
    # Load needed variables for sync
    load_global_variables()
    load_parameters()
    
    # Call this function at the start of the script to initialize the sync
    initialize_sync()    
    
    # Start the trivia round
    start_trivia_round()

except Exception as e:
    sentry_sdk.capture_exception(e)
    print(f"Unhandled exception: {e}. Restarting in 5 seconds...")
    traceback.print_exc()  # Print the stack trace for debugging
    time.sleep(5)
    reddit_login()
    login_to_chat()
    load_global_variables()
    initialize_sync()
    start_trivia_round()  # Restart the bot
