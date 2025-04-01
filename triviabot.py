
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
from datetime import timezone
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
from collections import Counter, defaultdict
import math
import sys
import signal


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

question_responders = []  # Tracks users who responded during the current question
round_responders = []

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
webster_api_key = os.getenv("webster_api_key")
webster_thes_api_key = os.getenv("webster_thes_api_key")
target_room_id = os.getenv("target_room_id")
question_time = int(os.getenv("question_time"))
questions_per_round = int(os.getenv("questions_per_round"))
time_between_questions = int(os.getenv("time_between_questions"))
time_between_questions_default = time_between_questions
max_retries = int(os.getenv("max_retries"))
delay_between_retries = int(os.getenv("delay_between_retries"))
id_limits = {"general": 2000, "mysterybox": 2000, "crossword": 100000, "jeopardy": 100000, "wof": 1500, "list": 20, "feud": 1000, "posters": 2000, "movie_scenes": 5000, "missing_link": 2500, "people": 2500, "ranker_list": 4000, "animal": 2000, "riddle": 2500, "dictionary": 100000, "flags": 800}
first_place_bonus = 0
magic_time = 10
magic_number = 0000

ghost_mode_default = False
ghost_mode = ghost_mode_default
god_mode_default = False
god_mode = god_mode_default
god_mode_points = 5000
god_mode_players = 5
yolo_mode_default = False
yolo_mode = yolo_mode_default
emoji_mode_default = True
emoji_mode = emoji_mode_default
collect_feedback_mode_default = True
collect_feedback_mode = collect_feedback_mode_default
magic_number_correct = False
wf_winner = False
nice_okra = False
creep_okra = False
seductive_okra = False
joke_okra = False
blind_mode_default = False
blind_mode = blind_mode_default
marx_mode_default = False
marx_mode = marx_mode_default



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


def handle_sigterm(signum, frame):
    print(f"Received signal {signum}. Printing stack trace:")
    traceback.print_stack(frame)
    sys.exit(0)  # Exit gracefully after handling SIGTERM


def select_intro_image_url():
    # Connect to the collection
    collection = db["intro_image_urls"]

    # Fetch one random document where enabled is True
    result = list(
        collection.aggregate([
            {"$match": {"enabled": True}},
            {"$sample": {"size": 1}}
        ])
    )

    if not result:
        print("No enabled intro image URLs found.")
        return None

    # Return just the URL field
    return result[0].get("url")


def create_family_feud_board_image(total_answers, user_answers, num_of_xs=0):
    """
    Creates a Family Feud–style board with:
      - A golden arc / scoreboard at the top
      - Wide boxes for each answer, in a single column
      - Circles on the left for numbering
      - Large fonts for both scoreboard and answers
      - Unrevealed answers display a small Okra image from a URL
      - Overlays up to 3 big red 'X's if num_of_xs > 0 (Family Feud strikes)
      - Returns (image_mxc, width, height).
    """

    # Convert user_answers to a case-insensitive set
    lower_user_answers = {ua.lower() for ua in user_answers}
    n = len(total_answers)

    width = 3600
    height = 800 + (n * 240)
    bg_color = (10, 10, 10)
    gold_color = (255, 215, 0)
    #gold_color = (62, 145, 45)
    box_color = (0, 60, 220)
    #box_color = (97, 51, 47)
    box_outline = (255, 255, 255)
    txt_color = (255, 255, 255)
    circle_color = (0, 0, 150)
    #circle_color = (34, 49, 29)

    # Create the blank image
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Fonts
    try:
        scoreboard_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 80)
    except:
        scoreboard_font = ImageFont.load_default()

    try:
        answer_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 140)
    except:
        answer_font = ImageFont.load_default()

    try:
        circle_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 80)
    except:
        circle_font = ImageFont.load_default()

    # 1) Golden Arc
    arc_x1, arc_y1 = 0, 0
    arc_x2, arc_y2 = width, height * 2
    draw.pieslice([arc_x1, arc_y1, arc_x2, arc_y2], start=180, end=360, fill=gold_color)

    # 2) Scoreboard rectangle
    scoreboard_w = 800
    scoreboard_h = 150
    scoreboard_x = (width - scoreboard_w) // 2
    scoreboard_y = 60
    scoreboard_rect = [scoreboard_x, scoreboard_y, scoreboard_x + scoreboard_w, scoreboard_y + scoreboard_h]
    draw.rectangle(scoreboard_rect, fill=(0, 0, 150))

    scoreboard_text = "Okra"
    # measure scoreboard text
    try:
        left, top, right, bottom = draw.textbbox((0, 0), scoreboard_text, font=scoreboard_font)
        txt_w, txt_h = right - left, bottom - top
    except:
        mask = scoreboard_font.getmask(scoreboard_text)
        txt_w, txt_h = mask.size

    sb_text_x = scoreboard_x + (scoreboard_w - txt_w) // 2
    sb_text_y = scoreboard_y + (scoreboard_h - txt_h) // 2
    draw.text((sb_text_x, sb_text_y), scoreboard_text, fill=(255, 255, 255), font=scoreboard_font)

    # 3) Single-column answer boxes
    box_height = 240
    box_width = 2500
    box_spacing = 40
    top_offset = scoreboard_y + scoreboard_h + 160
    left_margin = (width - box_width) // 2

    # -- Download Okra image from URL for unrevealed answers --
    #    We'll fetch it and convert to RGBA for transparency if needed.
    okra_url = "https://triviabotwebsite.s3.us-east-2.amazonaws.com/okra/okra_ff.png"
    okra_icon = None
    try:
        response = requests.get(okra_url, timeout=5)
        response.raise_for_status()
        okra_icon = Image.open(io.BytesIO(response.content)).convert("RGBA")
        # optional: resize if it's too large
        okra_icon = okra_icon.resize((106, 190), resample=Image.LANCZOS)
    except Exception as e:
        print(f"Could not load Okra image from URL: {e}")
        okra_icon = None

    for i, ans in enumerate(total_answers):
        box_x = left_margin
        box_y = top_offset + i*(box_height + box_spacing)

        draw.rectangle(
            [box_x, box_y, box_x + box_width, box_y + box_height],
            fill=box_color,
            outline=box_outline,
            width=8
        )

        # Circle for numbering
        circle_diam = 160
        circle_x1 = box_x - circle_diam//2
        circle_y1 = box_y + (box_height - circle_diam)//2
        circle_x2 = circle_x1 + circle_diam
        circle_y2 = circle_y1 + circle_diam
        draw.ellipse([circle_x1, circle_y1, circle_x2, circle_y2],
                     fill=circle_color, outline=box_outline, width=8)

        number_str = str(i + 1)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), number_str, font=circle_font)
            num_w, num_h = right - left, bottom - top
        except:
            mask = circle_font.getmask(number_str)
            num_w, num_h = mask.size
        
        num_x = circle_x1 + (circle_diam - num_w)//2
        num_y = circle_y1 + (circle_diam - num_h)//2
        draw.text((num_x, num_y), number_str, fill=(255, 255, 255), font=circle_font)

        # Check if user guessed it
        is_revealed = (ans.lower() in lower_user_answers)
        if is_revealed:
            # measure text
            try:
                left, top, right, bottom = draw.textbbox((0, 0), ans, font=answer_font)
                r_w, r_h = right - left, bottom - top
            except:
                mask = answer_font.getmask(ans)
                r_w, r_h = mask.size

            text_x = box_x + (box_width - r_w)//2
            text_y = box_y + (box_height - r_h)//2
            draw.text((text_x, text_y), ans, fill=txt_color, font=answer_font)
        else:
            # If not revealed, paste the okra image (if loaded)
            if okra_icon:
                okra_w, okra_h = okra_icon.size
                okra_x = box_x + (box_width - okra_w)//2
                okra_y = box_y + (box_height - okra_h)//2
                img.paste(okra_icon, (okra_x, okra_y), okra_icon)
            else:
                # fallback: show ??? if no image loaded
                hidden_text = "???"
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), hidden_text, font=answer_font)
                    h_w, h_h = right - left, bottom - top
                except:
                    mask = answer_font.getmask(hidden_text)
                    h_w, h_h = mask.size
                h_text_x = box_x + (box_width - h_w)//2
                h_text_y = box_y + (box_height - h_h)//2
                draw.text((h_text_x, h_text_y), hidden_text, fill=txt_color, font=answer_font)

    # 4) Overlay red X's if num_of_xs > 0
    if num_of_xs > 0:
        try:
            x_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 800)
        except:
            x_font = ImageFont.load_default()

        x_text = "X"
        try:
            lx, ty, rx, by = draw.textbbox((0, 0), x_text, font=x_font)
            x_w, x_h = rx - lx, by - ty
        except:
            mask = x_font.getmask(x_text)
            x_w, x_h = mask.size

        total_strikes_width = (num_of_xs - 1) * int(1.2 * x_w) + x_w
        start_x = (width - total_strikes_width)//2
        x_y = (height - x_h) // 2

        for i in range(num_of_xs):
            x_x = start_x + i * int(1.2 * x_w)
            draw.text((x_x, x_y), x_text, font=x_font, fill=(255, 0, 0))

    # Convert to bytes
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    image_mxc = upload_image_to_matrix(img_buffer.read())
    return image_mxc, width, height


import difflib
import nltk
from nltk.corpus import wordnet as wn
from metaphone import doublemetaphone
nltk.data.path.append("./nltk_data")


def word_similarity(guess, answer):
    guess = guess.lower().strip()
    answer = answer.lower().strip()

    if not guess or not answer:
        return 0.0

    if guess == answer:
        return 1.0

    seq_similarity = difflib.SequenceMatcher(None, guess, answer).ratio()
    first_letter_bonus = 0.1 if guess[0] == answer[0] else 0
    last_letter_bonus = 0.1 if guess[-1] == answer[-1] else 0

    max_len = max(len(guess), len(answer))
    len_diff = abs(len(guess) - len(answer))
    length_similarity = max(0, 1 - (len_diff / max_len)) * 0.2

    guess_phonetic = doublemetaphone(guess)
    answer_phonetic = doublemetaphone(answer)
    phonetic_match = 0.2 if any(g == a and g != '' for g in guess_phonetic for a in answer_phonetic) else 0

    guess_synonyms = {lemma.name().lower() for syn in wn.synsets(guess) for lemma in syn.lemmas()}
    synonym_match = 0.2 if answer in guess_synonyms else 0

    score = (seq_similarity * 0.5) + first_letter_bonus + last_letter_bonus + length_similarity + phonetic_match + synonym_match
    return round(min(score, 1.0), 3)


def ask_dictionary_challenge(winner):    
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner
   
    num_of_xs = 0
    correct_guesses = 0
    user_correct_answers = {}  # Initialize dictionary to track correct answers per user


    dictionary_gifs = [
    "https://triviabotwebsite.s3.us-east-2.amazonaws.com/wordnerd/wordnerd_nerds.gif",
    "https://triviabotwebsite.s3.us-east-2.amazonaws.com/wordnerd/wordnerd_urkel.gif",
    "https://triviabotwebsite.s3.us-east-2.amazonaws.com/wordnerd/wordnerd_simpsons.gif"
    ]

    dictionary_gif_url = random.choice(dictionary_gifs)
    message = f"🤓📚 Word Nerd\n"
    image_mxc, image_width, image_height = download_image_from_url(dictionary_gif_url)
    send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)
    send_message(target_room_id, message)
    time.sleep(3)
    message = f"\n5️⃣🥇 Let's do a best of 5...\n"
    message += f"\n😏🎯 I'll take the best answer per person.\n"
    send_message(target_room_id, message)
    time.sleep(3)

    dictionary_num = 1
    while dictionary_num <= 5:
        try:
            recent_dictionary_ids = get_recent_question_ids_from_mongo("dictionary")

            # Fetch wheel of fortune questions using the random subset method
            dictionary_collection = db["dictionary_questions"]
            pipeline_dictionary = [
                {
                    "$match": {
                        "_id": {"$nin": list(recent_dictionary_ids)}
                    }
                },
                {"$sample": {"size": 100}},  # Sample a larger set first
                {
                    "$group": {  
                        "_id": "$question",
                        "question_doc": {"$first": "$$ROOT"}
                    }
                },
                {"$replaceRoot": {"newRoot": "$question_doc"}},  
                {"$sample": {"size": 1}}  # Sample 1 unique question
            ]

            dictionary_questions = list(dictionary_collection.aggregate(pipeline_dictionary))
            dictionary_question = dictionary_questions[0]
            dictionary_word = dictionary_question["word"]
            pattern = re.compile(re.escape(dictionary_word), re.IGNORECASE)
            dictionary_definition = dictionary_question["definition"]
            dictionary_definition = pattern.sub("OKRA", dictionary_definition)
            dictionary_question_id = dictionary_question["_id"] 
            dictionary_category = "Dictionary"
            dictionary_url = ""
            

            print(f"Word {dictionary_num}: {dictionary_word}")
            print(f"Definition {dictionary_num}: {dictionary_definition}")

            if dictionary_question_id:
                store_question_ids_in_mongo([dictionary_question_id], "dictionary")  # Store it as a list containing a single ID

        except Exception as e:
            sentry_sdk.capture_exception(e)
            error_details = traceback.format_exc()
            print(f"Error selecting dictionary questions: {e}\nDetailed traceback:\n{error_details}")
            return None  # Return an empty list in case of failure

        processed_events = set()  # Track processed event IDs to avoid duplicates        

        word_first_char = dictionary_word[0]
        word_length = len(dictionary_word)
        message = f"\n⚠️🚨 Everyone's in!\n"
        message += f"\n🧠❓ Word {dictionary_num}/5\n"
        message += f"\n🔤 Starts with {word_first_char.upper()}"
        message += f"\n🔢 {word_length} characters\n"
                
        send_message(target_room_id, message)
        time.sleep(3)
        message = f"\n📘📝 Definition: {dictionary_definition}\n"
        message += f"\n🟢💨 GO!\n"   
        send_message(target_room_id, message)

        initialize_sync()
        start_time = time.time()  # Track when the question starts
        message_content = ""
        right_answer = False
        winner_name = ""
        winner_score = ""
        closest_guesses = []  # Track (display_name, guess, score)
        
        while time.time() - start_time < 20 and right_answer == False:
            try:                                                      
                if since_token:
                    params["since"] = since_token
    
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
    
                        if sender == bot_user_id:
                            continue
    
                        sender_display_name = get_display_name(sender)
                        message_content = event.get("content", {}).get("body", "")
                        
                        similarity_score = word_similarity(message_content, dictionary_word)

                        # Track all guesses for later ranking
                        closest_guesses.append((sender_display_name, message_content, similarity_score))

                        if similarity_score == 1:
                            message = f"\n✅🎉 Correct! @{sender_display_name} got it! {dictionary_word.upper()}\n"
                            send_message(target_room_id, message)
                            right_answer = True
                
                            if sender_display_name not in user_correct_answers:
                                user_correct_answers[sender_display_name] = 0
                            user_correct_answers[sender_display_name] += 1
                        
            except Exception as e:
                print(f"Error processing events: {e}")
        
        if right_answer == False:    
            message = f"\n❌😢 No one got it.\n\nAnswer: {dictionary_word.upper()}\n"
            send_message(target_room_id, message)
            message = ""
            if closest_guesses:
            # Sort and show top 3 closest guesses
                closest_guesses.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity score
            
                # Keep only the highest-scoring guess per user
                best_guesses = {}
                for user, guess, score in closest_guesses:
                    if user not in best_guesses or score > best_guesses[user][1]:
                        best_guesses[user] = (guess, score)
                
                # Convert to list and sort by score descending
                deduped_guesses = [(user, guess, score) for user, (guess, score) in best_guesses.items()]
                deduped_guesses.sort(key=lambda x: x[2], reverse=True)
                
                # Show top 3
                top_n = 3
                message += "\n🔍 Top 3 least worst guesses:\n"
                for i, (user, guess, score) in enumerate(deduped_guesses[:top_n], 1):
                    point_str = f"{score:.2f}"
                    message += f"{i}. @{user}: \"{guess}\" — score: {point_str}\n"
                
                send_message(target_room_id, message)
                
                message = f"\n🥈🤡 50% credit for your 'effort'.\n"
                send_message(target_room_id, message)
                
                # Award fractional points to top 3
                for user, _, score in deduped_guesses[:top_n]:
                    if user not in user_correct_answers:
                        user_correct_answers[user] = 0
                    user_correct_answers[user] += score * 0.5
            
        time.sleep(2)

        dictionary_num = dictionary_num + 1
                        
        # Sort the dictionary by the count (value) in descending order
        message = ""
        sorted_users = sorted(user_correct_answers.items(), key=lambda x: x[1], reverse=True)
        if sorted_users:
            if dictionary_num > 5:
                message += "\n🏁🏆 Final Standings\n"
            else:   
                message += "\n📊🏆 Current Standings\n"
            winner_name, winner_score = sorted_users[0]


        for counter, (user, count) in enumerate(sorted_users, start=1):
            message += f"{counter}. @{user}: {count:.2f}\n"
            
        send_message(target_room_id, message)
        
    time.sleep(2)
    message = f"\n🎉🥇 The winner is @{winner_name}!\n"
    send_message(target_room_id, message)
    
    wf_winner = True
    time.sleep(3)
    return None




def ask_riddle_challenge(winner):    
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner
   
    num_of_xs = 0
    correct_guesses = 0
    user_correct_answers = {}  # Initialize dictionary to track correct answers per user


    riddler_gifs = [
    "https://triviabotwebsite.s3.us-east-2.amazonaws.com/riddler/riddler-carey.gif",
    "https://triviabotwebsite.s3.us-east-2.amazonaws.com/riddler/riddler-vintage.gif",
    "https://triviabotwebsite.s3.us-east-2.amazonaws.com/riddler/riddler-cartoon.gif"
    ]

    riddler_gif_url = random.choice(riddler_gifs)
    message = f"🟢🎩 The Riddler\n"
    image_mxc, image_width, image_height = download_image_from_url(riddler_gif_url)
    send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)
    send_message(target_room_id, message)
    time.sleep(3)
    message = f"\n5️⃣🥇 Let's do a best of 5...\n"
    send_message(target_room_id, message)
    time.sleep(3)

    riddle_num = 1
    while riddle_num <= 5:
        try:
            recent_riddle_ids = get_recent_question_ids_from_mongo("riddle")

            # Fetch wheel of fortune questions using the random subset method
            riddle_collection = db["riddle_questions"]
            pipeline_riddle = [
                {
                    "$match": {
                        "_id": {"$nin": list(recent_riddle_ids)},
                        "enabled": "1"  # Ensure only enabled riddles are included
                    }
                },
                {"$sample": {"size": 100}},  # Sample a larger set first
                {
                    "$group": {  
                        "_id": "$question",
                        "question_doc": {"$first": "$$ROOT"}
                    }
                },
                {"$replaceRoot": {"newRoot": "$question_doc"}},  
                {"$sample": {"size": 1}}  # Sample 1 unique question
            ]

            riddle_questions = list(riddle_collection.aggregate(pipeline_riddle))
            riddle_question = riddle_questions[0]
            riddle_text = riddle_question["question"]
            riddle_answers = riddle_question["answers"]
            riddle_main_answer = riddle_answers[0]
            riddle_category = riddle_question["category"]
            riddle_url = riddle_question["url"]
            riddle_question_id = riddle_question["_id"] 
            print(f"Category {riddle_num}: {riddle_category}")
            print(f"Riddle {riddle_num}: {riddle_text}")
            print(f"Answer {riddle_num}: {riddle_main_answer}")

            if riddle_question_id:
                store_question_ids_in_mongo([riddle_question_id], "riddle")  # Store it as a list containing a single ID

        except Exception as e:
            sentry_sdk.capture_exception(e)
            error_details = traceback.format_exc()
            print(f"Error selecting riddle questions: {e}\nDetailed traceback:\n{error_details}")
            return None  # Return an empty list in case of failure

        processed_events = set()  # Track processed event IDs to avoid duplicates        
            
        message = f"\n⚠️🚨 Everyone's in!\n"
        time.sleep(2)
        message += f"\n🧠❓ Riddle {riddle_num}/5: {riddle_text}"       
        send_message(target_room_id, message)

        initialize_sync()
        start_time = time.time()  # Track when the question starts
        message_content = ""
        right_answer = False
        winner_name = ""
        winner_score = ""
        
        while time.time() - start_time < 20 and right_answer == False:
            try:                                                      
                if since_token:
                    params["since"] = since_token
    
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
    
                        if sender == bot_user_id:
                            continue
    
                        sender_display_name = get_display_name(sender)
                        message_content = event.get("content", {}).get("body", "")

                        for answer in riddle_answers:
                        
                            if fuzzy_match(message_content, answer, riddle_category, riddle_url):
                                message = f"\n✅🎉 Correct! @{sender_display_name} got it! {answer.upper()}\n"
                                send_message(target_room_id, message)
                                right_answer = True
    
                                # Update user-specific correct answer count
                                if sender_display_name not in user_correct_answers:
                                    user_correct_answers[sender_display_name] = 0
                                    
                                user_correct_answers[sender_display_name] += 1
                        
            except Exception as e:
                print(f"Error processing events: {e}")
        
        if right_answer == False:    
            message = f"\n❌😢 No one got it.\n\nAnswer: {riddle_main_answer.upper()}\n"
            send_message(target_room_id, message)
        
        time.sleep(2)

        riddle_num = riddle_num + 1
                        
        # Sort the dictionary by the count (value) in descending order
        message = ""
        sorted_users = sorted(user_correct_answers.items(), key=lambda x: x[1], reverse=True)
        if sorted_users:
            if riddle_num > 5:
                message += "\n🏁🏆 Final Standings\n"
            else:   
                message += "\n📊🏆 Current Standings\n"
            winner_name, winner_score = sorted_users[0]


        for counter, (user, count) in enumerate(sorted_users, start=1):
            message += f"{counter}. @{user}: {count}\n"
            
        send_message(target_room_id, message)
        
    time.sleep(2)
    message = f"\n🎉🥇 The winner is @{winner_name}!\n"
    send_message(target_room_id, message)
    
    wf_winner = True
    time.sleep(3)
    return None



def ask_animal_challenge(winner):    
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner
   
    num_of_xs = 0
    correct_guesses = 0
    user_correct_answers = {}  # Initialize dictionary to track correct answers per user

    message = f"\n❓🦓 Name that OkrAnimal!!\n"
    send_message(target_room_id, message)
    time.sleep(5)
    
    animal_main_url= "https://a-z-animals.com/animals/"
    animal_category  = "Animals"

    counter = 1
    while num_of_xs < 3 and correct_guesses < 20:
        try:
            recent_animal_ids = get_recent_question_ids_from_mongo("animal")

            # Fetch wheel of fortune questions using the random subset method
            animal_collection = db["animal_questions"]
            pipeline_animal = [
                {"$match": {"_id": {"$nin": list(recent_animal_ids)}}},  # Exclude recent IDs
                {"$sample": {"size": 100}},  # Sample a larger set first
                {"$group": {  
                    "_id": "$question",
                    "question_doc": {"$first": "$$ROOT"}
                }},
                {"$replaceRoot": {"newRoot": "$question_doc"}},  
                {"$sample": {"size": 1}}  # Sample 1 unique question
            ]

            animal_questions = list(animal_collection.aggregate(pipeline_animal))
            animal_question = animal_questions[0]
            
            animal_detail_url = animal_question["animal_url"]   
            animal_image_url = animal_question["image_url"]   
            animal_name = animal_question["name"]
            animal_question_id = animal_question["_id"] 
            
            # Fields to clean and redact
            taxonomy_fields = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
            taxonomy_data = {}
            
            # Case-insensitive pattern for the animal name
            pattern = re.compile(re.escape(animal_name), re.IGNORECASE)
            
            for field in taxonomy_fields:
                raw_value = animal_question.get(field) or "N/A"
                redacted_value = pattern.sub("OKRA", raw_value)
                taxonomy_data[field] = redacted_value
            
            # Extract individual fields if needed
            animal_kingdom = taxonomy_data["kingdom"]
            animal_phylum = taxonomy_data["phylum"]
            animal_class = taxonomy_data["class"]
            animal_order = taxonomy_data["order"]
            animal_family = taxonomy_data["family"]
            animal_genus = taxonomy_data["genus"]
            animal_species = taxonomy_data["species"]
            print(f"Animal Name: {animal_name}")

            if animal_question_id:
                store_question_ids_in_mongo([animal_question_id], "animal")  # Store it as a list containing a single ID

        except Exception as e:
            sentry_sdk.capture_exception(e)
            error_details = traceback.format_exc()
            print(f"Error selecting animal questions: {e}\nDetailed traceback:\n{error_details}")
            return None  # Return an empty list in case of failure

        processed_events = set()  # Track processed event IDs to avoid duplicates        
        animal_mxc, animal_width, animal_height = download_image_from_url(animal_image_url)
        animal_size = 100

        start_message = ""
        
        if num_of_xs == 0:
            start_message += f"\n🟩🤔 Okrans, you have 0/3 strikes.\n"
        elif num_of_xs == 1:
            start_message += f"\n🟨🤔 Okrans, you have 1/3 strikes...\n"
        elif num_of_xs == 2:
            start_message += f"\n🟥🤔 Okrans, you have 2/3 strikes!\n"
       
        if correct_guesses > 0:
            start_message += f"\nCorrect guesses: {correct_guesses}\n"

        send_message(target_room_id, start_message)
        time.sleep(2)

        image_response = send_image(target_room_id, animal_mxc, animal_width, animal_height, animal_size)

        if image_response == False:
            error_message = f"\n⚠️🚨 Reddit is preventing a image of {animal_name.upper()}.\n"
            error_message += f"\n🔄🤔 Let's try a different one.\n"
            send_message(target_room_id, error_message)
            continue
            
        message = f"\n⚠️🚨 Everyone's in!\n"
        message += f"\n❓🦓 The $@!# is dat?!?\n"
        message += f"\n👑🏰 Kingdom: {animal_kingdom}"
        message += f"\n🧩🧱 Phylum: {animal_phylum}"
        message += f"\n🏫📚 Class: {animal_class}"
        message += f"\n🧾🔢 Order: {animal_order}"
        message += f"\n👨‍👩‍👧‍👦🏡 Family: {animal_family}"        
        send_message(target_room_id, message)
        
        message = f"🔬🧪 Genus: {animal_genus}"
        message += f"\n🐾🧍 Species: {animal_species}"
        send_message(target_room_id, message)

        initialize_sync()
        start_time = time.time()  # Track when the question starts
        message_content = ""
        right_answer = False
        
        while time.time() - start_time < 15 and right_answer == False:
            try:                                                      
                if since_token:
                    params["since"] = since_token
    
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
    
                        if sender == bot_user_id:
                            continue
    
                        sender_display_name = get_display_name(sender)
                        message_content = event.get("content", {}).get("body", "")
                        
                       
                        if fuzzy_match(message_content, animal_name, animal_category, animal_detail_url):
                            message = f"\n✅🎉 Correct! @{sender_display_name} got it! {animal_name.upper()}"
                            send_message(target_room_id, message)
                            right_answer = True
                            correct_guesses = correct_guesses + 1

                            # Update user-specific correct answer count
                            if sender_display_name not in user_correct_answers:
                                user_correct_answers[sender_display_name] = 0
                                
                            user_correct_answers[sender_display_name] += 1
                        
            except Exception as e:
                print(f"Error processing events: {e}")
        
        message = ""
        if right_answer == False:    
            message += f"❌😢 No one got it.\n\nAnswer: {animal_name.upper()}\n"
            num_of_xs = num_of_xs + 1
        
        message += f"\n{animal_detail_url}"
        send_message(target_room_id, message)
        time.sleep(2)
                        
    if correct_guesses == 0:
        message = f"\n👎😢 No right answers. I'm ashamed to call you Okrans.\n"
    else:
        if correct_guesses >= 20:
            message = f"\n✅✌️ {correct_guesses} right! I get the point. Let's move on.\n"
        else:
            message = f"\n🎉✅ Congrats Okrans! you got {correct_guesses} right!\n"
        message += "\n 🏆 Commendable Okrans\n"

        # Sort the dictionary by the count (value) in descending order
        sorted_users = sorted(user_correct_answers.items(), key=lambda x: x[1], reverse=True)
    
        for counter, (user, count) in enumerate(sorted_users, start=1):
            message += f"{counter}. @{user}: {count}\n"
        
    send_message(target_room_id, message)
    time.sleep(2)
    
    message = f"\nSee more cute animals."
    message += f"\n{animal_main_url}\n"
    send_message(target_room_id, message)
    
    wf_winner = True
    time.sleep(3)
    return None


def ask_ranker_people_challenge(winner):    
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner
   
    num_of_xs = 0
    correct_guesses = 0
    user_correct_answers = {}  # Initialize dictionary to track correct answers per user

    message = f"\n👤🌟 ID Ranker.com's All Time Greats\n"
    send_message(target_room_id, message)
    time.sleep(3)
    
    ranker_url= "https://www.ranker.com/crowdranked-list/the-most-influential-people-of-all-time"
    people_category  = "Ranker People"
    
    while num_of_xs < 3 and correct_guesses < 20:
        try:
            recent_people_ids = get_recent_question_ids_from_mongo("people")

            # Fetch wheel of fortune questions using the random subset method
            people_collection = db["ranker_people_questions"]
            pipeline_people = [
                {"$match": {"_id": {"$nin": list(recent_people_ids)}}},  # Exclude recent IDs
                {"$sample": {"size": 100}},  # Sample a larger set first
                {"$group": {  
                    "_id": "$question",
                    "question_doc": {"$first": "$$ROOT"}
                }},
                {"$replaceRoot": {"newRoot": "$question_doc"}},  
                {"$sample": {"size": 1}}  # Sample 1 unique question
            ]

            people_questions = list(people_collection.aggregate(pipeline_people))
            people_question = people_questions[0]
            people_rank = people_question["rank"]
            people_answers = people_question["answers"]   
            people_main_answer = people_answers[0]
            people_detail_url = people_question["detail_url"]
            people_birthplace = people_question["birthplace"]
            people_nationality = people_question["nationality"]
            people_profession = people_question["profession"]
            people_question_id = people_question["_id"]  # Get the ID of the selected question
            people_url= people_question["url"]  # Get the ID of the selected question
            
            if people_question_id:
                store_question_ids_in_mongo([people_question_id], "people")  # Store it as a list containing a single ID
            print(people_question)

        except Exception as e:
            sentry_sdk.capture_exception(e)
            error_details = traceback.format_exc()
            print(f"Error selecting people questions: {e}\nDetailed traceback:\n{error_details}")
            return None  # Return an empty list in case of failure

        processed_events = set()  # Track processed event IDs to avoid duplicates        
        people_mxc, people_width, people_height = download_image_from_url(people_url)
        people_size = 100

        start_message = ""
        
        if num_of_xs == 0:
            start_message += f"\n🟩🤔 Okrans, you have 0/3 strikes.\n"
        elif num_of_xs == 1:
            start_message += f"\n🟨🤔 Okrans, you have 1/3 strikes...\n"
        elif num_of_xs == 2:
            start_message += f"\n🟥🤔 Okrans, you have 2/3 strikes!\n"
       
        if correct_guesses > 0:
            start_message += f"\nCorrect guesses: {correct_guesses}\n"

        send_message(target_room_id, start_message)
        time.sleep(2)
            
        message = f"\n⚠️🚨 Everyone's in!\n"
        message += f"\n🎥🌟 #{people_rank}: Who dat?!?\n"
        message += f"\n🌍🏡 Birthplace: {people_birthplace}"
        message += f"\n🏳️🆔 Nationality: {people_nationality}"
        message += f"\n💼⚒️ Profession: {people_profession}\n"
        
        image_response = send_image(target_room_id, people_mxc, people_width, people_height, people_size)

        if image_response == False:
            error_message = f"\n⚠️🚨 Reddit is preventing a image of {people_main_answer.upper()}.\n"
            error_message += f"\n🔄🤔 Let's try a different one.\n"
            send_message(target_room_id, error_message)
            continue
        
        send_message(target_room_id, message)

        initialize_sync()
        start_time = time.time()  # Track when the question starts
        message_content = ""
        right_answer = False
        
        while time.time() - start_time < 15 and right_answer == False:
            try:                                                      
                if since_token:
                    params["since"] = since_token
    
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
    
                        if sender == bot_user_id:
                            continue
    
                        sender_display_name = get_display_name(sender)
                        message_content = event.get("content", {}).get("body", "")
                        
                        for answer in people_answers:
                            if fuzzy_match(message_content, answer, people_category, people_url):
                                message = f"\n✅🎉 Correct! @{sender_display_name} got it! {answer.upper()}"
                                send_message(target_room_id, message)
                                right_answer = True
                                correct_guesses = correct_guesses + 1

                                # Update user-specific correct answer count
                                if sender_display_name not in user_correct_answers:
                                    user_correct_answers[sender_display_name] = 0
                                user_correct_answers[sender_display_name] += 1
                                
                                break   
                        
                        if right_answer == True:
                            break

                    if right_answer == True:
                        break
                        
            except Exception as e:
                print(f"Error processing events: {e}")
        
        message = ""
        if right_answer == False:    
            message += f"❌😢 No one got it.\n\nAnswer: {people_answers[0].upper()}\n"
        
            if int(people_rank) > 2500:
                message += f"\n🆓✅ No penalty for 2500+.\n"
            else:
                num_of_xs = num_of_xs + 1
        
        message += f"\n{people_detail_url}"
        send_message(target_room_id, message)
        time.sleep(2)
                        
    if correct_guesses == 0:
        message = f"\n👎😢 No right answers. I'm ashamed to call you Okrans.\n"
    else:
        if correct_guesses >= 20:
            message = f"\n✅✌️ {correct_guesses} right! I get the point. Let's move on.\n"
        else:
            message = f"\n🎉✅ Congrats Okrans! you got {correct_guesses} right!\n"
        message += "\n 🏆 Commendable Okrans\n"

        # Sort the dictionary by the count (value) in descending order
        sorted_users = sorted(user_correct_answers.items(), key=lambda x: x[1], reverse=True)
    
        for counter, (user, count) in enumerate(sorted_users, start=1):
            message += f"{counter}. @{user}: {count}\n"
        
    send_message(target_room_id, message)
    time.sleep(2)
    
    message = f"\n Ranks from Ranker.com\n"
    message += f"\n{ranker_url}\n"
    send_message(target_room_id, message)
    
    wf_winner = True
    time.sleep(3)
    return None




def ask_flags_challenge(winner):    
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner
   
    num_of_xs = 0
    correct_guesses = 0
    user_correct_answers = {}  # Initialize dictionary to track correct answers per user

    flags_gifs = [
    "https://triviabotwebsite.s3.us-east-2.amazonaws.com/flagsnet/flags_usc.gif",
    #"https://triviabotwebsite.s3.us-east-2.amazonaws.com/flagsnet/flags_cartoon.gif",
    #"https://triviabotwebsite.s3.us-east-2.amazonaws.com/flagsnet/flags_friends.gif"
    ]

    flags_gif_url = random.choice(flags_gifs)
    message = f"🎏🎉 Flag Fest\n"
    image_mxc, image_width, image_height = download_image_from_url(flags_gif_url)
    send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)
    send_message(target_room_id, message)
    time.sleep(3)
    
    while num_of_xs < 3 and correct_guesses < 20:
        try:
            recent_flags_ids = get_recent_question_ids_from_mongo("flags")

            # Fetch wheel of fortune questions using the random subset method
            flags_collection = db["flags_questions"]
            pipeline_flags = [
                {"$match": {"_id": {"$nin": list(recent_flags_ids)}}},
                {"$sample": {"size": 20}},  # sample 50 first
                {"$group": {
                    "_id": "$question",
                    "question_doc": {"$first": "$$ROOT"}
                }},
                {"$replaceRoot": {"newRoot": "$question_doc"}},
                {"$sample": {"size": 1}}  # optional, if you want to randomize final pick again
            ]

            flags_questions = list(flags_collection.aggregate(pipeline_flags))
            flags_question = flags_questions[0]
            flags_category = flags_question["category"]
            flags_answer = flags_question["answer"]   
            flags_detail = flags_question["flag_detail"]   
            flags_source_url = flags_question["source_url"]
            flags_url = flags_question["flag_url"]
            flags_question_id = flags_question["_id"]  # Get the ID of the selected question
            
            if flags_question_id:
                store_question_ids_in_mongo([flags_question_id], "flags")  # Store it as a list containing a single ID
            print(flags_question)

        except Exception as e:
            sentry_sdk.capture_exception(e)
            error_details = traceback.format_exc()
            print(f"Error selecting flags questions: {e}\nDetailed traceback:\n{error_details}")
            return None  # Return an empty list in case of failure

        processed_events = set()  # Track processed event IDs to avoid duplicates        
        flags_mxc, flags_width, flags_height = download_image_from_url(flags_url)
        flags_size = 100

        start_message = ""
        
        if num_of_xs == 0:
            start_message += f"\n🟩🤔 Okrans, you have 0/3 strikes.\n"
        elif num_of_xs == 1:
            start_message += f"\n🟨🤔 Okrans, you have 1/3 strikes...\n"
        elif num_of_xs == 2:
            start_message += f"\n🟥🤔 Okrans, you have 2/3 strikes!\n"
       
        if correct_guesses > 0:
            start_message += f"\nCorrect guesses: {correct_guesses}\n"

        send_message(target_room_id, start_message)
        time.sleep(2)

        message = f"\n⚠️🚨 Everyone's in!\n"
        if flags_category == "country_region_org":
            message += f"\n🌍🏳️ Which country or international organization does this flag represent?\n"
            message += f"\n✋🔤 Do NOT abbreviate countries.\n"
        elif flags_category == "signal":
            message += f"\n🔣🏳️ What symbol does this flag represent?\n"
            
        image_response = send_image(target_room_id, flags_mxc, flags_width, flags_height, flags_size)

        if image_response == False:
            error_message = f"\n⚠️🚨 Reddit is preventing a poster from {flags_answer.upper()}.\n"
            error_message += f"\n🔄🤔 Let's try a different one.\n"
            send_message(target_room_id, error_message)
            continue
        
        send_message(target_room_id, message)

        initialize_sync()
        start_time = time.time()  # Track when the question starts
        message_content = ""
        right_answer = False
        
        while time.time() - start_time < 15 and right_answer == False:
            try:                                                      
                if since_token:
                    params["since"] = since_token
    
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
    
                        if sender == bot_user_id:
                            continue
    
                        sender_display_name = get_display_name(sender)
                        message_content = event.get("content", {}).get("body", "")
                        
                        if fuzzy_match(message_content, flags_answer, flags_category, flags_url):
                            message = f"\n✅🎉 Correct! @{sender_display_name} got it! {flags_answer.upper()}\n"
                            message += f"🏴‍☠️📖 Details: {flags_detail}\n"
                            message += f"\n👀➡️ See more: {flags_source_url}\n"
                            send_message(target_room_id, message)
                            right_answer = True
                            correct_guesses = correct_guesses + 1

                            # Update user-specific correct answer count
                            if sender_display_name not in user_correct_answers:
                                user_correct_answers[sender_display_name] = 0
                            user_correct_answers[sender_display_name] += 1
                            
                        if right_answer == True:
                            break

                    if right_answer == True:
                        break
                        
            except Exception as e:
                print(f"Error processing events: {e}")
        
        if right_answer == False:    
            num_of_xs = num_of_xs + 1
            message = f"\n❌😢 No one got it.\n\nAnswer: {flags_answer.upper()}\n"
            message += f"🏴‍☠️📖 Details: {flags_detail}\n"
            message += f"\n👀➡️ See more: {flags_source_url}\n"
            send_message(target_room_id, message)
            time.sleep(1)
                        
    if correct_guesses == 0:
        message = f"\n👎😢 No right answers. I'm ashamed to call you Okrans.\n"
    else:
        if correct_guesses >= 20:
            message = f"\n✅✌️ {correct_guesses} right! I get the point. Let's move on.\n"
        else:
            message = f"\n🎉✅ Congrats Okrans! you got {correct_guesses} right!\n"
        message += "\n 🏆 Commendable Okrans\n"

        # Sort the dictionary by the count (value) in descending order
        sorted_users = sorted(user_correct_answers.items(), key=lambda x: x[1], reverse=True)
    
        for counter, (user, count) in enumerate(sorted_users, start=1):
            message += f"{counter}. @{user}: {count}\n"
        
    send_message(target_room_id, message)
    time.sleep(2)
    message = f"\n👀➡️ See more flags...if you want: http://flags.net/\n"
    send_message(target_room_id, message)
    wf_winner = True
    time.sleep(3)
    return None








def ask_poster_challenge(winner):    
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner
   
    num_of_xs = 0
    correct_guesses = 0
    user_correct_answers = {}  # Initialize dictionary to track correct answers per user
    
    while num_of_xs < 3 and correct_guesses < 20:
        try:
            recent_posters_ids = get_recent_question_ids_from_mongo("posters")

            # Fetch wheel of fortune questions using the random subset method
            posters_collection = db["posters_questions"]
            pipeline_posters = [
                {"$match": {"_id": {"$nin": list(recent_posters_ids)}}},  # Exclude recent IDs
                {"$group": {  # Group by question text to ensure uniqueness
                    "_id": "$question",  # Group by the question text field
                    "question_doc": {"$first": "$$ROOT"}  # Select the first document with each unique text
                }},
                {"$replaceRoot": {"newRoot": "$question_doc"}},  # Flatten the grouped results
                {"$sample": {"size": 1}}  # Sample 1 unique question
            ]

            posters_questions = list(posters_collection.aggregate(pipeline_posters))
            posters_question = posters_questions[0]
            posters_category = posters_question["category"]
            posters_answers = posters_question["answers"]   
            posters_main_answer = posters_answers[0]
            posters_year = posters_question["question"]
            posters_url = posters_question["url"]
            posters_question_id = posters_question["_id"]  # Get the ID of the selected question
            
            if posters_question_id:
                store_question_ids_in_mongo([posters_question_id], "posters")  # Store it as a list containing a single ID
            print(posters_question)

        except Exception as e:
            sentry_sdk.capture_exception(e)
            error_details = traceback.format_exc()
            print(f"Error selecting posters questions: {e}\nDetailed traceback:\n{error_details}")
            return None  # Return an empty list in case of failure

        poster_category_emojis = get_category_title(posters_category, "")
        processed_events = set()  # Track processed event IDs to avoid duplicates        
        posters_mxc, posters_width, posters_height = download_image_from_url(posters_url)
        posters_size = 100

        start_message = ""
        
        if num_of_xs == 0:
            start_message += f"\n🟩🤔 Okrans, you have 0/3 strikes.\n"
        elif num_of_xs == 1:
            start_message += f"\n🟨🤔 Okrans, you have 1/3 strikes...\n"
        elif num_of_xs == 2:
            start_message += f"\n🟥🤔 Okrans, you have 2/3 strikes!\n"
       
        if correct_guesses > 0:
            start_message += f"\nCorrect guesses: {correct_guesses}\n"

        send_message(target_room_id, start_message)
        time.sleep(2)
            
        message = f"\n⚠️🚨 Everyone's in!\n"
        message += f"\n🎥🌟 What {posters_category.upper()} is depicted in the poster above?\n"
        message += f"\n📅💡 Year: {posters_year}\n"
        image_response = send_image(target_room_id, posters_mxc, posters_width, posters_height, posters_size)

        if image_response == False:
            error_message = f"\n⚠️🚨 Reddit is preventing a poster from {posters_main_answer.upper()}.\n"
            error_message += f"\n🔄🤔 Let's try a different one.\n"
            send_message(target_room_id, error_message)
            continue
        
        send_message(target_room_id, message)

        initialize_sync()
        start_time = time.time()  # Track when the question starts
        message_content = ""
        right_answer = False
        
        while time.time() - start_time < 15 and right_answer == False:
            try:                                                      
                if since_token:
                    params["since"] = since_token
    
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
    
                        if sender == bot_user_id:
                            continue
    
                        sender_display_name = get_display_name(sender)
                        message_content = event.get("content", {}).get("body", "")
                        
                        for answer in posters_answers:
                            if fuzzy_match(message_content, answer, posters_category, posters_url):
                                message = f"\n✅🎉 Correct! @{sender_display_name} got it! {answer.upper()}"
                                send_message(target_room_id, message)
                                right_answer = True
                                correct_guesses = correct_guesses + 1

                                # Update user-specific correct answer count
                                if sender_display_name not in user_correct_answers:
                                    user_correct_answers[sender_display_name] = 0
                                user_correct_answers[sender_display_name] += 1
                                
                                break   
                        
                        if right_answer == True:
                            break

                    if right_answer == True:
                        break
                        
            except Exception as e:
                print(f"Error processing events: {e}")
        
        if right_answer == False:    
            num_of_xs = num_of_xs + 1
            message = f"\n❌😢 No one got it.\n\nAnswer: {posters_answers[0].upper()}\n"
            send_message(target_room_id, message)
            time.sleep(1)
                        
    if correct_guesses == 0:
        message = f"\n👎😢 No right answers. I'm ashamed to call you Okrans.\n"
    else:
        if correct_guesses >= 20:
            message = f"\n✅✌️ {correct_guesses} right! I get the point. Let's move on.\n"
        else:
            message = f"\n🎉✅ Congrats Okrans! you got {correct_guesses} right!\n"
        message += "\n 🏆 Commendable Okrans\n"

        # Sort the dictionary by the count (value) in descending order
        sorted_users = sorted(user_correct_answers.items(), key=lambda x: x[1], reverse=True)
    
        for counter, (user, count) in enumerate(sorted_users, start=1):
            message += f"{counter}. @{user}: {count}\n"
        
    send_message(target_room_id, message)
    wf_winner = True
    time.sleep(3)
    return None


def ask_missing_link(winner):    
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner
   
    num_of_xs = 0
    correct_guesses = 0
    user_correct_answers = {}  # Initialize dictionary to track correct answers per user

    while num_of_xs < 3 and correct_guesses < 20:
        try:
            recent_missing_link_ids = get_recent_question_ids_from_mongo("missing_link")

            # Fetch wheel of fortune questions using the random subset method
            missing_link_collection = db["missing_link_questions"]
            pipeline_missing_link = [
                {"$match": {"_id": {"$nin": list(recent_missing_link_ids)}}},  # Exclude recent IDs
                {"$group": {  # Group by question text to ensure uniqueness
                    "_id": "$question",  # Group by the question text field
                    "question_doc": {"$first": "$$ROOT"}  # Select the first document with each unique text
                }},
                {"$replaceRoot": {"newRoot": "$question_doc"}},  # Flatten the grouped results
                {"$sample": {"size": 1}}  # Sample 1 unique question
            ]

            missing_link_questions = list(missing_link_collection.aggregate(pipeline_missing_link))
            missing_link_question = missing_link_questions[0]
            
            missing_link_category = missing_link_question["category"]
            missing_link_answers = missing_link_question["answers"]   
            
            missing_link_list = missing_link_question["question"]
            missing_link_hint = missing_link_question["url"]
            missing_link_question_id = missing_link_question["_id"]  # Get the ID of the selected question
            
            if missing_link_question_id:
                store_question_ids_in_mongo([missing_link_question_id], "missing_link")  # Store it as a list containing a single ID
            print(missing_link_question)

        except Exception as e:
            sentry_sdk.capture_exception(e)
            error_details = traceback.format_exc()
            print(f"Error selecting missing_link questions: {e}\nDetailed traceback:\n{error_details}")
            return None  # Return an empty list in case of failure

        missing_link_category_emojis = get_category_title(missing_link_category, "")
        processed_events = set()  # Track processed event IDs to avoid duplicates        
        
        start_message = ""
        
        if num_of_xs == 0:
            start_message += f"\n🟩🤔 Okrans, you have 0/3 strikes.\n"
        elif num_of_xs == 1:
            start_message += f"\n🟨🤔 Okrans, you have 1/3 strikes...\n"
        elif num_of_xs == 2:
            start_message += f"\n🟥🤔 Okrans, you have 2/3 strikes!\n"
       
        if correct_guesses > 0:
            start_message += f"\nCorrect guesses: {correct_guesses}\n"

        send_message(target_room_id, start_message)
        time.sleep(2)
            
        message = f"\n⚠️🚨 Everyone's in!\n"

        if missing_link_category == "Movie Characters":
            message += f"\n🎥🌟 What MOVIE is the missing link?\n"
            list_header = "Characters"

        if  missing_link_category == "Movie Actors":
            message += f"\n🎥🌟 What MOVIE is the missing link?\n"
            list_header = "Actors / Actresses"
        
        if missing_link_category == "Actor / Actress":
            message += f"\n🎥🌟 What ACTOR or ACTRESS is the missing link?\n"
            list_header = "Movies"
            
            
        message += f"\n📅💡 Clue: {missing_link_hint}\n"
        send_message(target_room_id, message)
        time.sleep(2)

        message = f"\n{list_header}"

        formatted_list = [name.strip() for name in missing_link_list.split(",") if name.strip()]  # Split by commas, remove extra spaces
        for i, element in enumerate(formatted_list, start=1):
            message += f"\n{i}. {element}"

        message += "\n"
        send_message(target_room_id, message)

        initialize_sync()
        start_time = time.time()  # Track when the question starts
        message_content = ""
        right_answer = False
        
        while time.time() - start_time < 15 and right_answer == False:
            try:                                                      
                if since_token:
                    params["since"] = since_token
    
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
    
                        if sender == bot_user_id:
                            continue
    
                        sender_display_name = get_display_name(sender)
                        message_content = event.get("content", {}).get("body", "")
                        
                        for answer in missing_link_answers:
                            if fuzzy_match(message_content, answer, missing_link_category, missing_link_hint):
                                message = f"\n✅🎉 Correct! @{sender_display_name} got it! {answer.upper()}"
                                send_message(target_room_id, message)
                                right_answer = True
                                correct_guesses = correct_guesses + 1

                                # Update user-specific correct answer count
                                if sender_display_name not in user_correct_answers:
                                    user_correct_answers[sender_display_name] = 0
                                user_correct_answers[sender_display_name] += 1
                                
                                break   
                        
                        if right_answer == True:
                            break

                    if right_answer == True:
                        break
                        
            except Exception as e:
                print(f"Error processing events: {e}")
        
        if right_answer == False:    
            num_of_xs = num_of_xs + 1
            message = f"\n❌😢 No one got it.\n\nAnswer: {missing_link_answers[0].upper()}\n"
            send_message(target_room_id, message)
            time.sleep(1)
                        
    if correct_guesses == 0:
        message = f"\n👎😢 No right answers. I'm ashamed to call you Okrans.\n"
    else:
        if correct_guesses >= 20:
            message = f"\n✅✌️ {correct_guesses} right! I get the point. Let's move on.\n"
        else:
            message = f"\n🎉✅ Congrats Okrans! you got {correct_guesses} right!\n"
        message += "\n 🏆 Commendable Okrans\n"

        # Sort the dictionary by the count (value) in descending order
        sorted_users = sorted(user_correct_answers.items(), key=lambda x: x[1], reverse=True)
    
        for counter, (user, count) in enumerate(sorted_users, start=1):
            message += f"{counter}. @{user}: {count}\n"
        
    send_message(target_room_id, message)
    wf_winner = True
    time.sleep(3)
    return None




def ask_movie_scenes_challenge(winner):    
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner
   
    num_of_xs = 0
    correct_guesses = 0
    user_correct_answers = {}  # Initialize dictionary to track correct answers per user

    while num_of_xs < 3 and correct_guesses < 20:
        try:
            recent_movie_scenes_ids = get_recent_question_ids_from_mongo("movie_scenes")

            # Fetch wheel of fortune questions using the random subset method
            movie_scenes_collection = db["movie_scenes_questions"]
            pipeline_movie_scenes = [
                {"$match": {"_id": {"$nin": list(recent_movie_scenes_ids)}}},  # Exclude recent IDs
                {"$group": {  # Group by question text to ensure uniqueness
                    "_id": "$question",  # Group by the question text field
                    "question_doc": {"$first": "$$ROOT"}  # Select the first document with each unique text
                }},
                {"$replaceRoot": {"newRoot": "$question_doc"}},  # Flatten the grouped results
                {"$sample": {"size": 1}}  # Sample 1 unique question
            ]

            movie_scenes_questions = list(movie_scenes_collection.aggregate(pipeline_movie_scenes))
            movie_scenes_question = movie_scenes_questions[0]
            movie_scenes_category = movie_scenes_question["category"]
            movie_scenes_answers = movie_scenes_question["answers"]   
            movie_scene_main_answer = movie_scenes_answers[0]
            movie_scenes_year = movie_scenes_question["question"]
            movie_scenes_url = movie_scenes_question["url"]
            movie_scenes_question_id = movie_scenes_question["_id"]  # Get the ID of the selected question
            
            if movie_scenes_question_id:
                store_question_ids_in_mongo([movie_scenes_question_id], "movie_scenes")  # Store it as a list containing a single ID
            print(movie_scenes_question)

        except Exception as e:
            sentry_sdk.capture_exception(e)
            error_details = traceback.format_exc()
            print(f"Error selecting movie_scenes questions: {e}\nDetailed traceback:\n{error_details}")
            return None  # Return an empty list in case of failure

        movie_scenes_category_emojis = get_category_title(movie_scenes_category, "")
        processed_events = set()  # Track processed event IDs to avoid duplicates        
        movie_scenes_mxc, movie_scenes_width, movie_scenes_height = download_image_from_url(movie_scenes_url)
        movie_scenes_size = 100

        start_message = ""
        
        if num_of_xs == 0:
            start_message += f"\n🟩🤔 Okrans, you have 0/3 strikes.\n"
        elif num_of_xs == 1:
            start_message += f"\n🟨🤔 Okrans, you have 1/3 strikes...\n"
        elif num_of_xs == 2:
            start_message += f"\n🟥🤔 Okrans, you have 2/3 strikes!\n"
       
        if correct_guesses > 0:
            start_message += f"\nCorrect guesses: {correct_guesses}\n"

        send_message(target_room_id, start_message)
        time.sleep(2)
            
        message = f"\n⚠️🚨 Everyone's in!\n"
        message += f"\n🎥🌟 What {movie_scenes_category.upper()} is depicted in the scene above?\n"
        message += f"\n📅💡 Year: {movie_scenes_year}\n"
        image_response = send_image(target_room_id, movie_scenes_mxc, movie_scenes_width, movie_scenes_height, movie_scenes_size)

        if image_response == False:
            error_message = f"\n⚠️🚨 Reddit is preventing a scene from {movie_scene_main_answer.upper()}.\n"
            error_message += f"\n🔄🤔 Let's try a different one.\n"
            send_message(target_room_id, error_message)
            continue
        
        send_message(target_room_id, message)

        initialize_sync()
        start_time = time.time()  # Track when the question starts
        message_content = ""
        right_answer = False
        
        while time.time() - start_time < 15 and right_answer == False:
            try:                                                      
                if since_token:
                    params["since"] = since_token
    
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
    
                        if sender == bot_user_id:
                            continue
    
                        sender_display_name = get_display_name(sender)
                        message_content = event.get("content", {}).get("body", "")
                        
                        for answer in movie_scenes_answers:
                            if fuzzy_match(message_content, answer, movie_scenes_category, movie_scenes_url):
                                message = f"\n✅🎉 Correct! @{sender_display_name} got it! {answer.upper()}"
                                send_message(target_room_id, message)
                                right_answer = True
                                correct_guesses = correct_guesses + 1

                                # Update user-specific correct answer count
                                if sender_display_name not in user_correct_answers:
                                    user_correct_answers[sender_display_name] = 0
                                user_correct_answers[sender_display_name] += 1
                                
                                break   
                        
                        if right_answer == True:
                            break

                    if right_answer == True:
                        break
                        
            except Exception as e:
                print(f"Error processing events: {e}")
        
        if right_answer == False:    
            num_of_xs = num_of_xs + 1
            message = f"\n❌😢 No one got it.\n\nAnswer: {movie_scenes_answers[0].upper()}\n"
            send_message(target_room_id, message)
            time.sleep(1)
                        
    if correct_guesses == 0:
        message = f"\n👎😢 No right answers. I'm ashamed to call you Okrans.\n"
    else:
        if correct_guesses >= 20:
            message = f"\n✅✌️ {correct_guesses} right! I get the point. Let's move on.\n"
        else:
            message = f"\n🎉✅ Congrats Okrans! you got {correct_guesses} right!\n"
        message += "\n 🏆 Commendable Okrans\n"

        # Sort the dictionary by the count (value) in descending order
        sorted_users = sorted(user_correct_answers.items(), key=lambda x: x[1], reverse=True)
    
        for counter, (user, count) in enumerate(sorted_users, start=1):
            message += f"{counter}. @{user}: {count}\n"
        
    send_message(target_room_id, message)
    wf_winner = True
    time.sleep(3)
    return None






def ask_feud_question(winner, mode):    
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner
    
    try:
        recent_feud_ids = get_recent_question_ids_from_mongo("feud")
        
        # Fetch wheel of fortune questions using the random subset method
        feud_collection = db["feud_questions"]
        pipeline_feud = [
            {"$match": {"_id": {"$nin": list(recent_feud_ids)}}},  # Exclude recent IDs
            {"$group": {  # Group by question text to ensure uniqueness
                "_id": "$question",  # Group by the question text field
                "question_doc": {"$first": "$$ROOT"}  # Select the first document with each unique text
            }},
            {"$replaceRoot": {"newRoot": "$question_doc"}},  # Flatten the grouped results
            {"$sample": {"size": 1}}  # Sample 1 unique question
        ]

    except Exception as e:
        sentry_sdk.capture_exception(e)
        error_details = traceback.format_exc()
        print(f"Error selecting feud questions: {e}\nDetailed traceback:\n{error_details}")
        return None  # Return an empty list in case of failure

    num_of_xs = 0
    correct_guesses = 0
    user_correct_answers = {}  # Initialize dictionary to track correct answers per user
    
    try:
        feud_questions = list(feud_collection.aggregate(pipeline_feud))
        feud_question = feud_questions[0]
        feud_question_prompt = feud_question["question"]
        feud_question_answers = feud_question["answers"]   
        feud_question_category = ""
        feud_question_url = ""

        print(feud_question)
        
        feud_question_id = feud_question["_id"]  # Get the ID of the selected question
        if feud_question_id:
            store_question_ids_in_mongo([feud_question_id], "feud")  # Store it as a list containing a single ID

        win_image_mxc, win_image_width, win_image_height = download_image_from_url("https://triviabotwebsite.s3.us-east-2.amazonaws.com/harvey/harvey+win.gif")
        loss_image_mxc, loss_image_width, loss_image_height = download_image_from_url("https://triviabotwebsite.s3.us-east-2.amazonaws.com/harvey/harvey+loss.gif")
        win_image_size = 100
        loss_image_size = 100

    except Exception as e:
        sentry_sdk.capture_exception(e)
        error_details = traceback.format_exc()
        print(f"Error selecting feud questions: {e}\nDetailed traceback:\n{error_details}")
        return None  # Return an empty list in case of failure

    
    
    processed_events = set()  # Track processed event IDs to avoid duplicates  
    num_of_answers = len(feud_question_answers)
    user_progress = []
    num_of_xs = 0
    numbered_blocks = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    right_answer = False

    if mode == "solo":
        guess_time = 20
    elif mode == "cooperative":
        guess_time = 20

    if mode == "solo":
        prompt_message = f"\n⚠{numbered_blocks[num_of_answers - 1]} Top {num_of_answers} answers on the board. We asked 100 Okrans...\n"
    elif mode == "cooperative":
        prompt_message = f"\n⚠{numbered_blocks[num_of_answers - 1]} Top {num_of_answers} answers on the board. We asked 100 of you...\n"
        
    prompt_message += f"\n👉👉 {feud_question_prompt.upper()}\n"
    feud_image_size = 100
    
    while num_of_xs < 3 and right_answer == False:

        feud_image_mxc, feud_image_width, feud_image_height = create_family_feud_board_image(feud_question_answers, user_progress)
      
        if correct_guesses > 0:
            correct_message += f"\nCorrect guesses: {correct_guesses}\n"
            send_message(target_room_id, correct_message)
            
        time.sleep(1)

        send_image(target_room_id, feud_image_mxc, feud_image_width, feud_image_height, feud_image_size)
        time.sleep(3)
        send_message(target_room_id, prompt_message)
        time.sleep(2)

        start_message = ""
        if mode == "cooperative":
            start_message += f"\n⚠️🚨 Everyone's in!\n"
            
        if mode == "cooperative":
            if num_of_xs == 0:
                start_message += f"\n🟩🤔 Okrans, round 1/3...\n"
                start_message += f"\n📜🔢 List as many as you can!\n"
                start_message += f"\n🏁🚀 GO!\n"
            elif num_of_xs == 1:
                start_message += f"\n🟨🤔 Okrans, round 2/3...\n"
                start_message += f"\n📜🔢 List as many as you can!\n"
                start_message += f"\n🏁🚀 GO!\n"
            elif num_of_xs == 2:
                start_message += f"\n🟥🤔 Okrans, last round...\n"
                start_message += f"\n📜🔢 List as many as you can!\n"
                start_message += f"\n🏁🚀 GO!\n"
        if mode == "solo":
            if num_of_xs == 0:
                start_message += f"\n🟩🤔 @{winner}, round 1/3...\n"
                start_message += f"\n📜🔢 List as many as you can!\n"
                start_message += f"\n🏁🚀 GO!\n"
            elif num_of_xs == 1:
                start_message += f"\n🟨🤔 @{winner}, round 2/3...\n"
                start_message += f"\n📜🔢 List as many as you can!\n"
                start_message += f"\n🏁🚀 GO!\n"
            elif num_of_xs == 2:
                start_message += f"\n🟥🤔 @{winner}, last round...\n"
                start_message += f"\n📜🔢 List as many as you can!\n"
                start_message += f"\n🏁🚀 GO!\n"
        send_message(target_room_id, start_message)

        
        initialize_sync()
        start_time = time.time()  # Track when the question starts
        message_content = ""
        
        while time.time() - start_time < guess_time and right_answer == False:
            try:                                                      
                if since_token:
                    params["since"] = since_token
    
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
    
                        if sender == bot_user_id:
                            continue
    
                        sender_display_name = get_display_name(sender)

                        if mode == "solo" and sender_display_name != winner:
                            continue
                            
                        message_content = event.get("content", {}).get("body", "")
                        
                        
                        for answer in feud_question_answers:
                            if answer in user_progress:
                                continue
                                
                            if fuzzy_match(message_content, answer, feud_question_category, feud_question_url):
                                user_progress.append(answer)
                                
                                if len(user_progress) >= num_of_answers:
                                    right_answer = True

                                # Update user-specific correct answer count
                                if sender_display_name not in user_correct_answers:
                                    user_correct_answers[sender_display_name] = 0
                                    
                                user_correct_answers[sender_display_name] += 1
                                
                                break   
                        
                        if right_answer == True:
                            break

                    if right_answer == True:
                        break

                if right_answer == True:
                        break
                        
            except Exception as e:
                print(f"Error processing events: {e}")
        
        num_of_xs = num_of_xs + 1
        
        if right_answer == False and num_of_xs < 3:    
            message = f"\n🎤📊 Survey says...\n"
            send_message(target_room_id, message)
                        
    if mode == "cooperative":
        if len(user_progress) == 0:
            message = f"\n👎😢 No right answers out of {num_of_answers}. I'm ashamed to call you Okrans.\n"
        elif len(user_progress) < num_of_answers:
            message = f"\n🙄😒 Wow...you got {len(user_progress)}/{num_of_answers}.\n"
        else:
            message = f"\n🎉✅ Congrats Okrans! You got all {num_of_answers} right!\n"

        if len(user_progress) > 0:
            message += "\n 🏆 Commendable Okrans\n"

            # Sort the dictionary by the count (value) in descending order
            sorted_users = sorted(user_correct_answers.items(), key=lambda x: x[1], reverse=True)
        
            for counter, (user, count) in enumerate(sorted_users, start=1):
                message += f"{counter}. @{user}: {count}\n"
            
    elif mode == "solo":
        if len(user_progress) == 0:
            message = f"\n👎😢 No right answers out of {num_of_answers}. @{winner}, you're no Okran.\n"
        elif len(user_progress) < num_of_answers:
            message = f"\n🙄😒 Wow @{winner}...you got {len(user_progress)}/{num_of_answers}.\n"
        else:
            message = f"\n🎉✅ Congrats @{winner}! You got all {num_of_answers} right!\n"

    final_feud_image_mxc, final_feud_image_width, final_feud_image_height = create_family_feud_board_image(feud_question_answers, user_progress, 0)

    answer_message = f"\n🔑❓ REVEALED: {feud_question_prompt.upper()}\n"
    send_image(target_room_id, final_feud_image_mxc, final_feud_image_width, final_feud_image_height, feud_image_size)
    
    if len(user_progress) < num_of_answers:
        answer_feud_image_mxc, answer_feud_image_width, answer_feud_image_height = create_family_feud_board_image(feud_question_answers, feud_question_answers, 0)
        send_image(target_room_id, answer_feud_image_mxc, answer_feud_image_width, answer_feud_image_height, feud_image_size)
    send_message(target_room_id, answer_message)
    
    send_message(target_room_id, message)
    wf_winner = True
    time.sleep(3)
    return None



# Configure logging
#logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
#logger = logging.getLogger(__name__)

def log_execution_time(func):
    """Decorator to log the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper


def fetch_random_word_thes(min_length=5, max_length=12, max_retries=20, max_related=5):
    for attempt in range(1, max_retries + 1):
        print(f"[Attempt {attempt}/{max_retries}] Fetching a random word...")
        try:
            # Fetch a random word
            word = get_random_word()

            if not word:
                print("No word returned from local list.")
                continue

            # Look up the word in Merriam-Webster Thesaurus
            url = f"https://www.dictionaryapi.com/api/v3/references/thesaurus/json/{word}?key={webster_thes_api_key}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            if not data or isinstance(data[0], str):
                # Nothing returned or suggestions instead of definitions
                print(f"Merriam-Webster Thesaurus did not recognize the word '{word}'. Suggestions: {data}")
                continue

            # Extract part of speech, synonyms, and antonyms
            for sense in data:
                if isinstance(sense, dict):
                    pos = sense.get("fl", "").lower()  # Functional label (part of speech)
                    synonyms = sense.get("meta", {}).get("syns", [])
                    antonyms = sense.get("meta", {}).get("ants", [])

                    # Flatten synonyms and antonyms lists
                    synonyms = [syn for group in synonyms for syn in group][:max_related]
                    antonyms = [ant for group in antonyms for ant in group][:max_related]

                    if pos and synonyms:
                        mw_url = f"https://www.merriam-webster.com/thesaurus/{word}"
                        return word, pos, synonyms, antonyms, mw_url

            # If no valid synonyms are found
            print(f"No valid synonyms found for '{word}'.")
            continue

        except Exception as e:
            print(f"Error processing word details: {e}")
            continue

    # Return None after exhausting retries
    print("Exceeded maximum retries. No valid word found.")
    return None, None, None, None, None



def load_words_from_file(filepath):
    """Load words from a local text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_random_word(min_length=5, max_length=12):
    words = load_words_from_file("wordlist.txt")
    valid_words = [w for w in words if min_length <= len(w) <= max_length]
    if not valid_words:
        return None
    return random.choice(valid_words)



def fetch_random_word(min_length=5, max_length=12, max_retries=5):
    for attempt in range(1, max_retries + 1):
        print(f"[Attempt {attempt}/{max_retries}] Fetching a random word...")
        try:
            # Fetch a random word
            word = get_random_word()

            if not word:
                print("No word returned from local list.")
                continue

            # Look up the word in Merriam-Webster
            url = f"https://www.dictionaryapi.com/api/v3/references/collegiate/json/{word}?key={webster_api_key}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            if not data or isinstance(data[0], str):
                # Nothing returned or suggestions instead of definitions
                print(f"Merriam-Webster did not recognize the word '{word}'. Suggestions: {data}")
                continue

            # Extract part of speech and definitions
            for sense in data:
                if isinstance(sense, dict):
                    pos = sense.get("fl", "").lower()  # Functional label (part of speech)
                    definitions = sense.get("shortdef", [])
                    if pos and definitions:
                        mw_url = f"https://www.merriam-webster.com/dictionary/{word}"
                        return word, pos, definitions, mw_url

            # If no valid definitions are found
            print(f"No valid definitions found for '{word}'.")
            continue

        except Exception as e:
            print(f"Error processing word details: {e}")
            continue

    # Return None after exhausting retries
    print("Exceeded maximum retries. No valid word found.")
    return None, None, None, None




def update_audit_question(question, message_content, display_name):
    if question["trivia_db"] == "math" or question["trivia_db"] == "stats":
        return

    collection = db[question["trivia_db"]]
    document_id = question["trivia_id"]

    audit_entry = {
        "display_name": display_name,
        "message_content": message_content
    }

    for attempt in range(max_retries):
        try:
            update = {
                "$push": {"audit": audit_entry},
                "$setOnInsert": {"timestamp": time.time()}
            }

            # Ensure 'audit' exists and append the new entry
            collection.update_one({"_id": document_id}, update, upsert=False)
            break  # Success, exit loop

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {delay_between_retries} seconds...")
                time.sleep(delay_between_retries)
            else:
                print(f"Failed to update audit for document '{document_id}' in {question['db']}.")



def insert_audit_question(collection_name, question, message_content, display_name):
    """
    Insert a structured question into a MongoDB collection with a frequency field and comments.
    If the question already exists, increment its frequency and append a new comment.

    :param collection_name: The name of the MongoDB collection.
    :param question: The question to be inserted or updated, expected as a dictionary.
    :param message_content: The message content to add to the comments.
    :param display_name: The display name to pair with the message content in the comments.
    """

    if not isinstance(question, dict):
        raise TypeError("The question parameter must be a dictionary")

    now = time.time()
    comment = f"{display_name}: {message_content}"  # Concatenate display name and message content

    for attempt in range(max_retries):
        try:
            # Step 1: Ensure the document exists with upsert
            filter_query = question
            initial_update = {
                "$setOnInsert": {"timestamp": now, "comments": [], "frequency": 0}  # Initialize fields if new
            }
            db[collection_name].update_one(filter_query, initial_update, upsert=True)

            # Step 2: Increment frequency and append comment
            update_data = {
                "$inc": {"frequency": 1},  # Increment frequency
                "$push": {"comments": comment}  # Append the new comment
            }
            db[collection_name].update_one(filter_query, update_data)

            break  # Exit the loop if successful

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {delay_between_retries} seconds...")
                time.sleep(delay_between_retries)
            else:
                print(f"Failed to add/update question '{question}' in {collection_name}.")


def load_previous_question():
    global previous_question
    
    for attempt in range(max_retries):
        try:
            
            # Retrieve the current longest answer streak from MongoDB
            previous_question_retrieved = db.previous_question.find_one({"_id": "previous_question"})

            if previous_question_retrieved is not None:
                previous_question = {
                    "trivia_catetgory": previous_question_retrieved.get("trivia_category"),
                    "trivia_question": previous_question_retrieved.get("trivia_question"),
                    "trivia_url": previous_question_retrieved.get("trivia_url"),
                    "trivia_answer_list": previous_question_retrieved.get("trivia_answer_list"),
                    "trivia_db": previous_question_retrieved.get("trivia_db"),
                    "trivia_id": previous_question_retrieved.get("trivia_id"),
                }
            else:
                # If the document is not found, set default values
                previous_question = {"trivia_category": None, "trivia_question": None, "trivia_url": None, "trivia_answer_list": None}            
                
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay_between_retries} seconds...")
                time.sleep(delay_between_retries)
            else:
                print("Max retries reached. Data loading failed.")
                # Set to default values if loading fails
                previous_question = {"trivia_category": None, "trivia_question": None, "trivia_url": None, "trivia_answer_list": None}





def ask_ranker_list_number(winner):
    global since_token, params, headers, max_retries, delay_between_retries

    sync_url = f"{matrix_base_url}/sync"
    collected_responses = []  # Store all responses
    processed_events = set()  # Track processed event IDs to avoid duplicates
    
    initialize_sync()
    start_time = time.time()  # Track when the question starts
    
    selected_question = 0
    while time.time() - start_time < magic_time + 5:
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
                        

                    if str(message_content) in {"1", "2", "3", "4", "5"}:
                        selected_question = str(message_content).lower()
                        react_to_message(event_id, target_room_id, "okra21")
                        message = f"\n💪🛡️ I got you {winner}. {message_content} it is.\n"
                        send_message(target_room_id, message)
                        return selected_question
                    else:
                        react_to_message(event_id, target_room_id, "okra5")
    
        except requests.exceptions.RequestException as e:
                sentry_sdk.capture_exception(e)
                print(f"Error collecting responses: {e}")                    

    set_a = ["1", "2", "3", "4", "5"]
    selected_question = random.choice(set_a)
    send_message(target_room_id, f"\n🐢⏳ Too slow. I choose {selected_question}.\n")
    return selected_question


def ask_ranker_list_question(winner, target_percentage = 1.00):    
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner
    
    try:
        time.sleep(2) 
        recent_ranker_list_ids = get_recent_question_ids_from_mongo("ranker_list")
        
        # Fetch wheel of fortune questions using the random subset method
        ranker_list_collection = db["ranker_list_questions"]
        pipeline_ranker_list = [
            {"$match": {"_id": {"$nin": list(recent_ranker_list_ids)}}},  # Exclude recent IDs
            {"$sample": {"size": 100}},  # Sample 100 documents first
            {"$group": {  
                "_id": "$question",  # Group by unique question text
                "question_doc": {"$first": "$$ROOT"}  # Keep the first document per unique question
            }},
            {"$replaceRoot": {"newRoot": "$question_doc"}},  # Flatten the grouped results
            {"$sample": {"size": 5}}  # Sample 3 random questions from the 100
        ]

        ranker_list_questions = list(ranker_list_collection.aggregate(pipeline_ranker_list))
        ranker_list_question_1 = ranker_list_questions[0]
        ranker_list_question_2 = ranker_list_questions[1]
        ranker_list_question_3 = ranker_list_questions[2]
        ranker_list_question_4 = ranker_list_questions[3]
        ranker_list_question_5 = ranker_list_questions[4]

        message = f"\n@{winner}, Choose the list #:"
        message += f"\n1️⃣. {ranker_list_question_1["question"]}"
        message += f"\n2️⃣. {ranker_list_question_2["question"]}"
        message += f"\n3️⃣. {ranker_list_question_3["question"]}"
        message += f"\n4️⃣. {ranker_list_question_4["question"]}"
        message += f"\n5️⃣. {ranker_list_question_5["question"]}\n"
        send_message(target_room_id, message)

        selected_list_question = int(ask_ranker_list_number(winner))
        
        ranker_list_question = ranker_list_questions[selected_list_question-1]
        print(ranker_list_question)

        ranker_list_question_clue = ranker_list_question["question"]
        ranker_list_question_answers = ranker_list_question["answers"]   
        ranker_list_question_category = ranker_list_question["category"]
        ranker_list_question_url = ranker_list_question["url"]
        ranker_list_question_id = ranker_list_question["_id"]  # Get the ID of the selected question
        
        if ranker_list_question_id:
            store_question_ids_in_mongo([ranker_list_question_id], "ranker_list")  # Store it as a list containing a single ID
       
    except Exception as e:
        # Capture the exception in Sentry and print detailed error information
        sentry_sdk.capture_exception(e)
        
        # Print a detailed error message with traceback
        error_details = traceback.format_exc()
        print(f"Error selecting ranker list questions: {e}\nDetailed traceback:\n{error_details}")
        
        return None  # Return an empty list in case of failure

    ranker_list_category_emojis = get_category_title(ranker_list_question_category, "")
    num_of_answers = len(ranker_list_question_answers)
    target_num_answers = int(target_percentage * num_of_answers)
    
    message = f"\n⚠️🚨 Everyone's in...{ranker_list_category_emojis}\n" 
    message += f"\n📝1️⃣ List ONE per message of...\n"
    send_message(target_room_id, message)

    time.sleep(5)

    message = f"\n👉👉 {ranker_list_question_clue}\n\n🟢🚀 GO!"
    send_message(target_room_id, message)

    processed_events = set()  # Track processed event IDs to avoid duplicates
    user_progress = defaultdict(set)
    total_progress = set()
    
    initialize_sync()
    start_time = time.time()  # Track when the question starts

    while time.time() - start_time < 30:
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
                    message_content = event.get("content", {}).get("body", "")

                    if sender == bot_user_id:
                        continue

                    current_answers = user_progress[sender_display_name]

                    # Iterate over all validAnswers
                    for answer in ranker_list_question_answers:
                        # Skip if user already has this answer
                        if answer in total_progress:
                            continue
                
                        # Compare user's guess to this official answer
                        if fuzzy_match(message_content, answer, ranker_list_question_category, ranker_list_question_url):
                            # It's a match => store the *official answer* in the user's set
                            current_answers.add(answer)
                            total_progress.add(answer)
                
                            # Check if they have enough correct answers total
                            if len(total_progress) >= num_of_answers:
                                message = f"\n🏆🎉 Okrans, you got all {len(total_progress)}/{num_of_answers}!\n"
                            
                                # 1) Compute individual scores
                                score_list = [(user, len(answers)) for user, answers in user_progress.items()]
                                
                                # 2) Sort descending by score
                                score_list.sort(key=lambda x: x[1], reverse=True)
                                
                                # 3) Filter out users with a score of 0
                                filtered_score_list = [(user, score) for user, score in score_list if score > 0]
                                
                                # 5) Print each user's score
                                message += f"\n	🏆👏 Commendable Okrans\n"
                                for rank, (user, score) in enumerate(filtered_score_list, start=1):
                                    message += f"{rank}. @{user}: {score}\n"

                                message += "\nSee the full list:\n"
                                message +=f"{ranker_list_question_url}"
                                
                                # 6) Send the message
                                send_message(target_room_id, message)
                                wf_winner = True

                                
                                return None
                                
                            break
        
        except Exception as e:
            print(f"Error processing events: {e}")

    
    score_list = [(user, len(answers)) for user, answers in user_progress.items()]
    score_list.sort(key=lambda x: x[1], reverse=True)
    filtered_score_list = [(user, score) for user, score in score_list if score > 0]

    message = ""
    if len(filtered_score_list) == 0:
        message += f"\n😬🤦 Wow. No one got a single one right. Embarassing.\n"

    
    if len(filtered_score_list) > 0:
        message += f"\n🏅💪 Okrans, you got {len(total_progress)}/{num_of_answers}.\n"
        message += f"\n	🏆👏 Commendable Okrans\n"
        for rank, (user, score) in enumerate(filtered_score_list, start=1):
            message += f"{rank}. @{user}: {score}\n"

    message += "\nSee the full list:\n"
    message +=f"{ranker_list_question_url}"
    
    send_message(target_room_id, message)
    wf_winner = True
    
    return None
    

def ask_list_question(winner, mode="competition", target_percentage = 1.00):    
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner
    
    try:
        time.sleep(2) 
        recent_list_ids = get_recent_question_ids_from_mongo("list")
        
        # Fetch wheel of fortune questions using the random subset method
        list_collection = db["list_questions"]
        pipeline_list = [
            {"$match": {"_id": {"$nin": list(recent_list_ids)}}},  # Exclude recent IDs
            {"$group": {  # Group by question text to ensure uniqueness
                "_id": "$question",  # Group by the question text field
                "question_doc": {"$first": "$$ROOT"}  # Select the first document with each unique text
            }},
            {"$replaceRoot": {"newRoot": "$question_doc"}},  # Flatten the grouped results
            {"$sample": {"size": 1}}  # Sample 1 unique question
        ]

        list_questions = list(list_collection.aggregate(pipeline_list))
        list_question = list_questions[0]

        list_question_clue = list_question["question"]
        list_question_answers = list_question["answers"]   
        list_question_category = list_question["category"]
        list_question_url = list_question["url"]
        list_question_id = list_question["_id"]  # Get the ID of the selected question
        if list_question_id:
            store_question_ids_in_mongo([list_question_id], "list")  # Store it as a list containing a single ID

       
    except Exception as e:
        # Capture the exception in Sentry and print detailed error information
        sentry_sdk.capture_exception(e)
        
        # Print a detailed error message with traceback
        error_details = traceback.format_exc()
        print(f"Error selecting list questions: {e}\nDetailed traceback:\n{error_details}")
        
        return None  # Return an empty list in case of failure


    list_category_emojis = get_category_title(list_question_category, "")
    num_of_answers = len(list_question_answers)
    target_num_answers = int(target_percentage * num_of_answers)
    
    message = f"\n⚠️🚨 Everyone's in...{list_category_emojis}\n" 
    message += f"\n📝1️⃣ List ONE per message of...\n"
    send_message(target_room_id, message)

    time.sleep(5)

    message = f"\n👉👉 {list_question_clue}\n\n🟢🚀 GO!"
    send_message(target_room_id, message)

    processed_events = set()  # Track processed event IDs to avoid duplicates
    user_progress = defaultdict(set)
    total_progress = set()
    
    initialize_sync()
    start_time = time.time()  # Track when the question starts

    
    while time.time() - start_time < 30:
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
                    message_content = event.get("content", {}).get("body", "")

                    if sender == bot_user_id:
                        continue

                    if sender_display_name == winner and mode == "solo":
                        continue

                    current_answers = user_progress[sender_display_name]

                    # Iterate over all validAnswers
                    for answer in list_question_answers:
                        # Skip if user already has this answer
                        if answer in current_answers:
                            continue
                
                        # Compare user's guess to this official answer
                        if fuzzy_match(message_content, answer, list_question_category, list_question_url):
                            # It's a match => store the *official answer* in the user's set
                            current_answers.add(answer)
                            total_progress.add(answer)
                
                            # Check if they have enough correct answers total
                            if len(current_answers) >= num_of_answers and mode == "competition":
                                message = f"\n🏆🎉 @{sender_display_name} got all {num_of_answers}!"
                            
                                # Now figure out 2nd and 3rd places.
                                # 1) Build a list of (user, score) for *all* users
                                score_list = [(user, len(answers)) for user, answers in user_progress.items()]
                
                                # 2) Sort descending by score
                                score_list.sort(key=lambda x: x[1], reverse=True)
                
                                # The top user is score_list[0], second place = score_list[1], etc.
                                # But only if they exist
                                if len(score_list) > 1:
                                    second_user, second_score = score_list[1]
                                    message += f"\n2nd place: @{second_user} with {second_score}/{num_of_answers}."
                                if len(score_list) > 2:
                                    third_user, third_score = score_list[2]
                                    message += f"\n3rd place: @{third_user} with {third_score}/{num_of_answers}."
                
                                send_message(target_room_id, message)

                                if winner == sender_display_name:
                                    wf_winner = True
                                    
                                return None
                                
                            if len(total_progress) >= num_of_answers and mode == "cooperative":
                                message = f"\n🏆🎉 Okrans got all {num_of_answers}!"
                                return None

                            if len(total_progress) >= num_of_answers and mode == "solo":
                                message = f"\n🏆🎉 @{winner} got all {num_of_answers}!"
                                return None
                                
                            break
        
        except Exception as e:
            print(f"Error processing events: {e}")

    if mode == "competition":
      
        # Now figure out 2nd and 3rd places.
        # 1) Build a list of (user, score) for *all* users
        score_list = [(user, len(answers)) for user, answers in user_progress.items()]
    
        # 2) Sort descending by score
        score_list.sort(key=lambda x: x[1], reverse=True)
    
        # The top user is score_list[0], second place = score_list[1], etc.
        # But only if they exist
        
        if len(score_list) == 0:
            message = f"\n😬🤦 Wow. No one got a single one right. Embarassing."
            send_message(target_room_id, message)
            return None
        
        if len(score_list) > 0:
            first_user, first_score = score_list[0]
            message = f"\n🥇🏆 1st place: @{first_user} with {first_score}/{num_of_answers}!"
    
        if len(score_list) > 1:
            second_user, second_score = score_list[1]
            message += f"\n🥈🎊 2nd place: @{second_user} with {second_score}/{num_of_answers}."
       
        if len(score_list) > 2:
            third_user, third_score = score_list[2]
            message += f"\n🥉🎉 3rd place: @{third_user} with {third_score}/{num_of_answers}."
    
        send_message(target_room_id, message)
        
        if winner == first_user:
            wf_winner = True
        
        return None
    
    if mode == "cooperative":
        message = f"\n😢👎 Sorry. Okrans only got {len(current_answers)}/{num_of_answers}."
        return None
    
    if mode == "solo":
        message = f"\n😢👎 Sorry. @{winner} only got {len(current_answers)}/{num_of_answers}."
        return None

    return None




def ask_survey_question():
    global since_token, params, headers, max_retries, delay_between_retries
    # Connect to the database and collection
    collection = db["survey_questions"]

    # Fetch the document by question_id
    random_question = list(
        collection.aggregate([
            {"$match": {"enabled": True}},  # Only include documents where enabled is true
            {"$sample": {"size": 1}}       # Randomly select one document
        ])
    )
    if not random_question:
        print(f"No documents found in collection 'survey_questions'.")
        return None

    # Extract relevant details
    survey_question = random_question[0]
    question_text = survey_question.get("question", "No question text available.")
    question_type = survey_question.get("question_type", "yes/no")
    valid_answers = survey_question.get("valid_answers", [])
    responses = survey_question.get("responses", {})
    processed_events = set()  # Track processed event IDs to avoid duplicates
    collected_responses = {}  # Collect responses locally
    current_time = datetime.datetime.now(timezone.utc).isoformat()

    initialize_sync()
    start_time = time.time()  # Track when the question starts

    question_type_lookup = {
        "yes-no": {
            "emojis": "👍👎",
            "intro_text": "Answer YES or NO"
        },
        "multiple-choice": {
            "emojis": "🔠📝",
            "intro_text": "Choose a letter:"
        },
        "rating-10": {
            "emojis": "⭐️🔟",
            "intro_text": "On a scale from 1 to 10"
        },
         "word-3": {
            "emojis": "3️⃣🔤",
            "intro_text": "3 word limit"
        }
    }

    question_info = question_type_lookup.get(question_type, {})
    emojis = question_info.get("emojis", "🤔❓")
    intro_text = question_info.get("intro_text", "What do you think?")
       
    #message = f"\n{emojis} {intro_text}\n"
    #message += f"\n❓ {question_text}\n"

    message = "\n🤔❓Should I rename LiveTrivia?\n"
    message += "\nA. No."
    message += "\nB. Yes. Trivia Okra."
    message += "\nC. Yes. Okra Trivia."
    message += "\nD. Yes. Something else.\n"
    
    send_message(target_room_id, message)
        
    while time.time() - start_time < 15:
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
                    message_content = event.get("content", {}).get("body", "")

                    if sender == bot_user_id:
                        continue

                # Process responses based on question type
                if question_type == "yes-no":
                   if message_content.lower().startswith("y") or message_content.strip() == "👍":
                       collected_responses[sender_display_name] = {"answer": "Yes", "timestamp": current_time}
                   elif message_content.lower().startswith("n") or message_content.strip() == "👎":
                       collected_responses[sender_display_name] = {"answer": "No", "timestamp": current_time}

                elif question_type == "multiple-choice":
                    matched_answer = next((answer for answer in valid_answers if answer.lower() in message_content.lower()), None)
                    if matched_answer:
                        collected_responses[sender_display_name] = {"answer": matched_answer, "timestamp": current_time}
                
                elif question_type == "rating-10":
                    # Search for any number in the message
                    match = re.search(r'\b\d+(\.\d+)?\b', message_content)
                    if match:
                        try:
                            rating = float(match.group())
                            if 1 <= rating <= 10:
                                collected_responses[sender_display_name] = {
                                    "answer": round(rating, 1),
                                    "timestamp": current_time
                                }
                        except ValueError:
                            continue  # Ignore invalid ratings

                elif question_type == "word-3":
                    words = message_content.split()
                    if words:  # Check if there are any words in the message
                        # Initialize or update the user's word list
                        if sender_display_name not in collected_responses:
                            collected_responses[sender_display_name] = {
                                "answer": [],  # Initialize an empty list for words
                                "timestamp": current_time
                            }
        
                        # Add the new words to the user's list
                        collected_responses[sender_display_name]["answer"].extend(words)
                        
                        # Keep only the 3 most recent words
                        collected_responses[sender_display_name]["answer"] = collected_responses[sender_display_name]["answer"][-3:]
                
                        # Update the timestamp
                        collected_responses[sender_display_name]["timestamp"] = current_time
        
        except Exception as e:
            print(f"Error processing events: {e}")

    # Update MongoDB with all collected responses at once
    responses.update(collected_responses)
    collection.update_one(
        {"_id": survey_question["_id"]},
        {"$set": {"responses": responses}}
    )

    # Summarize results after the time is up
    total_responses = len(responses)
    if total_responses > 0:
        if question_type == "yes-no":
            positive_responses = sum(1 for ans in responses.values() if ans["answer"].lower() == "yes")
            percentage_positive = (positive_responses / total_responses) * 100
            percentage_negative = 100 - percentage_positive
            if percentage_negative > 50:
                summary_message = f"🥀🪦 {int(percentage_negative)}% of Okrans have said NOkra. "
            else:
                summary_message = f"🏄‍♂️🌟 {int(percentage_positive)}% of Okrans have said OkraYeah!"
            send_message(target_room_id, summary_message)
    
        elif question_type == "rating-10":
            total_rating = sum(ans["answer"] for ans in responses.values() if isinstance(ans["answer"], (int, float)))
            average_rating = total_rating / total_responses
            summary_message = f"⭐️🔟 Average rating across all Okrans is {average_rating:.1f} out of 10."
            send_message(target_room_id, summary_message)
    
        elif question_type == "word-3":
            # Collect all words across all users
            from collections import Counter
            all_words = []
            for ans in responses.values():
                if isinstance(ans["answer"], list):  # Ensure the answer is a list of words
                    all_words.extend(ans["answer"])
    
            # Normalize words (case and punctuation insensitive)
            normalized_words = [word.strip(string.punctuation).lower() for word in all_words]
            word_counts = Counter(normalized_words)
            most_common_words = [f'"{word.capitalize()}"' for word, _ in word_counts.most_common(3)]
    
            # Format the message
            if most_common_words:
                summary_message = f"📚🔤 Okrans say Live Trivia is: {', '.join(most_common_words)}."
                send_message(target_room_id, summary_message)

            if all_words:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant who removes offensive or inappropriate words from user prompts for use with Dall-e."},
                            {"role": "user", "content": f"Remove any words that are against openAI's safety filter so I can use them in a Dall-e prompt: {all_words}"}
                        ],
                        max_tokens=500
                    )
                    sanitized_all_words = response["choices"][0]["message"]["content"]
                    print("OG Prompt:", all_words)
                    print("Sanitized Prompt:", sanitized_all_words)
   
                    prompt = f"Create a hyperrealistic futuristic okra themed environment described as {sanitized_all_words}."
                    response = openai.Image.create(
                        model="dall-e-3",  # Use the most advanced DALL-E model available
                        prompt=prompt,
                        n=1,
                        size="1024x1024"  # Adjust size as needed
                    )
                    # Return the image URL from the API response
                    image_url = response["data"][0]["url"]
                    image_mxc, image_width, image_height = download_image_from_url(image_url)
                    image_description = describe_image_with_vision(image_url, "title", prompt)

                    pre_message = f"\n🥒🌀 Behold, your Okraverse"
                    post_message = f"\n'{image_description}'\n"
                    
                    send_message(target_room_id, pre_message)
                    send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)
                    send_message(target_room_id, post_message)
                    
                    #collection = db["parameters"]
                    # Step 2: Update or insert the document
                    #collection.update_one(
                    #    {"_id": "okraverse_description"},  # Filter to find the document by its _id
                    #    {"$set": {"description": image_description}},  # Update the description field with the new value
                    #    upsert=True  # Ensure the document is created if it doesn't already exist
                    #)
                    

                    #image_data = requests.get(image_url).content
                    #image = Image.open(io.BytesIO(image_data))
                    #buffer = io.BytesIO()
                    #image.save(buffer, format="PNG")
                    #buffer.seek(0)
                    #upload_okraverse_to_s3(buffer)
                    return None
                    
                except openai.OpenAIError as e:
                    print(f"Error generating image: {e}")
                

                
    
        
        

def generate_themed_country_image(country, city):

    #prompt = (
    #    f"Create an okra themed image of a kitchen in the country: {country}. "
    #)

    prompt = (
        f"Show a stereotypical person with a face from {country} holding an okra in a stereotypical setting in {country}. "
    )
    
    # Generate the image using DALL-E
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"  # Adjust size as needed
        )
        # Return the image URL from the API response
        image_url = response["data"][0]["url"]

        return image_url
        
    except openai.OpenAIError as e:
        print(f"Error generating image: {e}")
        if "Your request was rejected as a result of our safety system" in str(e):
            # Use a default safe prompt
            default_prompt = f"Generate an image of an okra in {country}."
            try:
                response = openai.Image.create(
                    prompt=default_prompt,
                    n=1,
                    size="512x512"
                )
                
                # Return the image URL from the API response
                image_url = response["data"][0]["url"]
     
                return image_url
            
            except openai.OpenAIError as e2:
                print(f"Error generating default image: {e2}")
                return "Image generation failed!"
        
        else:
            return "Image generation failed!"


def get_google_maps(lat, lon):
    base_street_view_url = "https://maps.googleapis.com/maps/api/streetview"
    base_static_map_url = "https://maps.googleapis.com/maps/api/staticmap"
    base_metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    base_maps_url = "https://www.google.com/maps"
    
    # Check Street View availability using the Metadata API
    metadata_params = {
        "location": f"{lat},{lon}",
        "key": googlemaps_api_key
    }
    metadata_response = requests.get(base_metadata_url, params=metadata_params)
    metadata = metadata_response.json()
    
    if metadata.get("status") == "OK":
        street_view_params = {
            "size": "600x400",  # Image size
            "location": f"{lat},{lon}",  # Latitude and longitude
            "fov": 90,  # Field of view
            "heading": 0,  # Camera direction
            "pitch": 0,  # Camera angle
            "key": googlemaps_api_key
        }
        street_view_url = f"{base_street_view_url}?{requests.compat.urlencode(street_view_params)}"
    else:
        street_view_url = None  # No Street View data available

    # Construct Satellite View URLs
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
    # Connect to the collection
    collection = db["where_is_okra"]

    # Count total documents
    total = collection.count_documents({})
    if total == 0:
        print("No cities found in 'where_is_okra' collection.")
        return None

    # Generate a random skip index
    skip = random.randint(0, total - 1)

    # Use skip + limit to select a random document
    result = collection.find().skip(skip).limit(1)
    random_city = next(result, None)

    if not random_city:
        print("Failed to fetch a random city.")
        return None

    # Extract city details
    city_name = random_city.get("city")
    country_name = random_city.get("country")
    is_capital = random_city.get("capital")
    lat = random_city.get("lat")
    lon = random_city.get("lon")

    # Conversion factors
    miles_per_lat_degree = 1 / 69  # 1 degree latitude ≈ 69 miles
    miles_per_lon_degree = 1 / (69 * math.cos(math.radians(lat)))  # Adjust for latitude

    # Generate random offsets within ±0.5 miles (to stay within 1 square mile)
    lat_offset = random.uniform(-0.5, 0.5) * miles_per_lat_degree
    lon_offset = random.uniform(-0.5, 0.5) * miles_per_lon_degree

    # Apply offsets to the original latitude and longitude
    lat = lat + lat_offset
    lon = lon + lon_offset

    street_view_url, satellite_view_url, satellite_view_live_url = get_google_maps(lat, lon)

    
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

    themed_image_url = generate_themed_country_image(country_name, city_name)
    
    return city_name, country_name, "World Cities", location_clue, street_view_url, satellite_view_url, satellite_view_live_url, themed_image_url




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

                if len(intro_text) < 500:
                    continue

                
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

 

def describe_image_with_vision(image_url, mode, prompt):
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
                                "text": f"Based on what you see in the image, give the image a name with 5 words maximum. The prompt used to create this image was '{prompt}'."
                                #"text": f"Based on what you see in the image, give the image an okra themed title with 5 words maximum."
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






def get_user_data(username, num_of_subreddits):
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
        top_subreddits = ", ".join([subreddit for subreddit, _ in subreddit_counts.most_common(num_of_subreddits)])

        return {
            "avatar_url": avatar_url,
            "top_subreddits": top_subreddits
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def sovereign_check(user):
    sovereigns = {sovereign['user'] for sovereign in db.hall_of_sovereigns.find()}
    if user in sovereigns:
        return True
    else:
        return False



def get_image_url_from_s3():
    bucket_name = "triviabotwebsite"
    prefix = "generated-images/"
    
    # Connect to S3
    s3 = boto3.client("s3")
    
    # List objects in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    # Extract file keys
    files = [item['Key'] for item in response.get('Contents', []) if item['Key'].lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    # Pick a random file
    random_file = random.choice(files)
    public_url = f"https://{bucket_name}.s3.amazonaws.com/{random_file}"
   
    image_mxc, image_width, image_height = download_image_from_url(public_url)
    

    print(random_file)
    # Step 1: Remove the prefix and file extension
    filename = os.path.basename(random_file).replace('.png', '')

    # Step 2: Extract with regex
    pattern = r'^(.+?)\s*&\s*(.+?)\s+\((.+?)\)$'
    match = re.match(pattern, filename)

    if match:
        title = match.group(1)
        user = match.group(2)
        full_date = match.group(3)
        
        # Remove the time from the date string
        date_only = ' '.join(full_date.split()[:-1])

        message = f"\nMasterpiece: '{title}'\n"
        message += f"Okra's Muse: {user}\n"
        message += f"Creation Date: {date_only}\n"

    else:
        message += f"\nA masterpiece from the Okra Museum.\n"
                

    send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)
    send_message(target_room_id, message)


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

def upload_okraverse_to_s3(buffer):
    try:
        bucket_name='triviabotwebsite'
        folder_name='okraverse'
        object_name = "okraverse.png"
             
        # Step 3: Connect to S3 and upload the file
        s3_client = boto3.client("s3")
        s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=buffer.getvalue(), ContentType="image/png")

        # Step 4: Generate and return the S3 URL
        return None

    except (BotoCoreError, ClientError) as boto_err:
        print(f"Error uploading to S3: {boto_err}")
        return None

def load_parameters():
    global image_wins, image_points
    global num_list_players
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
    global skip_summary
    global discount_step_amount
    global discount_streak_amount
 
    
    # Default values
    default_values = {
        "image_wins": 5,
        "image_points": 5000,
        "num_list_players": 5,
        "num_mysterybox_clues_default": 3,
        "num_crossword_clues_default": 0,
        "num_jeopardy_clues_default": 3,
        "num_wof_clues_default": 0,
        "num_wof_clues_final_default": 3,
        "num_wf_letters": 3,
        "num_math_questions_default": 1,
        "num_stats_questions_default": 0,
        "skip_summary": False,
        "discount_step_amount": 20,
        "discount_streak_amount": 5
    }
    
    for attempt in range(max_retries):
        try:

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
            image_points = parameters["image_points"]
            num_list_players = parameters["num_list_players"]
            num_mysterybox_clues_default = parameters["num_mysterybox_clues_default"]
            num_crossword_clues_default = parameters["num_crossword_clues_default"]
            num_jeopardy_clues_default = parameters["num_jeopardy_clues_default"]
            num_wof_clues_default = parameters["num_wof_clues_default"]
            num_wof_clues_final_default = parameters["num_wof_clues_final_default"]
            num_wf_letters = parameters["num_wf_letters"]
            num_math_questions_default = parameters["num_math_questions_default"]
            num_stats_questions_default = parameters["num_stats_questions_default"]
            skip_summary = parameters["skip_summary"]
            discount_step_amount = parameters["discount_step_amount"]
            discount_streak_amount = parameters["discount_streak_amount"]
            
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
                image_points = default_values["image_points"]
                num_list_players = default_values["num_list_players"]
                num_mysterybox_clues_default = default_values["num_mysterybox_clues_default"]
                num_crossword_clues_default = default_values["num_crossword_clues_default"]
                num_jeopardy_clues_default = default_values["num_jeopardy_clues_default"]
                num_wof_clues_default = default_values["num_wof_clues_default"]
                num_wof_clues_final_default = default_values["num_wof_clues_final_default"]
                num_wf_letters = default_values["num_wf_letters"]
                num_math_questions_default = default_values["num_math_questions_default"]
                num_stats_questions_default = default_values["num_stats_questions_default"]
                skip_summary = default["skip_summary"]
                discount_step_amount = default["discount_step_amount"]
                discount_streak_amount = default["discount_streak_amount"]

                num_mysterybox_clues = num_mysterybox_clues_default
                num_crossword_clues = num_crossword_clues_default
                num_jeopardy_clues = num_jeopardy_clues_default
                num_wof_clues = num_wof_clues_default
                num_wof_clues_final = num_wof_clues_final_default
                num_math_questions = num_math_questions_default
                num_stats_questions = num_stats_questions_default


def nice_creep_okra_option(winner):
    global since_token, params, headers, max_retries, delay_between_retries, nice_okra, creep_okra, wf_winner, seductive_okra, joke_okra
    nice_okra = False
    creep_okra = False
    wf_winner = False
    seductive_okra = False
    joke_okra = False

    sync_url = f"{matrix_base_url}/sync"
    processed_events = set()  # Track processed event IDs to avoid duplicates
    
    # Initialize the sync and message to prompt user for letters
    initialize_sync()
    start_time = time.time()  # Track when the question starts
    
    message = f"\n☕🤝 Thank you @{winner} for your support.\n\n" 
    message += f"🥒😊 Say 'okra' and I'll be nice.\n"
    message += f"👀🔭 Say 'creep' and I'll snoop your Reddit profile.\n"
    message += f"💋👠 Say 'love me' and I'll seduce you.\n"
    message += f"🤡🤣 Say 'joke' and I'll write you a dad joke.\n"
    message += f"🔥🍗 Say nothing and I'll roast you.\n\n"
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
                        seductive_okra = False
                        joke_okra = False
                        return None
                        
                    if "creep" in message_content.lower():
                        react_to_message(event_id, target_room_id, "okra21")
                        creep_okra = True
                        nice_okra = False
                        wf_winner = False
                        seductive_okra = False
                        joke_okra = False
                        return None

                    if "love me" in message_content.lower():
                        react_to_message(event_id, target_room_id, "okra21")
                        creep_okra = False
                        nice_okra = False
                        wf_winner = False
                        seductive_okra = True 
                        joke_okra = False
                        return None

                    if "joke" in message_content.lower():
                        react_to_message(event_id, target_room_id, "okra21")
                        nice_okra = False
                        creep_okra = False
                        wf_winner = False
                        seductive_okra = False
                        joke_okra = True
                        return None
                        
                    if "nothing" in message_content.lower():
                        react_to_message(event_id, target_room_id, "okra21")
                        nice_okra = False
                        creep_okra = False
                        wf_winner = False
                        seductive_okra = False
                        joke_okra = False
                        return None
                        
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error collecting responses: {e}")

    return None


def generate_round_summary_image(round_data, winner):
    if skip_summary == True:
        message += "\nBe sure to drink your Okratine.\n"
        send_message(target_room_id, message)
        return None
        
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
            f"Draw What you think {winner} looks like surrounded by okra and money. "
            "Add glowing lights, hearts, and a festive atmosphere."
        )
        message = f"✊🔥 {winner_at}, thank you for your donation to the cause. And nice streak!\n"
    
    else:
        categories = {
            "0": "😠🥒 Okrap (Horror)",
            "1": "🌹🏰 Okrenaissance",
            "2": "😇✨ Okroly and Divine",
            "3": "🎲🔀 (OK)Random",
            "4": f"🖼️🔤 Provide the Prompt ☕",
            "5": f"🖼️👤 From your Reddit Avatar ☕",
            "6": f"🖼️📜 From your Top Subreddit ☕"
        }

        # Ask the user to choose a category
        selected_category, additional_prompt = ask_category(winner, categories, winner_coffees)
        reddit_avatar_description = ""
        top_subreddits = ""

        if selected_category == "5" or selected_category == "6":
            try:
                user_data = get_user_data(winner, 1)
                if not user_data:
                    print(f"Failed to fetch data for {winner}.")
                    user_data = {"avatar_url": "N/A", "subreddits": []}
            except Exception as e:
                print(f"Error fetching Reddit data for {winner}: {e}")
                user_data = {"avatar_url": "N/A", "subreddits": []}
        
            # Extract user data
            reddit_avatar_url = user_data.get("avatar_url", "No avatar available.")
            reddit_avatar_description = describe_image_with_vision(reddit_avatar_url, "describe", "")
            top_subreddits = user_data.get("top_subreddits", "")
                    
        prompts_by_category = {
            "0": [
                f"A scary scene from a horror movie with what you think {winner} looks running from an okra."
            ],
            "1": [
                f"A Renaissance painting of what you think {winner} looks like holding an okra. Make the painting elegant and refined."
            ],
            "2": [
                f"An image of what you think {winner} looks like worshipping an okra. Make it appealing and accepting of religions of all types."
            ],
            "3": [
                f"An image of what you think {winner} looks like intereracting with an okra in the most crazy, ridiculous, and over the top random way."
            ],
            "4": [
                f"Draw an okra themed picture of {winner} {additional_prompt}.\n"  
            ],
            "5": [
                f"Draw an okra themed picture of what you think {winner} looks like based on their avatar, which is described as '{reddit_avatar_description}'.\n"
            ],
            "6": [
                f"Draw an okra themed caricature of what you think {winner} looks like based on their most visited subreddit, which is '{top_subreddits}'.\n"
            ]
        }

        # Select a prompt based on the chosen category
        if selected_category and selected_category in prompts_by_category:
            prompt = random.choice(prompts_by_category[selected_category])
        else:
            prompt = f"A horror image of what you think {winner} looks like being pursued by something okra themed."

            
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
            image_description = describe_image_with_vision(image_url, "title", prompt)
        else:
            image_description = describe_image_with_vision(image_url, "title", prompt)
            
        image_mxc, image_width, image_height = download_image_from_url(image_url)
        send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)

        message = f"🔥💖 {winner_at}, you've done well. I drew this for you.\n"
        message += f"\nI call it: '{image_description}'\n"
        message += f"\n🏛️👋 Welcome to the Okra Museum"
        message += "\n🌐➡️ https://livetriviastats.com/okra-museum\n"
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
            default_prompt = f"A Renaissance painting of what you think {winner} looks like holding an okra. Make the painting elegant and refined."
            try:
                response = openai.Image.create(
                    prompt=default_prompt,
                    n=1,
                    size="512x512"
                )
                
                # Return the image URL from the API response
                image_url = response["data"][0]["url"]
                image_description = describe_image_with_vision(image_url, "title", prompt)
                image_mxc, image_width, image_height = download_image_from_url(image_url)
                send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)
        
                message = f"😈😉 {winner_at} Naughty naughty, I'll have to pick another.\n\n"
                message += f"🔥💖 {winner_at}, you've done well. I drew this for you.\n"
                message += f"\nI call it: '{image_description}'\n"
                message += f"\n🏛️👋 Welcome to the Okra Museum"
                message += "\n🌐➡️ https://livetriviastats.com/okra-museum\n"
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
    category_message = f"\n🎨🖍️ @{winner} Pick a theme for the Okra Museum! Some require ☕.\n\n"
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
                        message = f"\n🙏😔 Sorry {winner}. Choice {message_content} requires ☕️.\n"
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
    message = f"\n🖼️🔟 @{winner}, Fill in the blank. 10 words max and be good.\n"
    message += f"\n'Draw an okra themed picture of @{winner} _____________.'\n"
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
        message = f"💥🤯 Ok...ra I got: 'Draw an okra themed picture of @{winner} {' '.join(collected_words)}'"
    send_message(target_room_id, message)
    return ' '.join(collected_words)



def get_coffees(username):
    username = username.lower()
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
        donors_collection = db["donors"]
        new_donors = []
        next_page_url = base_url

        while next_page_url:
            # Retry logic for fetching the page
            for attempt in range(max_retries):
                try:
                    response = requests.get(next_page_url, headers=headers)
                    response.raise_for_status()
                    api_response = response.json()
                    break  # success, exit retry loop
                except (requests.exceptions.RequestException, ValueError) as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay_between_retries)
                    else:
                        sentry_sdk.capture_exception(e)
                        print(f"Failed to fetch page after {max_retries} attempts. Aborting.")
                        return new_donors  # Return what we got so far

            # Extract donations list from the 'data' key
            donations = api_response.get("data", [])
            if not isinstance(donations, list):
                print(f"Unexpected donations format: {type(donations)}. Donations: {donations}")
                break

            for donor in donations:
                if isinstance(donor, dict):
                    donor_id = donor.get("support_id")
                    donor_name = donor.get("supporter_name", "")
                    donor_coffees = donor.get("support_coffees")
                    donor_coffee_price = donor.get("support_coffee_price")
                    donor_date = donor.get("support_created_on")
                    donor_note = donor.get("support_note", "")

                    # Normalize initial donor_name
                    if donor_name.startswith("@"):
                        donor_name = donor_name[1:]
                    donor_name = donor_name.lower()

                    # Extract a word starting with @ or r/ from donor_note if present
                    if isinstance(donor_note, str):
                        words = donor_note.split()
                        for word in words:
                            if word.startswith("@"):
                                donor_name = word[1:].lower()
                                break
                            elif word.startswith("r/"):
                                donor_name = word[2:].lower()
                                break

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
                            donors_collection.insert_one(new_donor)
                            new_donors.append(new_donor)
                        except Exception as e:
                            print(f"Error inserting donor into MongoDB: {e}")
                else:
                    print(f"Skipping invalid donor format: {donor}")

            next_page_url = api_response.get("next_page_url")
        
        print(f"New donors added: {new_donors}")
        return new_donors

    except Exception as e:
        print(f"Unexpected error: {e}")
        sentry_sdk.capture_exception(e)
        return []


def get_math_question():
    question_functions = [create_mean_question, create_median_question, create_derivative_question, create_zeroes_question, create_factors_question, create_base_question, create_trig_question, create_algebra_question]
    #question_functions = [create_algebra_question]
    selected_question_function = random.choice(question_functions)
    return selected_question_function()

        
def get_stats_question():
    question_functions = [create_mean_question, create_median_question]
    selected_question_function = random.choice(question_functions)
    return selected_question_function()



def create_trig_question():
    return {
        "category": "Mathematics: Trigonometry",
        "question": "",
        "url": "trig",
        "answers": [""]
    }


def create_algebra_question():
    return {
        "category": "Mathematics: Algebra",
        "question": "",
        "url": "algebra",
        "answers": [""]
    }


# Function to create a mean question in dictionary format
def create_mean_question():
    return {
        "category": "Mathematics: Mean",
        "question": "What is the MEAN of the following set?",
        "url": "mean",
        "answers": [""]
    }

# Function to create a median question in dictionary format
def create_median_question():
    return {
        "category": "Mathematics: Median",
        "question": "What is the MEDIAN of the following set?",
        "url": "median",
        "answers": [""]
    }

def create_base_question():
    return {
        "category": "Mathematics: Bases",
        "question": f"What is the DECIMAL equivalent of the following BASE number:",
        "url": "base",
        "answers": [""]
    }


# Function to create a derivative question in dictionary format
def create_derivative_question():
    return {
        "category": "Mathematics: Derivatives",
        "question": "What is the DERIVATIVE with respect to x?",
        "url": "derivative",
        "answers": [""]
    }

def create_sum_zeroes_question():
    return {
        "category": "Mathematics: Polynomials",
        "question": "What is the SUM of the zeroes (or roots) of the function defined:",
        "url": "zeroes sum",
        "answers": [""]
    }

def create_product_zeroes_question():
    return {
        "category": "Mathematics: Polynomials",
        "question": "What is the PRODUCT of the zeroes (or roots) of the function defined:",
        "url": "zeroes product",
        "answers": [""]
    }

def create_zeroes_question():
    return {
        "category": "Mathematics: Polynomials",
        "question": "What are the 2 ZEROES (or roots) of the function defined:",
        "url": "zeroes",
        "answers": [""]
    }


def create_factors_question():
    return {
        "category": "Mathematics: Polynomials",
        "question": "Factor the function defined:",
        "url": "factors",
        "answers": [""]
    }
        
def select_wof_questions(winner):
    global fixed_letters
    
    try:
        time.sleep(2)
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
            {"$sample": {"size": 5}}  # Sample 3 unique questions
        ]

        wof_questions = list(wof_collection.aggregate(pipeline_wof))
        #print(wof_questions)

        message = f"\n🍷⚔️ @{winner}: Choose wisely.  Some require ☕.\n\n"
        # Assuming wof_questions contains the sampled questions, with each document as a list/tuple
        counter = 0
        for doc in wof_questions:
            category = doc["question"]  # Use the key name to access category
            message += f"{counter}. {category}\n"
            counter = counter + 1
        send_message(target_room_id, message)  
        premium_counts = counter
        message = f"{counter}. 🌐🎲 Wikipedia Roulette ☕\n"
        counter = counter + 1
        message += f"{counter}. 📚🎲 Dictionary Roulette ☕\n"
        counter = counter + 1
        message += f"{counter}. 📖🎲 Thesaurus Roulette ☕\n"
        counter = counter + 1
        message += f"{counter}. 🌍❔ Where's Okra? ☕\n"
        counter = counter + 1
        message += f"{counter}. ⚔️🧍 FeUd ☕\n"
        counter = counter + 1
        message += f"{counter}. ⚔️⚡ FeUd Blitz ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        counter = counter + 1
        message += f"{counter}. 📝🥊 List Battle ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        counter = counter + 1
        message += f"{counter}. 🎥⚡ Poster Blitz ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        counter = counter + 1
        message += f"{counter}. 🎬💥 Movie Mayhem ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        counter = counter + 1
        message += f"{counter}. 🧩🔗 Missing Link ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        counter = counter + 1
        send_message(target_room_id, message)  

        message = f"{counter}. 👤🌟 Famous Peeps ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        counter = counter + 1
        message += f"{counter}. 🔢📜 Ranker Lists ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        counter = counter + 1
        message += f"{counter}. 👁️✨ Magic Eye D ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        counter = counter + 1
        message += f"{counter}. ❓🦓 OkrAnimal ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        counter = counter + 1
        message += f"{counter}. 🟢🎩 The Riddler ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        counter = counter + 1
        message += f"{counter}. 🤓📚 Word Nerd ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        counter = counter + 1
        message += f"{counter}. 🎏🎉 Flag Fest ✨ALL PLAY ({num_list_players}+)✨ ☕\n"
        message += f"\n00. 🥗🌟 Okra's Choice\n"
        send_message(target_room_id, message) 
        
        selected_wof_category = ask_wof_number(winner)

        if int(selected_wof_category) < premium_counts:
            wof_question = wof_questions[int(selected_wof_category)]
            wof_answer = wof_question["answers"][0]
            wof_clue = wof_question["question"]
                    
            wof_question_id = wof_question["_id"]  # Get the ID of the selected question
            if wof_question_id:
                store_question_ids_in_mongo([wof_question_id], "wof")  # Store it as a list containing a single ID
        
        elif selected_wof_category == "11":
            ask_list_question(winner)
            time.sleep(3)
            return None

        elif selected_wof_category == "10":
            ask_feud_question(winner, "cooperative")
            time.sleep(3)
            return None

        elif selected_wof_category == "9":
            ask_feud_question(winner, "solo")
            time.sleep(3)
            return None

        elif selected_wof_category == "12":
            ask_poster_challenge(winner)
            time.sleep(3)
            return None

        elif selected_wof_category == "13":
            ask_movie_scenes_challenge(winner)
            time.sleep(3)
            return None

        elif selected_wof_category == "14":
            ask_missing_link(winner)
            time.sleep(3)
            return None

        elif selected_wof_category == "15":
            ask_ranker_people_challenge(winner)
            time.sleep(3)
            return None

        elif selected_wof_category == "16":
            ask_ranker_list_question(winner)
            time.sleep(3)
            return None

        elif selected_wof_category == "17":
            ask_magic_challenge(winner)
            time.sleep(3)
            return None

        elif selected_wof_category == "18":
            ask_animal_challenge(winner)
            time.sleep(3)
            return None

        elif selected_wof_category == "19":
            ask_riddle_challenge(winner)
            time.sleep(3)
            return None

        elif selected_wof_category == "20":
            ask_dictionary_challenge(winner)
            time.sleep(3)
            return None

        elif selected_wof_category == "21":
            ask_flags_challenge(winner)
            time.sleep(3)
            return None
        
        elif selected_wof_category == "5":
            wof_answer, redacted_intro, wof_clue, wiki_url = get_wikipedia_article(3, 16)
            wikipedia_message = f"\n🥒⬛ Okracted Clue:\n\n{redacted_intro}\n"
            send_message(target_room_id, wikipedia_message)
            time.sleep(3)

        elif selected_wof_category == "6":
            wof_answer, wof_clue, word_definition, word_url = fetch_random_word()
            dictionary_message = f"\n📖🔍 Definition:\n"
            for i, definition in enumerate(word_definition, start=1):
                dictionary_message += f"\n {i}. {definition}"
            dictionary_message += "\n"
            send_message(target_room_id, dictionary_message)
            time.sleep(3)

        elif selected_wof_category == "7":
            wof_answer, wof_clue, word_syn, word_ant, word_url = fetch_random_word_thes()
            thesaurus_message = f"\n📖✅ Synonyms\n"
            for i, synonym in enumerate(word_syn, start=1):
                thesaurus_message += f"\n {i}. {synonym}"
            thesaurus_message += "\n"
            if word_ant:
                thesaurus_message += "\n📖❌ Antonyms:"
                for i, antonym in enumerate(word_ant, start=1):
                    thesaurus_message += f"\n  {i}. {antonym}"
                thesaurus_message += "\n"
            send_message(target_room_id, thesaurus_message)
            time.sleep(3)

        elif selected_wof_category == "8":
            wof_answer, country_name, wof_clue, location_clue, street_view_url, satellite_view_url, satellite_view_live_url, themed_country_url = get_random_city(winner)
            location_clue = f"\n🌦️📊 We intercepted this message...\n\n{location_clue}\n"
            send_message(target_room_id, location_clue)
            fixed_letters = []
            time.sleep(3)

            image_size = 100

            satellite_view_mxc, satellite_view_width, satellite_view_height = download_image_from_url(satellite_view_url)  
            themed_country_mxc, themed_country_width, themed_country_height = download_image_from_url(themed_country_url)

            if street_view_url != None:
                message = "\n🏙️👁️ We saw OkraStrut post this to X...\n"
                street_view_mxc, street_view_width, street_view_height = download_image_from_url(street_view_url)  
                send_message(target_room_id, message)
                street_response = send_image(target_room_id, street_view_mxc, street_view_width, street_view_height, image_size)
            
                if street_response == False:                      
                    print("Error: Failed to send street image.")
                
                time.sleep(2)
            
            message = "\n🛰️🌍 Our spies tracked him to this area...\n"
            send_message(target_room_id, message)
            satellite_response = send_image(target_room_id, satellite_view_mxc, satellite_view_width, satellite_view_height, image_size)
            
            if satellite_response == False:                      
                print("Error: Failed to send satellite image.")
                
            time.sleep(2)

            message = "\n📸🥒 We found this on OkraStrut's Insta...\n"
            send_message(target_room_id, message)
            themed_response = send_image(target_room_id, themed_country_mxc, themed_country_width, themed_country_height, image_size)
            
            if themed_response == False:                      
                print("Error: Failed to send satellite image.")
                
            time.sleep(2)

        image_mxc, image_width, image_height, display_string = generate_wof_image(wof_answer, wof_clue, fixed_letters)
        print(f"{wof_clue}: {wof_answer}")
            
        image_size = 100
        
        if image_questions == True:    
            response = send_image(target_room_id, image_mxc, image_width, image_height, image_size)
            if response == False:                      
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
                if response == False:                      
                    print("Error: Failed to send image.")
            else:
                wof_letters_str = "Revealed Letters: " + ' '.join(wof_letters)
                message = f"{display_string}\n{wof_clue}\n{wof_letters_str}\n"
                send_message(target_room_id, message)

            process_wof_guesses(winner, wof_answer, 5)

        if selected_wof_category == "5":
            time.sleep(1.5)
            wikipedia_message = f"\n🌐📄 Wikipedia Link: {wiki_url}\n"
            send_message(target_room_id, wikipedia_message)
            time.sleep(1.5)

        if selected_wof_category == "6":
            time.sleep(1.5)
            webster_message = f"\n📚📄 Webster Link: {word_url}\n"
            send_message(target_room_id, webster_message)
            time.sleep(1.5)

        if selected_wof_category == "7":
            time.sleep(1.5)
            webster_message = f"\n📖📄 Webster Link: {word_url}\n"
            send_message(target_room_id, webster_message)
            time.sleep(1.5)

        if selected_wof_category == "8":
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
    global since_token, params, headers, max_retries, delay_between_retries, wf_winner, num_wf_letters

    revealed_count = sum(ch.lower() in "okra" for ch in answer)
    answer_length = length_without_spaces = len(answer.replace(" ", ""))
    letters_remaining = answer_length - revealed_count
    #num_wf_letters = int(letters_remaining * 0.5) - 2

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
    
    selected_question = 0
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

                    if str(message_content) in {"00"}:
                        set_a = ["0", "1", "2", "3", "4"]
    
                        # Possible set for the 10% case (exclude '9' if scoreboard length ≤ 4)
                        if len(round_responders) >= num_list_players:
                            set_b = ["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
                        else:
                            set_b = ["5", "6", "7", "8", "9"]
                    

                        if random.random() < 0.50:
                            selected_question = random.choice(set_a)
                            message = f"\n💪🛡️ I got you @{winner}. {selected_question} it is.\n"
                        else:
                            selected_question = random.choice(set_b)
                            message = f"\n💫🎁 @{winner}, let's do something special. {selected_question} it is.\n"

                        react_to_message(event_id, target_room_id, "okra21")
                        send_message(target_room_id, message)
                        selected_question = selected_question.lower()
                        return selected_question 

                    if str(message_content) in {"5"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Wikipedia Roulette' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"6"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Dictionary Roulette' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"7"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Thesaurus Roulette' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"8"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Where's Okra?' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"9"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'FeUd' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"10"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'FeUd Blitz' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"10"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'FeUd Blitz' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue

                    
                    if str(message_content) in {"12"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Poster Blitz' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"12"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Poster Blitz' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue


                    if str(message_content) in {"13"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Movie Mayhem' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"13"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Movie Mayhem' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"14"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Missing Link' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"14"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Missing Link' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"15"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Famous Peeps' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"15"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Famous Peeps' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"16"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Ranker Lists' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"16"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Ranker Lists' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"17"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Magic Eye D' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"17"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Magic Eye D' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"18"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'OkrAnimal' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"18"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'OkrAnimal' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"19"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'The Riddler' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"19"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'The Riddler' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"20"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Word Nerd' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"20"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Word Nerd' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"21"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Flag Fest' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"21"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'Flag Fest' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"11"} and winner_coffees <= 0:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'List Battle' requires ☕️.\n"
                        send_message(target_room_id, message)
                        continue

                    if str(message_content) in {"11"} and len(round_responders) < num_list_players:
                        react_to_message(event_id, target_room_id, "okra5")
                        message = f"\n🙏😔 Sorry {winner}. 'List Battle' requires {num_list_players}+ players.\n"
                        send_message(target_room_id, message)
                        continue
                        

                    if str(message_content) in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"}:
                        selected_question = str(message_content).lower()
                        react_to_message(event_id, target_room_id, "okra21")
                        message = f"\n💪🛡️ I got you {winner}. {message_content} it is.\n"
                        send_message(target_room_id, message)
                        return selected_question
                    else:
                        react_to_message(event_id, target_room_id, "okra5")
    
        except requests.exceptions.RequestException as e:
                sentry_sdk.capture_exception(e)
                print(f"Error collecting responses: {e}")                    

    
    set_a = ["0", "1", "2", "3", "4"]
    
    # Possible set for the 10% case (exclude '9' if scoreboard length ≤ 4)
    if len(round_responders) >= num_list_players:
        set_b = ["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    else:
        set_b = ["5", "6", "7", "8", "9"]

    # Choose from set_a 90% of the time, set_b 10% of the time
    if random.random() < 0.50:
        selected_question = random.choice(set_a)
    else:
        selected_question = random.choice(set_b)

    send_message(target_room_id, f"\n🐢⏳ Too slow. I choose {selected_question}.\n")
    return selected_question

  
def generate_wof_image(
    phrase,
    clue,
    revealed_letters,
    image_questions=True,
    img_width=800,
    img_height=450,
    base_tile_width=50,
    base_tile_height=70,
    base_spacing=15,
    base_font_size=50,
    base_clue_font_size=40,
    base_revealed_font_size=20,
    max_puzzle_width=700  # how wide the puzzle area can be before scaling or line-wrap
):
    """
    Generate a Wheel of Fortune style puzzle image that:
     - Breaks the puzzle into multiple lines if needed
     - Never breaks a single word across lines
     - Preserves spaces as their own tiles (green squares)
     - If a single word is too wide, scale down tiles+font for the entire puzzle.
    """

    # Convert everything to uppercase for consistent drawing
    phrase = phrase.upper()
    clue = clue.upper()
    revealed_letters = [ch.upper() for ch in revealed_letters]

    # Colors
    background_color         = (0, 0, 0)
    tile_border_color        = (0, 128, 0)
    tile_fill_color          = (255, 255, 255)
    space_tile_color         = (0, 128, 0)
    text_color               = (0, 0, 0)
    clue_color               = (255, 255, 255)
    revealed_letters_color   = (200, 200, 200)

    # Create the base image & drawing context
    img = Image.new('RGB', (img_width, img_height), color=background_color)
    draw = ImageDraw.Draw(img)

    # Load base fonts
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif-Bold.ttf")
    try:
        font           = ImageFont.truetype(font_path, base_font_size)
        clue_font      = ImageFont.truetype(font_path, base_clue_font_size)
        revealed_font  = ImageFont.truetype(font_path, base_revealed_font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return None, None, None, None

    # --------------------------------------------------
    # 1) Split the phrase into chunks, preserving spaces as separate tokens
    #    Example: "HELLO WORLD" -> ["HELLO", " ", "WORLD"]
    # --------------------------------------------------
    # \S+ matches one or more non-whitespace chars
    # \s+ matches one or more whitespace chars (incl. spaces)
    chunks = re.findall(r'\S+|\s+', phrase)
    # Now each chunk is either a word or a run of spaces.

    # --------------------------------------------------
    # 2) Define a function to measure how wide a chunk is in tiles
    #    We measure each character (letter or space) as one tile
    # --------------------------------------------------
    def chunk_tile_width(chunk, tile_w, spacing):
        """
        For a chunk of length L (letters or spaces), we have L tiles,
        so total width = L * (tile_w + spacing) - spacing
        """
        L = len(chunk)
        if L == 0:
            return 0
        return L * (tile_w + spacing) - spacing

    # --------------------------------------------------
    # 3) Check if any single chunk (esp. a word) is too wide for max_puzzle_width
    #    If so, scale down until it fits
    # --------------------------------------------------
    max_chunk_length = max(len(ch) for ch in chunks) if chunks else 0
    unscaled_max_chunk_width = chunk_tile_width("X" * max_chunk_length, base_tile_width, base_spacing)

    if unscaled_max_chunk_width > max_puzzle_width:
        # Scale factor to ensure that the largest chunk fits
        scale_factor = max_puzzle_width / float(unscaled_max_chunk_width)

        # Scale tile sizes, spacing, and fonts
        tile_width     = int(base_tile_width * scale_factor)
        tile_height    = int(base_tile_height * scale_factor)
        spacing        = int(base_spacing * scale_factor)

        tile_width  = max(1, tile_width)
        tile_height = max(1, tile_height)
        spacing     = max(1, spacing)

        scaled_font_size           = max(10, int(base_font_size * scale_factor))
        scaled_clue_font_size      = max(10, int(base_clue_font_size * scale_factor))
        scaled_revealed_font_size  = max(8,  int(base_revealed_font_size * scale_factor))

        # Reload the fonts with scaled sizes
        font          = ImageFont.truetype(font_path, scaled_font_size)
        clue_font     = ImageFont.truetype(font_path, scaled_clue_font_size)
        revealed_font = ImageFont.truetype(font_path, scaled_revealed_font_size)
    else:
        tile_width  = base_tile_width
        tile_height = base_tile_height
        spacing     = base_spacing

    # --------------------------------------------------
    # 4) Break chunks (words/spaces) into multiple lines
    #    without breaking any chunk across lines
    # --------------------------------------------------
    lines = []
    current_line = []
    current_line_width = 0

    for ch in chunks:
        w_width = chunk_tile_width(ch, tile_width, spacing)

        if not current_line:
            # If the line is empty, start with this chunk
            current_line = [ch]
            current_line_width = w_width
        else:
            # If we can add this chunk to the current line
            prospective_width = current_line_width + spacing + w_width
            if prospective_width <= max_puzzle_width:
                current_line.append(ch)
                current_line_width = prospective_width
            else:
                # Start a new line
                lines.append(current_line)
                current_line = [ch]
                current_line_width = w_width

    # Add the last line if non-empty
    if current_line:
        lines.append(current_line)

    # --------------------------------------------------
    # 5) Draw the lines of tiles
    # --------------------------------------------------

    def line_tile_width(line_chunks):
        if not line_chunks:
            return 0
        total = 0
        for idx, c in enumerate(line_chunks):
            cwidth = chunk_tile_width(c, tile_width, spacing)
            if idx == 0:
                total += cwidth
            else:
                total += spacing + cwidth
        return total

    top_margin = 50
    line_spacing_px = tile_height + 20  # gap between lines

    # Figure out total puzzle height
    total_puzzle_height = len(lines) * line_spacing_px
    puzzle_y_start = (img_height - total_puzzle_height) // 2 - 60

    current_y = puzzle_y_start
    padding = 5

    for line_chunks in lines:
        lw = line_tile_width(line_chunks)
        line_start_x = (img_width - lw) // 2
        current_x = line_start_x

        for chunk in line_chunks:
            # draw each character as a tile
            for c in chunk:
                # Draw green rectangle for tile border/padding
                draw.rectangle(
                    [current_x - padding, current_y - padding,
                     current_x + tile_width + padding, current_y + tile_height + padding],
                    fill=tile_border_color
                )
                if c == ' ':
                    # Space tile
                    draw.rectangle(
                        [current_x, current_y, current_x + tile_width, current_y + tile_height],
                        outline=tile_border_color,
                        fill=space_tile_color
                    )
                else:
                    # Letter tile
                    draw.rectangle(
                        [current_x, current_y, current_x + tile_width, current_y + tile_height],
                        outline=tile_border_color,
                        fill=tile_fill_color
                    )
                    if c in revealed_letters:
                        letter_bbox = draw.textbbox((0, 0), c, font=font)
                        letter_w = letter_bbox[2] - letter_bbox[0]
                        letter_h = letter_bbox[3] - letter_bbox[1]
                        text_x = current_x + (tile_width - letter_w) // 2
                        text_y = current_y + (tile_height - letter_h) // 2
                        draw.text((text_x, text_y), c, fill=text_color, font=font)

                current_x += tile_width + spacing

        current_y += line_spacing_px

    # --------------------------------------------------
    # Draw the clue text
    # --------------------------------------------------
    clue_bbox = draw.textbbox((0, 0), clue, font=clue_font)
    clue_w = clue_bbox[2] - clue_bbox[0]
    clue_h = clue_bbox[3] - clue_bbox[1]

    clue_x = (img_width - clue_w) // 2
    clue_y = current_y + 20  
    draw.text((clue_x, clue_y), clue, fill=clue_color, font=clue_font)

    # --------------------------------------------------
    # Revealed letters below the clue
    # --------------------------------------------------
    revealed_text = ' '.join(revealed_letters)
    revealed_bbox = draw.textbbox((0, 0), revealed_text, font=revealed_font)
    revealed_w = revealed_bbox[2] - revealed_bbox[0]
    revealed_x = (img_width - revealed_w) // 2
    revealed_y = clue_y + clue_h + 10
    draw.text((revealed_x, revealed_y), revealed_text, fill=revealed_letters_color, font=revealed_font)

    # --------------------------------------------------
    # Create a puzzle display string (underscores for unrevealed letters)
    # --------------------------------------------------
    # Now that we have chunked the phrase, we can just iterate over the original phrase
    # or reconstruct from chunks.  We'll do from the original phrase to keep it simplest.
    display_string = ' '.join(
        [ch if ch in revealed_letters else ('_' if ch != ' ' else ' ') for ch in phrase]
    )

    # --------------------------------------------------
    # Output/return
    # --------------------------------------------------
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)

    if image_questions:
        image_mxc = upload_image_to_matrix(image_buffer.read())
    else:
        image_mxc = True

    if image_mxc:
        return image_mxc, img_width, img_height, display_string
    else:
        print("Failed to upload the image.")
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

        if response == False:
            print("Error: Failed to send image.")
            print(response)

        return

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running main.py: {e}")
        print("Error output:", e.stderr)


def ask_magic_challenge(winner):
    global since_token, params, headers, max_retries, delay_between_retries

    sync_url = f"{matrix_base_url}/sync"
    
    message = f"👁️✨ Who's got the Magic Eye?"
    send_message(target_room_id, message)
    time.sleep(3)

    user_scores = {}  

    for round_num in range(1, 6):  # Loop for 5 rounds
        magic_number = random.randint(1000, 9999)
        print(f"Magic number for Round {round_num}: {magic_number}")

        message = f"🔵 Round {round_num}: What do you see?\n"
         
        send_magic_image(magic_number)
        send_message(target_room_id, message)

        processed_events = set()
        start_time = time.time()
        magic_number_correct = False

        while time.time() - start_time < magic_time + 10:
            try:
                if since_token:
                    params["since"] = since_token

                response = requests.get(sync_url, headers=headers, params=params)

                if response.status_code != 200:
                    continue

                sync_data = response.json()
                since_token = sync_data.get("next_batch")  # Update since_token

                room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])

                for event in room_events:
                    if magic_number_correct:  # Stop checking if already guessed correctly
                        break

                    event_id = event["event_id"]
                    event_type = event.get("type")

                    if event_type == "m.room.message":
                        if event_id in processed_events:
                            continue  # Skip already processed messages

                        processed_events.add(event_id)
                        sender = event["sender"]
                        sender_display_name = get_display_name(sender)
                        message_content = event.get("content", {}).get("body", "")

                        if sender == bot_user_id:
                            continue  # Ignore bot messages

                        # Check if the user's guess matches the magic number
                        if str(magic_number).lower() in str(message_content).lower():
                            magic_number_correct = True
                            react_to_message(event_id, target_room_id, "okra21")

                            # Update user scores
                            if sender_display_name not in user_scores:
                                user_scores[sender_display_name] = 0
                            user_scores[sender_display_name] += 1

                            message = f"\n🎉🥳 @{sender_display_name} got it right!\n\n👀✨ The Magic Number was **{magic_number}**."
                            send_message(target_room_id, message)
                            break  # Stop checking more messages in this round

            except requests.exceptions.RequestException as e:
                sentry_sdk.capture_exception(e)
                print(f"Error collecting responses: {e}")

        # If no one got it right, announce the correct number
        if not magic_number_correct:
            message = f"\n😢 No one got it right this round!\n\n👀✨ The Magic Number was **{magic_number}**."
            send_message(target_room_id, message)

        time.sleep(3)  # Small delay before next round

    # Final score announcement
    message = "\n🏆✨ **Final Scores** ✨🏆\n"
    sorted_scores = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)

    for rank, (user, score) in enumerate(sorted_scores, start=1):
        message += f"{rank}. @{user}: {score}\n"

    send_message(target_room_id, message)        
    

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
        
        while words:
            word = words[0]
            
            # If the word itself is too long, split it
            while font.getbbox(word)[2] - font.getbbox(word)[0] > max_width:
                # Calculate the maximum number of characters that fit
                for i in range(1, len(word) + 1):
                    if font.getbbox(word[:i])[2] - font.getbbox(word[:i])[0] > max_width:
                        break
                # Add the chunk that fits to the line
                if line:
                    lines.append(line.strip())
                    line = ""
                lines.append(word[:i-1])  # Save the chunk as its own line
                # Update the remaining part of the word
                word = word[i-1:]
            words[0] = word
            
            # Check if adding the next word fits
            if font.getbbox(line + word)[2] - font.getbbox(line + word)[0] <= max_width:
                line += words.pop(0) + " "
            else:
                break
        
        # Append the line to lines
        if line.strip():
            lines.append(line.strip())
    
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
        #prefill_count = int(answer_length * .5) + 1  # At least 1 letter should be filled in
        prefill_count = math.ceil(answer_length * 0.5)
        prefill_positions = random.sample(range(answer_length), prefill_count)
    else:
        prefill_positions = []

    revealed_letters = [answer[i].upper() if i in prefill_positions else '_' for i in range(answer_length)]

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

    # Create the string representation
    display_string = ' '.join(revealed_letters)

    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer

    # Upload the image to Matrix or your media server
    content_uri = upload_image_to_matrix(image_buffer.read())

    # Return the content_uri, image width, height, and the answer
    return content_uri, img_width, img_height, display_string



def process_round_options(round_winner, winner_points):
    global since_token, time_between_questions, time_between_questions_default, ghost_mode, since_token, categories_to_exclude, num_crossword_clues, num_jeopardy_clues, num_mysterybox_clues, num_wof_clues, god_mode, yolo_mode, magic_number, wf_winner, num_math_questions, num_stats_questions, image_questions, nice_okra, creep_okra, marx_mode, blind_mode, seductive_okra, joke_okra
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
    seductive_okra = False
    joke_okra = False
    num_math_questions = num_math_questions_default
    num_stats_questions = num_stats_questions_default
    image_questions = image_questions_default
    marx_mode = marx_mode_default
    blind_mode = blind_mode_default

    if round_winner is None:
        return

    winner_coffees = get_coffees(round_winner)

    #if winner_coffees <= 0:
    #    message = f"\n☕✨ Buy coffee to enable the following options.\n"
    #else:
    message = f"\n🍔🍟 @{round_winner}, what's your order? Some choices require ☕.\n"
    
    send_message(target_room_id, message)

    message = (
        "⏱️⏳ <3 - 15>: Time (s) between questions.\n"
        "🔥🤘 Yolo: No scores shown until the end.\n"
        "🙈🚫 Blind: No question answers shown.\n"
        "🚩🔨 Marx: No recognizing right answers.\n"
        "📷❌ Blank: No image questions.\n"
        "👻🎃 Ghost: Responses will vanish."
    )

    send_message(target_room_id, message)

    message = (
        "🇺🇸🗽 Freedom: No multiple choice. ☕\n"
        "🔢❌ Greg: No math questions. ☕\n"
        "🟦❌ Xela: No Jeopardy-style questions. ☕\n"
        "📰❌ Cross: No Crossword clues. ☕\n"
        "🟦✋ Alex: 5 Jeopardy-style questions. ☕\n"
        "📰✋ Word: 5 Crossword clues. ☕\n"
        "🎖🥒 Dicktator: Choose the categories. ☕\n\n"
    )

    send_message(target_room_id, message)

    #if winner_coffees > 0:
    prompt_user_for_response(round_winner, winner_points, winner_coffees)


def prompt_user_for_response(round_winner, winner_points, winner_coffees):
    global since_token, time_between_questions, ghost_mode, num_jeopardy_clues, num_crossword_clues, num_mysterybox_clues, num_wof_clues, yolo_mode, god_mode, num_math_questions, num_stats_questions, image_questions, marx_mode, blind_mode
    
    # Call initialize_sync to set since_token
    initialize_sync()

    standings = sorted(scoreboard.items(), key=lambda x: x[1], reverse=True)
    num_of_players = len(standings)
    processed_events = set()  # Track processed event IDs to avoid duplicates
    
    # Fetch responses
    sync_url = f"{matrix_base_url}/sync"
    start_time = time.time()  # Track when the question starts
    
    while time.time() - start_time < magic_time:
        try:
            if since_token:
                params["since"] = since_token
                
            response = requests.get(sync_url, headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"Failed to fetch responses. Status code: {response.status_code}")
                return
    
            # Parse the response to get the timeline events
            sync_data = response.json()
            since_token = sync_data.get("next_batch")  # Update the since_token for future requests
            room_events = sync_data.get("rooms", {}).get("join", {}).get(target_room_id, {}).get("timeline", {}).get("events", [])
    
    
            # Process all responses in reverse order (latest response first)
            for event in room_events:
                event_id = event["event_id"]
                event_type = event.get("type")
    
                if event_type == "m.room.message" and event_id not in processed_events:
                    processed_events.add(event_id)
                    sender = event["sender"]
                    sender_display_name = get_display_name(sender)
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

                            if "blind" in message_content.lower():
                                blind_mode = True
                                send_message(target_room_id, f"🙈🚫 @{round_winner} is blind to the truth. No answers will be shown.\n")
        
                            if "marx" in message_content.lower():
                                marx_mode = True
                                send_message(target_room_id, f"🚩🔨 @{round_winner} is a commie. No celebrating right answers.\n")

                            if "yolo" in message_content.lower():
                                yolo_mode = True
                                send_message(target_room_id, f"🤘🔥 Yolo. @{round_winner} says 'don't sweat the small stuff'. No scores till the end.\n")

                            if "blank" in message_content.lower():
                                image_questions = False
                                send_message(target_room_id, f"❌📷 @{round_winner} thinks a word is worth 1000 images.\n")

                            if "ghost" in message_content.lower():
                                ghost_mode = 1
                                send_message(target_room_id, f"\n👻🎃 @{round_winner} says Boo! Your responses will disappear.\n")
        
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
                
                            #if any(word in message_content.lower() for word in ['trebek', 'cross', 'jeopardy', 'word', 'ghost', 'dicktator']) and winner_coffees <= 0:
                            #    react_to_message(event_id, target_room_id, "okra5")
                            #    message = f"\n🙏😔 Sorry {round_winner}. Choice {message_content} requires ☕️.\n"
                            #    send_message(target_room_id, message)
                            #    continue
                            
                            if "freedom" in message_content.lower():
                                if winner_coffees <= 0:
                                    message = f"\n🙏😔 Sorry @{round_winner}. 'Freedom' requires ☕️.\n"
                                else:
                                    num_mysterybox_clues = 0
                                    message = f"\n🇺🇸🗽 @{round_winner} has broken the chains. No multiple choice.\n"
                                send_message(target_room_id, message)
                            
                            if "alex" in message_content.lower():
                                if winner_coffees <= 0:
                                    message = f"\n🙏😔 Sorry @{round_winner}. 'Alex' requires ☕️.\n"
                                else:
                                    num_jeopardy_clues = 5
                                    message = f"\n🟦✋ @{round_winner} wants {num_jeopardy_clues} Jeopardy-style questions.\n"
                                send_message(target_room_id, message)
                
                            if "xela" in message_content.lower():
                                if winner_coffees <= 0:
                                    message = f"\n🙏😔 Sorry @{round_winner}. 'Xela' requires ☕️.\n"
                                else:
                                    num_jeopardy_clues = 0
                                    message = f"\n🟦❌ @{round_winner} doesn't like Jeopardy-style. Sorry Alex.\n"
                                send_message(target_room_id, message)
        
                            if "word" in message_content.lower():
                                if winner_coffees <= 0:
                                    message = f"\n🙏😔 Sorry @{round_winner}. 'Word' requires ☕️.\n"
                                else:
                                    num_crossword_clues = 5
                                    message = f"\n📰✏️ Word. @{round_winner} wants {num_crossword_clues} Crossword questions.\n"
                                send_message(target_room_id, message)

                            if "greg" in message_content.lower():
                                if winner_coffees <= 0:
                                    message = f"\n🙏😔 Sorry @{round_winner}. 'Chad' requires ☕️.\n"
                                else:
                                    num_math_questions = 0
                                    message = f"\n📰✏️ @{round_winner} hates math. What a 'Greg'.\n"
                                send_message(target_room_id, message)
                
                            if "cross" in message_content.lower():
                                if winner_coffees <= 0:
                                    message = f"\n🙏😔 Sorry @{round_winner}. 'Cross' requires ☕️.\n"
                                else:
                                    num_crossword_clues = 0
                                    message = f"\n📰❌ @{round_winner} has crossed off all Crossword questions.\n"
                                send_message(target_room_id, message)
        
                            if "dicktator" in message_content.lower():
                                if winner_coffees <= 0:
                                    message = f"\n🙏😔 Sorry @{round_winner}. 'Dicktator' requires ☕️.\n"
                                else:
                                    god_mode = True
                                    message = f"\n🎖🍆 @{round_winner} is a dick.\n"
                                send_message(target_room_id, message)
            
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
        f"Create a funny and creative dad joke and involve the winner's username '{winner_name}' in your joke. "
        "It should include an exaggerated pun or ridiculous statement about okra."
    )

    try:
        # Use OpenAI's ChatCompletion to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a funny comedian who makes dad jokes about okra."},
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



def generate_trig_question():

    # Generate a random number of 3 or 4 digits in the given base
    trig_operation = random.choice(["sin", "cos", "tan", "cot", "sec", "csc"])
    
    # Convert the number from the input base to decimal

    
    # Create the question text
    question_text = f"What is {trig_operation}(θ) in the triangle below?"
    image_description = f"A triangle with specified angle θ. Sides are opposite (x), adjacent (y), and hypotenuse (z)."

    if trig_operation == "sin":
        new_solution = "y/z"
    elif trig_operation == "cos":
        new_solution = "x/z"
    elif trig_operation == "tan":
        new_solution = "y/x"
    elif trig_operation == "cot":
        new_solution = "x/y"
    elif trig_operation == "sec":
        new_solution = "z/x"
    elif trig_operation == "csc":
        new_solution = "z/y"

    print(f"Question: {question_text}")
    print(f"Answer: {new_solution}")

    content_uri, image_width, image_height = download_image_from_url('https://triviabotwebsite.s3.us-east-2.amazonaws.com/math/triangle.png')

    # Return the content_uri, image dimensions, decimal equivalent, and base number
    return content_uri, image_width, image_height, question_text, new_solution, image_description



def generate_base_question():
    """
    Generate a question asking for the decimal equivalent of a number in a specific base.
    The number will be 3 or 4 digits in the given base, and an image of the number will be sent with the question.
    """
    # Generate a random number of 3 or 4 digits in the given base
    input_base = random.choice([2, 3, 4])
    num_digits = random.randint(3, 4)
    base_number = ''.join(random.choices([str(i) for i in range(input_base)], k=num_digits))
    
    # Convert the number from the input base to decimal
    decimal_equivalent = int(base_number, input_base)
    print(f"Decimal equivalent: {decimal_equivalent}")
    
    # Create the question text
    question_text = f"What is the DECIMAL EQUIVALENT of the following BASE {input_base} number?"
    
    # Create an image with the number
    img_width, img_height = 400, 150
    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load the font
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")
    
    # Adjust the font size based on the length of the base_number
    if len(base_number) > 3:
        font_size = 48  # Reduce font size for longer numbers
    else:
        font_size = 64  # Use larger font for shorter numbers

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return None, None, None, None, None

    # Draw the base number on the image
    text_bbox = draw.textbbox((0, 0), base_number, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img_width - text_width) // 2
    text_y = (img_height - text_height) // 2
    draw.text((text_x, text_y), base_number, fill=(255, 255, 0), font=font)

    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer

    # Upload the image to Matrix (assuming the upload function exists)
    content_uri = upload_image_to_matrix(image_buffer.read()) if image_questions else None

    # Return the content_uri, image dimensions, decimal equivalent, and base number
    return content_uri, img_width, img_height, question_text, str(decimal_equivalent), base_number





def generate_median_question():
    """
    Generate a question asking for the median of a set of random numbers.
    The set will contain 3 to 7 numbers between 1 and 20, and the image
    of the numbers will be sent to the user with the question.
    """
    # Generate a random n between 3 and 7
    content_uri = True
    n = random.randint(3, 5)
    
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
    global nice_okra, creep_okra, wf_winner, seductive_okra, joke_okra
    #ask_magic_number(winner) 

    if skip_summary == True:
        summary = "Make sure to drink your Okratine."
        return summary

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
            user_data = get_user_data(winner, 5)
            if not user_data:
                print(f"Failed to fetch data for {winner}.")
                user_data = {"avatar_url": "N/A", "subreddits": ""}
        except Exception as e:
            print(f"Error fetching Reddit data for {winner}: {e}")
            user_data = {"avatar_url": "N/A", "subreddits": ""}
    
        # Extract user data
        reddit_avatar_url = user_data.get("avatar_url", "No avatar available.")
        reddit_avatar_description = describe_image_with_vision(reddit_avatar_url, "describe", "")
        top_subreddits = user_data.get("top_subreddits", "")
        
        prompt = (
            f"{winner_at} is the username of the winner of the trivia round. "
            f"Roast and deeply insult them about their username, {winner}, their avatar description, {reddit_avatar_description}, and their top 5 subreddits, which are '{top_subreddits}'. Be very mean, negative, and use rough language. Be sarcastic and use emojis in your response. "
            f"Create no more than 10 sentences in total.\n"
        )

    elif joke_okra == True:
        joke = generate_okra_joke(winner)
        return joke

    elif seductive_okra == True and is_sovereign == True:
         prompt = (
            f"{winner_at} is the username of the winner of the trivia round. "
            f"Start by giving {winner_at} kudos for being a previous Sovereign. Then seduce {winner_at} using multiple pickup lines customized to their username: {winner}. Also mention how sexy specific answers they gave during the round were. "
            f"Be uncomfortably and embarrasingly forward in your approaches in trying to get them to go out with you. "
            "Create no more than 4 sentences in total. Here is a detailed summary of the trivia round with explicit mappings of their responses:\n"
            "Questions asked:\n"
        )
    
    elif seductive_okra == True and is_sovereign == False:
         prompt = (
            f"{winner_at} is the username of the winner of the trivia round. "
            f"Seduce {winner_at} using multiple pickup lines customized to their username: {winner}. Also mention how sexy specific answers they gave during the round were. "
            f"Be uncomfortably and embarrasingly forward in your approaches in trying to get them to go out with you. "
            "Create no more than 4 sentences in total. Here is a detailed summary of the trivia round with explicit mappings of their responses:\n"
            "Questions asked:\n"
        )

    elif wf_winner == True and is_sovereign == True:
         prompt = (
            f"{winner_at} is the username of the winner of the trivia round. "
            "Love bomb them about their username and be very specific, positive, and loving. Give them a lot of admiration for being a previous Sovereign. Then mention and compliment specific responses they gave during the round. Also mention about how much beter they are than eveyone else including yourself, who is the great OkraStrut. "
            "Create no more than 4 sentences in total. Be sweet, happy, positive, and use emojis in your response. "
            "Here is a detailed summary of the trivia round with explicit mappings of their responses:\n"
            "Questions asked:\n"
        )

    elif wf_winner == True and is_sovereign == False:
         prompt = (
            f"{winner_at} is the username of the winner of the trivia round. "
            "Love bomb them about their username and be very specific, positive, and loving. Specifically mention and compliment specific responses they gave during the round. Also mention about how much beter they are than eveyone else including yourself, who is the great OkraStrut."
            "Create no more than 4 sentences in total. Be sweet, happy, positive, and use emojis in your response. "
            "Here is a detailed summary of the trivia round with explicit mappings of their responses:\n"
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
            prompt += f"Correct Answers: {', '.join(correct_answers_str)}\n"
            
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

        elif seductive_okra == True:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a sleazy man trying to come onto the winner of a trivia game. You use cheesy pick up lines and are embarassingly forward in your approaches. Make the winner uncomfortable and be ruthless in your seduction. Use plenty of emojis for flair, but stay within 8 sentences."},
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
    global client, db
    client = None
    db = None
    
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
    return db

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
    return False  # Return the last response, even if it failed


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

    trivia_answer = trivia_answer_list[0]  # The first item is the main answer

    single_answer = (
        (len(trivia_answer_list) == 1 and (is_number(trivia_answer) or len(trivia_answer) == 1)) or
        trivia_url in [
            "median", "mean", "zeroes sum", "zeroes product", "zeroes", "base", "factors",
            "derivative", "trig", "algebra"
        ]
    )

    message_body = ""
    if single_answer:
        message_body += "\n🚨 1 GUESS 🚨"
        
    if is_valid_url(trivia_url): 
        image_mxc, image_width, image_height = download_image_from_url(trivia_url) 
        message_body += f"\n{number_block}📷 {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
        send_image_flag = True

    elif trivia_url == "algebra":
        image_mxc, image_width, image_height, new_question, new_solution, text_problem = generate_and_render_linear_problem()
        if image_questions == True:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{new_question}\n" 
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{new_question}\n{text_problem}\n"
    
    elif trivia_url == "trig":
        image_mxc, image_width, image_height, new_question, new_solution, img_description = generate_trig_question()
        if image_questions == True:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{new_question}\n" 
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{new_question}\n{img_description}\n"

    elif trivia_url == "base":
        image_mxc, image_width, image_height, new_question, new_solution, base_string = generate_base_question()
        if image_questions == True:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{new_question}\n" 
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{new_question}\n{base_string}\n"
    
    elif trivia_url == "zeroes sum":
        image_mxc, image_width, image_height, new_solution, polynomial = generate_and_render_polynomial(trivia_url)
        if image_questions == True:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n" 
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n{polynomial}\n"

    elif trivia_url == "characters":
        message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\nName the movie, book, or show:\n\n{trivia_question}\n"

    elif trivia_url == "zeroes product":
        image_mxc, image_width, image_height, new_solution, polynomial = generate_and_render_polynomial(trivia_url)
        if image_questions == True:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n" 
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n{polynomial}\n"

    elif trivia_url == "zeroes":
        image_mxc, image_width, image_height, new_solution, polynomial = generate_and_render_polynomial(trivia_url)
        if image_questions == True:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n" 
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n{polynomial}\n"

    elif trivia_url == "factors":
        image_mxc, image_width, image_height, new_solution, polynomial = generate_and_render_polynomial(trivia_url)
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
        image_mxc, image_width, image_height, string_representation = generate_crossword_image(trivia_answer_list[0])
        if image_questions == True: 
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n[{len(trivia_answer_list[0])} Letters] {trivia_question}\n"
            send_image_flag = True
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n[{len(trivia_answer_list[0])} Letters] {trivia_question}\n\n{string_representation}\n"
        
    elif trivia_url == "multiple choice" or trivia_url == "multiple choice opentrivia" or trivia_url == "multiple choice oracle": 
        if trivia_answer_list[0] in {"True", "False"}:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n🚨 T/F - 1 GUESS 🚨 {trivia_question}\n\n"
        else:
            message_body += f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n🚨 Letter - 1 GUESS 🚨 {trivia_question}\n\n"
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

        if response == False:                      
            print("Error: Failed to send image.")
            return None, None, None
            
    initialize_sync()
    

    if new_solution is None:
    # Use the original trivia answer list if no new solution is provided
        correct_answers = trivia_answer_list
    elif isinstance(new_solution, list):
        # If new_solution is already a list, use it as-is
        correct_answers = new_solution
    else:
        # If new_solution is a single value, wrap it in a list
        correct_answers = [new_solution]
    
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
    # Max points is 1000, min points is 0. Points decrease linearly over question_time
    points = max(1000 - int(response_time * (995 / question_time)), 5)
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
    response = response.lower()
    answer = answer.lower()
    response = response.replace(" ", "")      
    answer = answer.replace(" ", "")
    response = response.replace("^", "")      
    answer = answer.replace("^", "")
    response = response.replace("*", "")      
    answer = answer.replace("*", "")
    response = normalize_superscripts(response)
    answer = normalize_superscripts(answer)

    if (response == answer or jaccard_similarity(response, answer) == 1) and len(response) == len(answer):
        return True
    else:
        return False

def factors_checker(response, answer):
    response = response.lower()
    answer = answer.lower()
    response = response.replace(" ", "")      
    answer = answer.replace(" ", "")
    response = response.replace("*", "")      
    answer = answer.replace("*", "")

    if (response == answer or jaccard_similarity(response, answer) == 1) and len(response) == len(answer):
        return True
    else:
        return False


def trig_checker(response, answer):
    response = response.lower()
    answer = answer.lower()
    response = response.replace(" ", "")      
    answer = answer.replace(" ", "")
    response = response.replace("(", "")      
    answer = answer.replace("(", "")
    response = response.replace(")", "")      
    answer = answer.replace(")", "")

    if response == answer:
        return True
    else:
        return False


def fuzzy_match(user_answer, correct_answer, category, url):
    threshold = 0.90    

    if user_answer == correct_answer:
        return True

    no_spaces_user = user_answer.replace(" ", "")      
    no_spaces_correct = correct_answer.replace(" ", "") 

    if category == "Crossword":
        return no_spaces_user.lower() == no_spaces_correct.lower()

    if url == "zeroes":
        user_numbers = [int(num) for num in re.findall(r'-?\d+', user_answer)]
        correct_numbers = [int(num) for num in re.findall(r'-?\d+', correct_answer)]
        
        # Check if the two sets of numbers match (order does not matter)
        if set(user_numbers) == set(correct_numbers):
            return True
        else:
            return False

    if url == "derivative":
        return derivative_checker(user_answer, correct_answer)

    if url == "factors":
        return factors_checker(user_answer, correct_answer)

    if url == "trig":
        return trig_checker(user_answer, correct_answer)
    
        
    if is_number(correct_answer):
        return user_answer == correct_answer  # Only accept exact match if the correct answer is a number
    
    user_answer = normalize_text(str(user_answer))
    correct_answer = normalize_text(str(correct_answer))

    if url == "multiple choice" or url == "multiple choice opentrivia" or  url == "multiple choice oracle":
        return user_answer[0] == correct_answer[0];
    
    if is_number(correct_answer):
        return user_answer == correct_answer  # Only accept exact match if the correct answer is a number

    no_spaces_user = user_answer.replace(" ", "")      
    no_spaces_correct = correct_answer.replace(" ", "") 

    no_filler_user = remove_filler_words(user_answer)
    no_filler_correct = remove_filler_words(correct_answer)

    no_filler_spaces_user = no_filler_user.replace(" ", "")
    no_filler_spaces_correct = no_filler_correct.replace(" ", "")

    if no_spaces_user == no_spaces_correct or no_filler_user == no_filler_correct or no_filler_spaces_user == no_filler_spaces_correct:     
        return True

    if len(user_answer) < 4:
        return user_answer == correct_answer  # Only accept an exact match for short answers
    
    if user_answer == correct_answer:
        return True
    
         
    # New Step: First 5 characters match
    if user_answer[:5] == correct_answer[:5] or no_spaces_user[:5] == no_spaces_correct[:5] or no_filler_user[:5] == no_filler_correct[:5] or no_filler_spaces_user[:5] == no_filler_spaces_correct[:5]:
        return True
    
    # Remove filler words and split correct answer
    correct_answer_words = correct_answer.split()
    no_filler_answer_words = no_filler_correct.split()
    
    # Ensure correct_answer_words is not empty
    if correct_answer_words and len(correct_answer_words[0]) >= 3:
        if user_answer == correct_answer_words[0] or no_filler_user == correct_answer_words[0]:
            return True

    if no_filler_answer_words and len(no_filler_answer_words[0]) >= 3:
        if user_answer == no_filler_answer_words[0] or no_filler_user == no_filler_answer_words[0]:
            return True

    #Check if user's answer is a substring of the correct answer after normalization
    if user_answer in correct_answer:
        return True
    
    # Step 1: Exact match or Partial match
    if correct_answer in user_answer:
        return True
    
    # Step 2: Levenshtein similarity
    if levenshtein_similarity(user_answer, correct_answer) >= threshold or levenshtein_similarity(no_spaces_user, no_spaces_correct) >= threshold or levenshtein_similarity(no_filler_user, no_filler_correct) >= threshold or levenshtein_similarity(no_filler_spaces_user, no_filler_spaces_correct) >= threshold:
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
    global question_responders, round_responders, discount_percentage
    
    # Define the first item in the list as trivia_answer
    trivia_answer = trivia_answer_list[0]  # The first item is the main answer
    correct_responses = []  # To store users who answered correctly
    has_responses = False  # Track if there are any responses

    fastest_correct_user = None
    fastest_response_time = None
    fastest_correct_event_id = None

    # Check if trivia_answer_list is a single-element list with a numeric answer  
    single_answer = (
        (len(trivia_answer_list) == 1 and (is_number(trivia_answer) or len(trivia_answer) == 1)) or
        trivia_url in [
            "multiple choice opentrivia", "multiple choice oracle", "multiple choice",
            "median", "mean", "zeroes sum", "zeroes product", "zeroes", "base", "factors",
            "derivative", "trig", "algebra"
        ]
    )
    
    # Dictionary to track first numerical response from each user if answer is a number
    user_first_response = {}

    # Process collected responses
    for response in collected_responses:
        sender = response["user_id"]
        event_id = response["event_id"]
        display_name = get_display_name(sender)  # Get the display name from content
        
        message_content = response.get("message_content", "")  # Use 'response' instead of 'event'
        message_content = message_content.replace("\uFFFC", "")  # Remove U+FFFC

        # Track users who responded to the current question and round
        if display_name not in question_responders:
            question_responders.append(display_name)  # Add to question responders
        
            # Only add to round responders if not already present
            if display_name not in round_responders:
                round_responders.append(display_name)

        if "okra" in message_content.lower() and emoji_mode == True:
            react_to_message(event_id, target_room_id, "okra1")

        if "#prev" in message_content.lower() and collect_feedback_mode == True and sender != bot_user_id:
            if emoji_mode == True:
                react_to_message(event_id, target_room_id, "okra3")
            #stripped_message_content = " ".join(message_content.split()[1:])
            #insert_audit_question("audit_questions", previous_question, message_content, display_name)
            update_audit_question(previous_question, message_content, display_name)

        if "#curr" in message_content.lower() and collect_feedback_mode == True and sender != bot_user_id:
            if emoji_mode == True:
                react_to_message(event_id, target_room_id, "okra3")
            #stripped_message_content = " ".join(message_content.split()[1:])
            #insert_audit_question("audit_questions", current_question, message_content, display_name)
            update_audit_question(current_question, message_content, display_name)

        # Check if the user has already answered correctly, ignore if they have
        if any(resp[0] == display_name for resp in correct_responses):
            continue  # Ignore this response since the user has already answered correctly
        
        # If it's a single numeric answer question, and this user's response is numeric, only record the first one
        if single_answer:
            if display_name in user_first_response:
                continue  # Skip if we've already recorded a numeric response for this user
        
            if (
                is_number(message_content) or  # Rule 1: message_content is a number
                message_content[0].isdigit() or  # Rule 2: first character is a number
                message_content.lower() in {"a", "b", "c", "d", "t", "f", "true", "false"} or  # Rule 3: exact match
                message_content[0].lower() in {"-", "x", "y", "z", "("} or # Rule 4: first character match
                len(message_content) == 1
            ):
                user_first_response[display_name] = message_content
            else:
                continue  # Skip non-numeric responses for single numeric questions
        
        # Log user submission (MongoDB operation)
        #log_user_submission(display_name)
                
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

            # Check if the sender is the current user on the longest round streak
            if display_name == current_longest_round_streak["user"]:
                streak = current_longest_round_streak["streak"]
                # For every 5 in the streak, apply a 10% discount
                discount_percentage = discount_step_amount * (streak // discount_streak_amount)  # e.g., 5 => 10%, 10 => 20%, 15 => 30%, etc.

                # You might want to cap the discount so it doesn't go negative or too high
                discount_percentage = min(discount_percentage, 90)  # optional
        
                if discount_percentage > 0:
                    discount_factor = 1 - (discount_percentage / 100.0)
                    points *= discount_factor
                    points = round(points / 5) * 5

            correct_responses.append((display_name, points, response_time, message_content))
    
            # Check if this is the fastest correct response so far
            if fastest_correct_user is None or response_time < fastest_response_time:
                fastest_correct_user = display_name
                fastest_response_time = response_time
                fastest_correct_event_id = event_id
            
    if emoji_mode == True and fastest_correct_event_id is not None and blind_mode == False and marx_mode == False:
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
    message = ""
    if blind_mode == False:
        message = f"\n✅ Answer ({len(question_responders)}) ✅\n{trivia_answer}\n"
            
    # Notify the chat
    if correct_responses and marx_mode == False:    
        correct_responses_length = len(correct_responses)
        
        # Loop through the responses and append to the message
        for display_name, points, response_time, message_content in correct_responses:
            time_diff = response_time - fastest_response_time
            
            name_str = display_name
            if current_longest_round_streak["user"] == display_name and discount_percentage is not None and discount_percentage > 0:
                name_str += f" (-{discount_percentage}%)"
        
            # Display the formatted message based on yolo_mode
            if time_diff == 0:
                message += f"\n⚡ {name_str}"
                if not yolo_mode:
                    message += f": {points}"
                if points == 420:
                    message += " 🌿"
                if points == 690:
                    message += " 😎"
                if current_longest_answer_streak["streak"] > 1:
                    message += f"  🔥{current_longest_answer_streak['streak']}"
            else:
                message += f"\n👥 {name_str}"
                if not yolo_mode:
                    message += f": {points}"
                if points == 420:
                    message += " 🌿"
                if points == 690:
                    message += " 😎"

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
        streak = current_longest_round_streak["streak"]
        if streak > 1:
            message = f"\n🏆 Winner: @{user}...🔥{current_longest_round_streak['streak']} in a row!\n"
            
            if streak % discount_streak_amount == 0:
                # Compute discount percentage
                # e.g. if streak=10, discount_streak_amount=5, discount_step_amount=0.1 => 2 * 10% = 20%
                discount_fraction = min((streak // discount_streak_amount) * discount_step_amount, 90)
                message += f"\n⚖️ Going forward @{user} will incur a -{discount_fraction}% handicap.\n"
                
            message += f"\n▶️ Live trivia stats available: https://livetriviastats.com\n"

        else:
            message = f"\n🏆 Winner: @{user}!\n\n▶️ Live trivia stats available: https://livetriviastats.com\n"

        send_message(target_room_id, message)
        time.sleep(2)
        
        select_wof_questions(user)
        
        gpt_summary = generate_round_summary(round_data, user)

        print(gpt_summary)

        gpt_message = f"\n{gpt_summary}\n"
        send_message(target_room_id, gpt_message)


        highest_score_player = max(scoreboard, key=scoreboard.get)  # Player with the highest score
        highest_score = scoreboard[highest_score_player]  # The highest score itself

        
        if len(scoreboard) >= image_wins and highest_score > image_points:
        #if current_longest_round_streak['streak'] % image_wins == 0:
            time.sleep(5)
            generate_round_summary_image(round_data, user)
        else:
            #number_to_emoji = {
            #    1: "1️⃣",
            #    2: "2️⃣",
            #    3: "3️⃣",
            #    4: "4️⃣",
            #    5: "5️⃣",
            #    6: "6️⃣",
            #    7: "7️⃣",
            #    8: "8️⃣",
            #    9: "9️⃣",
            #    10: "🔟"
            #}
            
            time.sleep(4)
            #remaining_games = image_wins - (current_longest_round_streak['streak'] % image_wins)
            #dynamic_emoji = number_to_emoji[remaining_games]
            #dynamic_emoji = number_to_emoji.get(remaining_games, "❗")  # ❓ is the default emoji
            
            #if remaining_games == 1:
            #    image_message = f"\n{dynamic_emoji}🎨 @{user} Win the next game and I'll draw you something.\n"
            #else:
            #    image_message = f"\n{dynamic_emoji}🎨 @{user} Win {remaining_games} more in a row and I'll draw you something.\n"

            image_message = "\n🖼️✨ A memory from the Okra Museum"
            image_message += "\n🥒🏛️ https://livetriviastats.com/okra-museum\n"
            #if len(scoreboard) < image_wins and highest_score > image_points:
            #    image_message += f"\n🌟😞 @{user} Awesome score! But we need some more compeition.\n"
            #if len(scoreboard) >= image_wins and highest_score < image_points:
            #    image_message += f"\n🌟😞 @{user} You emerged at the top! But your score could be higher.\n"
            #else:
            #    image_message += f"\n🌟😞 @{user} You won! But more points (and more players).\n" 

            send_message(target_room_id, image_message)
            get_image_url_from_s3()
            
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
        standing_message = f"\n📈 Scoreboard ({len(round_responders)}) 📈"
        
        # Define the medals for the top 3 positions
        medals = ["🥇", "🥈", "🥉"]
        
        for rank, (user, points) in enumerate(standings, start=1):
            formatted_points = f"{points:,}"  # Format points with commas
            fastest_count = fastest_answers_count.get(user, 0)  # Get the user's fastest answer count, default to 0

            # Start by building a user string (possibly with discount info)
            user_str = user

            # Conditionally show discount, if the user matches the streak user
            # and discount_percentage > 0
            if current_longest_round_streak["user"] == user and discount_percentage > 0 and discount_percentage is not None:
                # Convert decimal discount (e.g., 0.3) to integer percent (30)
                user_str += f" (-{discount_percentage}%)"

            lightning_display = f" ⚡{fastest_count}" if fastest_count > 1 else " ⚡" if fastest_count == 1 else ""
            
            if "420" in str(points):
                standing_message += f"\n🌿 {user_str}: {formatted_points}"

            elif "69" in str(points):
                standing_message += f"\n😎 {user_str}: {formatted_points}"
                
            elif rank <= 3:
                standing_message += f"\n{medals[rank-1]} {user_str}: {formatted_points}"
                
            elif rank == len(standings) and rank > 4:
                standing_message += f"\n💩 {user_str}: {formatted_points}"
                
            else:
                standing_message += f"\n{rank}. {user_str}: {formatted_points}"

            standing_message += lightning_display
        
        send_message(target_room_id, standing_message)
        

def store_question_ids_in_mongo(question_ids, question_type):
    db = connect_to_mongodb()
    collection_name = f"asked_{question_type}_questions"
    questions_collection = db[collection_name]

    for _id in question_ids:
        # Use upsert to insert or update the document if it doesn't exist
        questions_collection.update_one(
            {"_id": _id},                  # Match the document by its _id
            {"$setOnInsert": {"_id": _id, "timestamp": datetime.datetime.now(datetime.UTC)}},  # Insert only if not present
            upsert=True                    # Enable upsert behavior
        )

    # Check if the collection exceeds its limit and delete old entries if necessary
    limit = id_limits[question_type]
    total_ids = questions_collection.count_documents({})
    if total_ids > limit:
        excess = total_ids - limit
        # Find the oldest entries based on timestamp
        oldest_entries = questions_collection.find().sort("timestamp", 1).limit(excess)
        for entry in oldest_entries:
            questions_collection.delete_one({"_id": entry["_id"]})


def get_recent_question_ids_from_mongo(question_type):
    db = connect_to_mongodb()
    collection_name = f"asked_{question_type}_questions"
    questions_collection = db[collection_name]

    recent_ids = questions_collection.find().sort("timestamp", -1).limit(id_limits[question_type])
    return {doc["_id"] for doc in recent_ids}



def get_all_recent_question_ids():
    """
    Fetch recent question IDs for all question types in a single call.
    Returns a dictionary with question type as the key and recent IDs as the value.
    """
    recent_ids = {}
    for question_type in ["general", "crossword", "jeopardy", "mysterybox", "wof"]:
        collection_name = f"asked_{question_type}_questions"
        questions_collection = db[collection_name]
        recent_ids[question_type] = {
            doc["_id"]
            for doc in questions_collection.find().sort("timestamp", -1).limit(id_limits[question_type])
        }
    return recent_ids


def store_all_question_ids(question_ids_by_type):
    """
    Store question IDs for all question types in a single call.
    Expects a dictionary with question type as the key and list of IDs as the value.
    """
    for question_type, question_ids in question_ids_by_type.items():
        if not question_ids:
            continue
        collection_name = f"asked_{question_type}_questions"
        questions_collection = db[collection_name]

        for _id in question_ids:
            # Use upsert to insert or update the document if it doesn't exist
            questions_collection.update_one(
                {"_id": _id},  # Match the document by its _id
                {"$setOnInsert": {"_id": _id, "timestamp": datetime.datetime.now(datetime.UTC)}},  # Insert only if not present
                upsert=True  # Enable upsert behavior
            )

        # Check if the collection exceeds its limit and delete old entries if necessary
        limit = id_limits[question_type]
        total_ids = questions_collection.count_documents({})
        if total_ids > limit:
            excess = total_ids - limit
            # Find the oldest entries based on timestamp
            oldest_entries = questions_collection.find().sort("timestamp", 1).limit(excess)
            for entry in oldest_entries:
                questions_collection.delete_one({"_id": entry["_id"]})


def select_trivia_questions(questions_per_round):
    global categories_to_exclude
    try:

        recent_question_ids = get_all_recent_question_ids()
        selected_questions = []
        question_ids_to_store = {  # Initialize a dictionary to batch store question IDs
            "general": [],
            "crossword": [],
            "jeopardy": [],
            "mysterybox": [],
            "wof": []
        }
        
        sample_size = min(num_crossword_clues, questions_per_round - len(selected_questions))
        if sample_size > 0:
            crossword_collection = db["crossword_questions"]
            pipeline_crossword = [
                {"$match": {"_id": {"$nin": list(recent_question_ids["crossword"])}}},
                {"$sample": {"size":sample_size}}  # Apply sampling on the filtered subset
            ]
            crossword_questions = list(crossword_collection.aggregate(pipeline_crossword))

            for doc in crossword_questions:
                doc["db"] = "crossword_questions"
                
            selected_questions.extend(crossword_questions)
            question_ids_to_store["crossword"].extend(doc["_id"] for doc in crossword_questions)

        sample_size = min(num_jeopardy_clues, questions_per_round - len(selected_questions))
        if sample_size > 0:
            jeopardy_collection = db["jeopardy_questions"]
            pipeline_jeopardy = [
                {"$match": {"_id": {"$nin": list(recent_question_ids["jeopardy"])}}},
                {"$sample": {"size": sample_size}}  # Apply sampling on the filtered subset
            ]
            jeopardy_questions = list(jeopardy_collection.aggregate(pipeline_jeopardy))

            for doc in jeopardy_questions:
                doc["db"] = "jeopardy_questions"
                
            selected_questions.extend(jeopardy_questions)
            question_ids_to_store["jeopardy"].extend(doc["_id"] for doc in jeopardy_questions)

        
        num_math_questions_mod = random.randint(0, num_math_questions)
        sample_size = min(num_math_questions_mod, questions_per_round - len(selected_questions))
        if sample_size > 0:
            math_questions = [get_math_question() for _ in range(sample_size)]

            for doc in math_questions:
                doc["db"] = "math_questions"
                doc["_id"] = str(random.randint(10000, 99999))
                
            selected_questions.extend(math_questions)

        sample_size = min(num_stats_questions, questions_per_round - len(selected_questions))
        if sample_size > 0:
            stats_questions = [get_stats_question() for _ in range(sample_size)]

            for doc in stats_questions:
                doc["db"] = "stats_questions"
                doc["_id"] = str(random.randint(10000, 99999))
                
            selected_questions.extend(stats_questions)

        sample_size = min(num_wof_clues, questions_per_round - len(selected_questions))
        if sample_size > 0:
            wof_collection = db["wof_questions"]
            pipeline_wof = [
                {"$match": {"_id": {"$nin": list(recent_question_ids["wof"])}}},
                {"$sample": {"size": sample_size}}  # Apply sampling on the filtered subset
            ]
            wof_questions = list(wof_collection.aggregate(pipeline_wof))

            for doc in wof_questions:
                doc["db"] = "wof_questions"
                
            selected_questions.extend(wof_questions)
            question_ids_to_store["wof"].extend(doc["_id"] for doc in wof_questions)
 
        sample_size = min(num_mysterybox_clues, questions_per_round - len(selected_questions))
        if sample_size > 0:
            mysterybox_collection = db["mysterybox_questions"]
            pipeline_mysterybox = [
                {"$match": {"_id": {"$nin": list(recent_question_ids["mysterybox"])}}},
                {"$sample": {"size": sample_size}}  # Apply sampling on the filtered subset
            ]
            mysterybox_questions = list(mysterybox_collection.aggregate(pipeline_mysterybox))

            for doc in mysterybox_questions:
                doc["db"] = "mysterybox_questions"
            
            selected_questions.extend(mysterybox_questions)
            question_ids_to_store["mysterybox"].extend(doc["_id"] for doc in mysterybox_questions)
        
        sample_size = max(questions_per_round - len(selected_questions), 0)
        if sample_size > 0:
            trivia_collection = db["trivia_questions"]

            if image_questions == False:
                # Define a list of substrings to exclude in URLs
                excluded_url_substring = "http"
                pipeline_trivia = [
                    {
                        "$match": {
                            "_id": {"$nin": list(recent_question_ids["general"])},
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
                    {"$sample": {"size": sample_size}}  # Sample from the resulting limited set
                ]
                
            else:
                pipeline_trivia = [
                    {"$match": {"_id": {"$nin": list(recent_question_ids["general"])}, "category": {"$nin": categories_to_exclude}}},
                    {
                        "$group": {
                            "_id": "$category",
                            "questions": {"$push": "$$ROOT"}  # Push full document to each category group
                        }
                    },
                    {"$unwind": "$questions"},  # Unwind the limited question list for each category back into individual documents
                    {"$replaceRoot": {"newRoot": "$questions"}},  # Flatten to original document structure
                    {"$sample": {"size": sample_size}}  # Sample from the resulting limited set
                ]

            trivia_questions = list(trivia_collection.aggregate(pipeline_trivia))

            for doc in trivia_questions:
                doc["db"] = "trivia_questions"
                
            selected_questions.extend(trivia_questions)
            question_ids_to_store["general"].extend(doc["_id"] for doc in trivia_questions)

        
        # Shuffle the combined list of selected questions
        random.shuffle(selected_questions)

        # Store question IDs in MongoDB (batch operation)
        store_all_question_ids(question_ids_to_store)

        final_selected_questions = [
            (doc["category"], doc["question"], doc["url"], doc["answers"], doc["db"], doc["_id"])
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
    top_users = list(db.top_users.find())
    sovereigns = {sovereign['user'] for sovereign in db.hall_of_sovereigns.find()}

    messages = []
    for user in top_users:
        username = user.get('user')
        top_count = user.get('top_count')

        # If the user is in the Hall of Sovereigns, only show the message if top_count == 6
        if username in sovereigns:
            if top_count == 6:
                send_message(target_room_id, f"👑  {username} is #1 across the board. We bow to you.\n\n▶️ Live trivia stats available: https://livetriviastats.com\n")
        else:
            # For users not in the Hall of Sovereigns, show all applicable messages
            if top_count == 6:
                send_message(target_room_id, f"👑  {username} is #1 across the board. We bow to you.\n\n▶️ Live trivia stats available: https://livetriviastats.com\n")
            elif top_count == 5:
                send_message(target_room_id, f"🔥​  {username} is on fire! Only 1 leaderboard left.\n\n▶️ Live trivia stats available: https://livetriviastats.com\n")
            elif top_count == 4:
                send_message(target_room_id, f"🌡️  {username} is heating up! Only 2 leaderboards left.\n\n▶️ Live trivia stats available: https://livetriviastats.com\n")
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


def generate_and_render_linear_problem():
    # Generate coefficients ensuring no value is zero
    while True:
        a = random.choice([i for i in range(-10, 11) if i != 0])  # Coefficient of x (-10 to 10, excluding 0)
        x = random.choice([i for i in range(-20, 21) if i != 0])  # Integer solution (-20 to 20, excluding 0)
        b = random.choice([i for i in range(-20, 21) if i != 0])  # Constant term (-20 to 20, excluding 0)
        if a != 0 and x != 0 and b != 0:
            break

    question_text = f"Solve for 'x' in the equation below."

    # Compute the constant on the other side of the equation
    c = a * x + b

    # Format the coefficient of x
    if a == 1:
        a_str = ""  # Ignore 1
    elif a == -1:
        a_str = "-"  # Use only "-"
    else:
        a_str = str(a)

    # Formulate the problem as a linear equation
    problem = f"{a_str}x {'+' if b >= 0 else '-'} {abs(b)} = {c}"
    solution = f"{x}"

    print(f"Problem: {problem}")
    print(f"Solution: {solution}")

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
        return None, None, None, None

    # Draw the problem text in the center in light purple
    text_bbox = draw.textbbox((0, 0), problem, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img_width - text_width) // 2
    text_y = (img_height - text_height) // 2
    draw.text((text_x, text_y), problem, fill=(200, 162, 200), font=font)  # Light purple color

    # Save the image to a bytes buffer
    image_buffer = io.BytesIO()
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Move the pointer to the beginning of the buffer

    # Upload the image to Matrix (if enabled)
    if image_questions == True:
        content_uri = upload_image_to_matrix(image_buffer.read())
    else:
        content_uri = True  # Mock successful upload

    if content_uri:
        return content_uri, img_width, img_height, question_text, solution, problem
    else:
        print("Failed to upload the image to Matrix.")
        return None, None, None, None



def generate_and_render_polynomial(type):
    # Randomly select two unique integers from -9 to 9, excluding 0
    content_uri = True
    zero1 = random.choice([i for i in range(-9, 10) if i != 0])
    zero2 = random.choice([i for i in range(-9, 10) if i != 0 and i != zero1])

    sum_zeroes = zero1 + zero2
    product_zeroes = zero1 * zero2
    # Format the factors
    factor1 = f"(x {'+' if zero1 < 0 else '-'} {abs(zero1)})"
    factor2 = f"(x {'+' if zero2 < 0 else '-'} {abs(zero2)})"
    factor1_mod = f"x {'+' if zero1 < 0 else '-'} {abs(zero1)}"
    factor2_mod = f"x {'+' if zero2 < 0 else '-'} {abs(zero2)}"
    

    # Construct the sum term for the polynomial
    if abs(sum_zeroes) == 1:
        sum_term = ""
    else:
        sum_term = abs(sum_zeroes)

    if sum_term == 0:
        polynomial = f"x² {'+' if product_zeroes >= 0 else '-'} {abs(product_zeroes)}"
    else:
        polynomial = f"x² {'-' if sum_zeroes >= 0 else '+'} {sum_term}x {'+' if product_zeroes >= 0 else '-'} {abs(product_zeroes)}"
    
    print(f"Polynomial: {polynomial}")

    if type == "zeroes sum":
         print(f"Sum of zeroes: {sum_zeroes * -1}")
    elif type == "zeroes product":
         print(f"Product of zeroes: {product_zeroes}")
    elif type == "zeroes":
         print(f"Zeroes: {zero1}, {zero2}")
    elif type == "factors":
         print(f"Factored: {factor1}{factor2}, {factor2}{factor1}")
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
        if type == "zeroes sum":
            sum_zeroes_invert = sum_zeroes * -1
            return content_uri, img_width, img_height, str(int(sum_zeroes_invert)), polynomial
        elif type == "zeroes product":
            return content_uri, img_width, img_height, str(int(product_zeroes)), polynomial
        elif type == "zeroes":
            zeroes_str = [
                f"{zero1} and {zero2}",
                f"{zero2} and {zero1}",
                f"{zero1}, {zero2}",
                f"{zero2}, {zero1}",
                f"{zero1} {zero2}",
                f"{zero2} {zero1}"
            ]
            return content_uri, img_width, img_height, zeroes_str, polynomial
        elif type == "factors":
            factored_str = [
                f"{factor1}{factor2}",
                f"{factor2}{factor1}",
                f"{factor1_mod}{factor2_mod}",
                f"{factor2_mod}{factor1_mod}"
            ]
            return content_uri, img_width, img_height, factored_str, polynomial
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
        "People": "🙋‍♂️🙋‍♀️",
        "Celebrities": "💃🕺",
        "Anatomy": "🧠🫀",
        "Characters": "🧙‍♂️🧛",
        "Music": "🎶🎸",
        "Art & Literature": "🎨📚",
        "Chemistry": "🧪⚗️",
        "Business": "💼📈",
        "Rebus Puzzle": "🤔🖼️",
        "Cars & Other Vehicles": "🚗🛩️",
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
        "Video Games": "💻🎮",
        "Flags": "🏳️🏴",
        "Miscellaneous": "🔀✨",
        "Biology": "🧬🦠",
        "Earth Science": "🌎🔬",
        "Superheroes": "🦸‍♀️🦸",
        "Television": "📺🎥",
        "Pop Culture": "🎉🌟",
        "History": "📜🕰️",
        "Movies": "🎬🍿",
        "TV Shows": "📺🎥",
        "Religion & Mythology": "🛐🐉",
        "Sports & Leisure": "⚽🌴",
        "Politics & History": "🏛️📜",
        "Sports": "🏈⚾",
        "Sports & Games": "⚽🎮",
        "World Culture": "🎭🗿",
        "General Knowledge": "📚💡",
        "Anything": "🌐🔀",
        "Crossword": "📰✏️",
        "English": "🇬🇧🗣️",
        "Philippines": "🇵🇭🏝️",
        "Renaissance": "🏰🎨",
        "Fashion Japan": "👘🇯🇵",
        "Spring": "🌸🌱",
        "Game Of Thrones": "🐉⚔️",
        "Earth Day": "🌍🌱",
        "Human Body": "🫀🦴",
        "Film": "🎥🎞️",
        "South Park": "📺🤣",
        "Beer": "🍺🍻",
        "Animation": "🎨📽️",
        "Casino": "🎰♠️",
        "1970s": "🕺📻",
        "Baking": "🧁🥣",
        "Australia": "🇦🇺🦘",
        "Shopping": "🛍️🛒",
        "Books & Publications": "📚📰",
        "Chicago": "🌆🍕",
        "World War 1": "🌍⚔️",
        "For Seniors": "👴👵",
        "Ice Cream": "🍦🍨",
        "Military History": "⚔️🎖️",
        "Places & Travel": "🌍✈️",
        "Military History": "⚔️🎖️",
        "British History": "🏰🇬🇧",
        "Wimbledon": "🎾🏆",
        "1960s": "✌️🎶",
        "Celebrity Weddings": "💒💍",
        "Movie Villains": "😈🎥",
        "Leap Year": "📅🐸",
        "Back To The Future": "⌛🚗",
        "Olympics": "🏅🏟️",
        "Car Parts": "🚗🔧",
        "August": "☀️📆",
        "Fashion": "👗👠",
        "Italian Cuisine": "🍝🍕",
        "Toy Story": "🤠🧸",
        "The Simpsons": "🟨👨‍👩‍👧‍👦",
        "Taylor Swift": "🎤💖",
        "Fruit Vegetables": "🍏🥕",
        "Avengers": "🛡️⚡",
        "Nintendo": "🕹️🍄",
        "Playstation Games": "🎮⚙️",
        "Games": "🎮🎲",
        "Swedish Cuisine": "🥔🐟",
        "Disney Princess": "👑🏰",
        "Extreme Sports": "🏂🚵",
        "Halloween": "🎃👻",
        "Summer": "☀️🏖️",
        "Home Alone": "🏠🧒",
        "Pokemon": "⚡🐭",
        "Cartoons": "📺🐱",
        "Minecraft": "⛏️🐷",
        "Eminem": "🎤🍬",
        "Marvel": "🦸‍♂️🦹‍♂️",
        "Sherlock Holmes": "🕵️‍♂️🔎",
        "Board Games": "♟️🎲",
        "Architecture": "🏛️🏗️",
        "Art & Architecture": "🎨🏛️",
        "Weather": "☀️🌧️",
        "Albert Einstein": "🧠💡",
        "Serial Killer": "🔪😈",
        "Civil War": "⚔️🛡️",
        "New Year Halloween": "🎉🎃",
        "Horse Racing": "🐎🏁",
        "Breaking Bad": "🧪👨‍🔬",
        "1990s": "📟💾",
        "Premier League": "⚽🏆",
        "Classic Rock": "🎸🎶",
        "Alcohol": "🍺🥃",
        "Outer Space": "🚀🌌",
        "Family Guy": "👨‍👩‍👧‍👦😂",
        "Reality Stars": "🌟📺",
        "Fast Food": "🍔🍟",
        "Comics": "💥🦸",
        "Weird": "🤪🌀",
        "Sci Fi": "👽🚀",
        "Graphic Design": "💻🎨",
        "Decades": "⏳📅",
        "Animals": "🐾🐼",
        "Boxing": "🥊💥",
        "Oldies Music": "🎶🕰️",
        "Fourth Of July": "🇺🇸🎆",
        "Shrek": "🟢👑",
        "September": "🍂📅",
        "Quran": "📖🕋",
        "Queen": "👑👸",
        "Disney": "🏰🐭",
        "Indian Cuisine": "🍛🥘",
        "Book": "📖📚",
        "Modern History": "📜🌐",
        "Festivals": "🎉🎆",
        "Winter Olympics": "🏅🏂",
        "Horse": "🐎🌿",
        "Quentin Tarantino": "🎬🩸",
        "Inventions": "💡⚙️",
        "Baby Shower": "👶🎉",
        "New Girl": "🏠👱‍♀️",
        "Kings Queens": "🤴👸",
        "Sexuality": "🏳️‍🌈💖",
        "Canada": "🇨🇦🍁",
        "Agriculture": "🌱🚜",
        "1940s": "💣📻",
        "Questions For Kids": "❓👧",
        "Travel": "✈️🌍",
        "Rainforest": "🌧️🌳",
        "Presidents Day": "🇺🇸🏛️",
        "Star Wars": "🌌⚔️",
        "Power": "⚡💪",
        "Supernatural": "👻🌙",
        "X Files": "👽🕵️",
        "Technology": "💻🤖",
        "Google": "🔍🌐",
        "The Beatles": "🎸🇬🇧",
        "Car": "🚗🛣️",
        "India": "🇮🇳🪔",
        "Greek Mythology": "🏛️⚡",
        "World Cup": "🌍🏆",
        "Scandal": "📰😱",
        "Easter": "🐰🥚",
        "Brands": "🏷️💼",
        "Poetry": "📜🖋️",
        "Ncis": "🕵️‍♂️⚓",
        "Shakespeare": "📝🎭",
        "Country Music": "🤠🎶",
        "Europe": "🇪🇺🏰",
        "Musicals": "🎶🎭",
        "Entertainment": "🎉🎭",
        "Coffee": "☕🍪",
        "Apple": "🍎💻",
        "Airlines Airports": "✈️🛫",
        "Sea Life And Oceans": "🌊🐠",
        "Science Fiction": "👽🤖",
        "Soundtracks": "🎶🎞️",
        "Canada Day": "🇨🇦🎉",
        "Survivor": "🌴🏆",
        "War History": "💣📜",
        "Labor Day": "🛠️🇺🇸",
        "Mlb Baseball": "⚾🏟️",
        "Bar": "🍸🪑",
        "Valentines Day": "❤️💌",
        "One Piece": "🏴‍☠️🍖",
        "Mental Health": "🧠💚",
        "Friends": "👫💞",
        "Russian Cuisine": "🥟🍲",
        "Hannukkah": "🕎✨",
        "Hispanic Heritage Month": "🪗🎉",
        "The Office": "🏢😂",
        "China": "🇨🇳🐉",
        "Silly": "🤪🎉",
        "Stranger Things": "🚲🔦",
        "Pop Music": "🎤🎶",
        "Elvis": "🕺🎤",
        "Lord Of The Rings": "💍🔥",
        "Tennis": "🎾🏅",
        "Plants Trees": "🌱🌳",
        "Us Presidents": "🇺🇸👔",
        "Sharks": "🦈🌊",
        "Childrens Literature": "🧒📚",
        "Africa": "🌍🦁",
        "Comedy": "😂🎭",
        "Medical": "🩺💊",
        "Sesame Street": "🐤📺",
        "Easy": "😌✅",
        "Soap Opera": "📺💔",
        "Romance": "❤️🌹",
        "Pixar": "🤠🦖",
        "Wwe": "🤼‍♂️💥",
        "Poker": "♠️💰",
        "Beach": "🏖️🌅",
        "Holiday": "🎉🌴",
        "Teens": "🧑‍🎓🤸",
        "Twilight": "🧛‍♂️🌆",
        "Parks And Recreation": "🏞️😆",
        "Pregnancy": "🤰👶",
        "Oktoberfest": "🍺🇩🇪",
        "Roald Dahl": "📚🍫",
        "Wonders Of The World": "🏰🌍",
        "Canadian Cuisine": "🥞🍁",
        "Current Royals": "🤴👸",
        "Blockbusters": "💥🍿",
        "Cooking": "🍳🧑‍🍳",
        "Dinosaurs": "🦕🦖",
        "60s70s80s90s": "🎶📻",
        "4th grade  questions": "4️⃣❓",
        "Modern Family": "👨‍👩‍👧‍👦🏠",
        "Star Trek": "🖖🚀",
        "Winter": "❄️☃️",
        "Politics News": "🗞️⚖️",
        "Early Art": "🖼️🏺",
        "Stephen King": "🕯️😱",
        "Classical Music": "🎼🎻",
        "British Music": "🇬🇧🎶",
        "Seinfeld": "🏙️🤣",
        "Film Timings": "🎬⏰",
        "Candy": "🍬🍭",
        "European Championships": "🇪🇺🏆",
        "Cycling": "🚴‍♂️🚴‍♀️",
        "Asia": "🌏🏯",
        "Bob Marley": "🎶🇯🇲",
        "American Cuisine": "🍔🥧",
        "Us States": "🇺🇸📍",
        "Titanic": "🚢💔",
        "War": "⚔️💣",
        "Education": "🏫📚",
        "Fall": "🍂🍁",
        "Novels": "📚✒️",
        "5th grade": "5️⃣❓",
        "Nickelodeon": "📺🧒",
        "Authors": "🖋️📚",
        "2010s": "📱💻",
        "Horror Movie": "🔪😱",
        "Christmas  For Kids": "🎄🧸",
        "Riddle": "❓🧩",
        "Christmas": "🎄🎅",
        "Sitcom": "😂📺",
        "Nhl Hockey": "🏒🥅",
        "Solar System": "☀️🪐",
        "Michael Jackson": "🕺🪄",
        "Hobbies": "⚽🎨",
        "United States": "🇺🇸🗽",
        "Golf": "⛳🏌️‍♂️",
        "Continents Countries": "🌍🌎",
        "Nutrition Month": "🥦🍏",
        "Transport": "🚗🚇",
        "Hard": "💪🔨",
        "Beneath The Sea": "🌊🐙",
        "Bollywood": "💃🎥",
        "Thanksgiving": "🦃🍁",
        "Super Bowl": "🏈🏆",
        "New Year": "🎆🍾",
        "1950s": "🎩🎶",
        "Mammals": "🐒🐘",
        "Nba Teams": "🏀🏅",
        "Crime": "🚓🕵️",
        "Oscars Awards": "🏆🎞️",
        "St Patrick S Day": "🍀🇮🇪",
        "Medicine": "💊🩺",
        "Science & Medicine": "🔬🧬",
        "Famous Authors": "🖋️📚",
        "Nfl": "🏈🏟️",
        "Funny": "🤣😜",
        "New York": "🗽🌃",
        "Fashion Design": "👗✂️",
        "Australian History": "🇦🇺📜",
        "Internet": "🌐💻",
        "Brands Worldwide": "🌐🏷️",
        "Gen Z": "📱😎",
        "Capital Cities": "🌆🗺️",
        "Mario": "👨‍🔧🍄",
        "2000s": "💻📱",
        "Back To School": "🎒🏫",
        "Philosophers": "🤔📜",
        "Spelling": "🔤📝",
        "Bible": "📖✝️",
        "Nascar": "🏁🏎️",
        "Current Affairs": "📰🌍",
        "London": "🇬🇧🎡",
        "Monday": "📅😴",
        "Us Tv": "🇺🇸📺",
        "Electricity": "⚡💡",
        "Classic Tv": "📺🕰️",
        "North America": "🌎🏒",
        "Top Gun": "✈️🕶️",
        "Harry Potter": "⚡🧙‍♂️",
        "Memorial Day": "🇺🇸🪖",
        "Actors Actresses": "🎭🎬",
        "Actor / Actress": "🎭🎬",
        "Royal Family": "👑👨‍👩‍👧‍👦",
        "Uk Football": "🇬🇧⚽",
        "Batman": "🦇🦸‍♂️",
        "Black History": "✊🏿📜",
        "Encanto": "🏠💃",
        "Middle School": "🏫👩‍🎓",
        "Reality Tv": "📺😜",
        "Jurassic Park": "🦕🎢",
        "Classic Movies": "🎞️🏆",
        "Rock Roll": "🎸🤘",
        "1980s": "💾📼",
        "Design": "🎨🖌️",
        "James Bond": "🤵🔫",
        "Monopoly": "💰🏠",
        "Sunset": "🌇🌅",
        "Hip Hop Rap": "🎤🔥",
        "Dogs": "🐶🦴",
        "Ancient Medieval History": "🏰⚔️",
        "Musicals Theatre": "🎭🎵",
        "Non Fiction": "📚📖",
        "Texas": "🤠🌵",
        "Hamilton": "🎩🎼",
        "World War 2": "💣🌍",
        "Ufc Martial Arts": "🥋🥊",
        "Humanities": "📖🎨",
        "Brain-Teasers": "🧠❓",
        "Rated": "⭐🔞",
        "Newest": "🆕✨",
        "Art": "🎨🖌️",
        "Drinks": "🍹🥂",
        "Religion": "🛐🙏",
        "Mathematics & Geometry": "➕📐",
        "Technology & Video Games": "💻🕹️",
        "Tourism And World Cultures": "✈️🗿",
        "Superhero": "🦸🦹",
        "Nature": "🌱🌳",
        "Worldwide History": "🌐📜",
        "Uk History": "🇬🇧📜",
        "Ocean": "🌊🐠",
        "Food & Drink": "🍽️🍸",
        "Space": "🚀🪐",
        "Science": "🔬🧪",
        "Tv": "📺🍿",
        "TV": "📺🍿",
        "People & Places": "👨‍👩‍👧‍👦🏙️",
        "Toys": "🧸🪀",
        "Food": "🍔🥗",
        "Maths": "➕🔢",
        "Elements": "🔥💧",
        "History & Holidays": "📜🎉",
        "Art And Literature": "🎨📚",
        "For-Kids": "👧🧩",
        "World": "🌍🌏",
        "Video-Games": "🎮👾",
        "Science-Technology": "🔬🤖",
        "Literature": "📚✒️",
        "Religion-Faith": "🛐📿",
        "Mathematics: Algebra": "🤓➕",
        "Mathematics: Trigonometry": "📐📊",
        "Mathematics: Mean": "➗📈",
        "Mathematics: Median": "🔢📊",
        "Mathematics: Polynomials": "📉✖️",
        "Mathematics: Bases": "2️⃣🔟",
        "Mathematics: Derivatives": "📉♾️"  
    }

    # Check if the question URL is "jeopardy"
    if trivia_url.lower() == "jeopardy":
        return f"{trivia_category} 🟦🇯"
    # Otherwise, get the emojis based on the lookup table, defaulting to the category itself if not found
    emojis = emoji_lookup.get(trivia_category, "❓❔")
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
    
    #Get a random new question from the database
    new_question = get_random_trivia_question()
    
    #Append the new question to the end of the list to maintain order
    questions.append(new_question)


def get_random_trivia_question():
    global categories_to_exclude
    """Fetch a random question from the trivia_questions collection."""
    try:
        trivia_collection = db["trivia_questions"]
        recent_general_ids = get_recent_question_ids_from_mongo("general")

        if image_questions == False:
            # Define a list of substrings to exclude in URLs
            excluded_url_substring = "http"
            pipeline = [
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
                {"$sample": {"size": 1}}  # Sample from the resulting limited set
            ]
            
        else:
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
                selected_question["answers"],
                "trivia_questions",
                selected_question["_id"]
            )
            
            return final_selected_question
        else:
            print("No available questions found.")
            return None
            
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Error selecting trivia and crossword questions: {e}")
        return None  # Return an empty list in case of failure



def start_trivia():
    global target_room_id, bot_user_id, bearer_token, question_time, questions_per_round, time_between_questions, filler_words
    global scoreboard, current_longest_round_streak, current_longest_answer_streak
    global headers, params, filter_json, since_token, round_count, selected_questions, magic_number
    global previous_question, current_question
    global db
    global question_responders, round_responders

    # You can now use and reset them in the function

    signal.signal(signal.SIGTERM, handle_sigterm)
    print(f"Script is running with PID: {os.getpid()}")
    
    try:
        reddit_login()
        login_to_chat()
        last_login_time = time.time()  # Store the current time when the script starts

        
        db =  connect_to_mongodb()
        load_parameters()
        load_global_variables()
        load_streak_data()
        load_previous_question()
        initialize_sync()  

        fetch_donations()
        
        round_winner = None
        selected_questions = select_trivia_questions(questions_per_round)  #Pick the initial question set
        
        while True:  # Endless loop       
            # Check if it's been more than an hour since the last login
            current_time = time.time()
            
            if current_time - last_login_time >= 3600:  # 3600 seconds = 1 hour
                print("Re-logging into Reddit and chat as one hour has passed...")
                reddit_login()
                login_to_chat()
                last_login_time = current_time  # Reset the login time
                load_global_variables()

            load_parameters()

            # Reset the scoreboard and fastest answers at the start of each round
            scoreboard.clear()
            fastest_answers_count.clear()
            
            # Reset round data for the next round
            round_responders.clear()  # Reset round responders
            round_data["questions"] = []

            if random.random() < 0:  # random.random() generates a float between 0 and 1
                magic_number = random_number = random.randint(1000, 9999)
                print(f"Magic number is {magic_number}")
                send_magic_image(magic_number)
            elif image_questions == True:
                selected_gif_url = select_intro_image_url()         
                image_mxc, image_width, image_height = download_image_from_url(selected_gif_url)
                send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)
                #time.sleep(2)

            start_message = f"\n⏩ Starting a round of {questions_per_round} questions ⏩\n"
            start_message += f"\n🚩 Flag errors during response time"
            start_message += f"\n↔️ #curr, #prev to tag question\n"
            send_message(target_room_id, start_message)
            time.sleep(3)
            
            start_message = f"\n✨🧪 Check out the new modes!\n"
            start_message += f"\n🎏🎉 Flag Fest"
            start_message += f"\n🤓📚 Word Nerd"
            start_message += f"\n🟢🎩 The Riddler\n"
            
            send_message(target_room_id, start_message)
            time.sleep(3)

            start_message = "\n🏁 Get ready 🏁\n"
            send_message(target_room_id, start_message)
            round_start_messages()
            time.sleep(5)
                
            # Randomly select n questions
            print_selected_questions(selected_questions)
            
            question_number = 1
            while question_number <= questions_per_round:
                question_responders.clear()  # Reset question responders for the new question
                
                if god_mode and round_winner:
                    selected_question = selected_questions[get_player_selected_question(selected_questions, round_winner) - 1]
                    
                else:
                    selected_question = selected_questions[0]

                trivia_category, trivia_question, trivia_url, trivia_answer_list, trivia_db, trivia_id = selected_question

                current_question = {
                    "trivia_category": trivia_category,
                    "trivia_question": trivia_question,
                    "trivia_url": trivia_url,
                    "trivia_answer_list": trivia_answer_list,
                    "trivia_db": trivia_db,
                    "trivia_id": trivia_id
                }

                initialize_sync()
                question_ask_time, new_question, new_solution = ask_question(trivia_category, trivia_question, trivia_url, trivia_answer_list, question_number)
                collected_responses = collect_responses(question_time, question_number, question_time)
                
                #send_message(target_room_id, f"\n🛑 TIME 🛑\n")
                
                solution_list = []

                if new_solution is None:
                    # Use the original trivia answer list if no new solution is provided
                    solution_list = trivia_answer_list
                elif isinstance(new_solution, list):
                    # If new_solution is already a list, use it as-is
                    solution_list = new_solution
                else:
                    # If new_solution is a single value, wrap it in a list
                    solution_list = [new_solution]        
                    
                check_correct_responses_delete(question_ask_time, solution_list, question_number, collected_responses, trivia_category, trivia_url)
                
                if not yolo_mode or question_number == questions_per_round:
                    show_standings()

                #Refill the question slot with a new random question from trivia_questions
                refill_question_slot(selected_questions, selected_question)

                time.sleep(time_between_questions)  # Small delay before the next question
                
                question_number = question_number + 1
                previous_question = {
                    "trivia_category": trivia_category,
                    "trivia_question": trivia_question,
                    "trivia_url": trivia_url,
                    "trivia_answer_list": trivia_answer_list,
                    "trivia_db": trivia_db,
                    "trivia_id": trivia_id
                }

                save_data_to_mongo("previous_question", "previous_question", previous_question)
                
            #Determine the round winner
            round_winner, winner_points = determine_round_winner()

            #Update round streaks
            fetch_donations()
            update_round_streaks(round_winner)
            # Increment the round count

            round_count += 1
        
            time.sleep(10)
            process_round_options(round_winner, winner_points)
            
            if round_count % 5 == 0:
                send_message(target_room_id, f"\n🧘‍♂️ A short breather. Relax, stretch, meditate.\n🎨 Live Trivia is a pure hobby effort.\n💡 Help make it better: https://forms.gle/iWvmN24pfGEGSy7n7\n")
                time.sleep(30)
                selected_questions = select_trivia_questions(questions_per_round)  #Pick the next question set
                round_preview(selected_questions)
                time.sleep(10)
            else:
                message = f"\n☕️ https://buymeacoffee.com/livetrivia\n💚 Use your Reddit name to unlock in-game perks.\n"
                message += f"\n👕 https://livetriviamerch.com\n🛒 Score Live Trivia merch featuring Okra.\n"
                send_message(target_room_id, message)
                selected_questions = select_trivia_questions(questions_per_round)  #Pick the next question set
                time.sleep(10)
                round_preview(selected_questions)
                time.sleep(10)  # Adjust this time to whatever delay you need between rounds
            
            if len(scoreboard) > 40000:
                ask_survey_question()
                
            time.sleep(5)

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Error occurred: {e}")
        traceback.print_exc()  # Print the stack trace of the error
        print("Restarting the trivia bot in 10 seconds...")
        time.sleep(10)  


try:
    sentry_sdk.capture_message("Sentry initiatlized...", level="info")
    start_trivia()
    
except Exception as e:
    sentry_sdk.capture_exception(e)
    print(f"Unhandled exception: {e}. Restarting in 5 seconds...")
    traceback.print_exc()  # Print the stack trace for debugging
    time.sleep(5)
    start_trivia()
