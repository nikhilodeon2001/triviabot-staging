
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
import io            
import hashlib 
from PIL import Image, ImageDraw, ImageFont 
import openai





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
message_headers = []

params = []
message_params = []
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
target_room_id = os.getenv("target_room_id")
question_time = int(os.getenv("question_time"))
questions_per_round = int(os.getenv("questions_per_round"))
time_between_rounds = int(os.getenv("time_between_rounds"))
time_between_questions = int(os.getenv("time_between_questions"))
questions_module = os.getenv("questions_module", "trivia_questions")
max_retries = int(os.getenv("max_retries"))
delay_between_retries = int(os.getenv("delay_between_retries"))
hash_limit = 2000 #DEDUP
first_place_bonus = 0


# Define the base API URL for Matrix
matrix_base_url = "https://matrix.redditspace.com/_matrix/client/v3"
upload_url = "https://matrix.redditspace.com/_matrix/media/v3/upload"
sync_url = f"{matrix_base_url}/sync"
messages_url = f"https://matrix.redditspace.com/_matrix/client/v3/rooms/{target_room_id}/messages"
direction = "b"  # 'b' for reverse-chronological order, 'f' for chronological
limit = 1000  # Max number of events to return (you can change this value)


def generate_scrambled_image(scrambled_text):
    """
    Generate an image with scrambled words using PIL (Pillow).
    """
    # Define the font path and size
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")
    font_size = 48
    
    # Create a blank image
    img_width, img_height = 400, 150
    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return

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
    content_uri = upload_image_to_matrix(image_buffer.read())
    if content_uri:
        return content_uri, img_width, img_height
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
    """
    Generate a summary of the trivia round using OpenAI's API.
    """
    # Construct the prompt with clear instructions
    prompt = (
        "You are an arrogan, sarcastic, and jaded trivia game host. Insult the trivia winner's username and let them know they're not as good as you. "
        "Vary your language, and make it sarcastic, ironic, and antagonistic. Here is a detailed summary of the trivia round with explicit mappings of user responses:\n"
        "Questions asked:\n"
    )

    # Add questions, their correct answers, users' responses, and scoreboard status after each question
    for question_data in round_data["questions"]:
        question_number = question_data["question_number"]
        question_text = question_data["question_text"]
        correct_answers = question_data["correct_answers"]
        
        prompt += f"Question {question_number}: {question_text}\n"
        prompt += f"Correct Answers: {', '.join(correct_answers)}\n"
        
        # Add users and their responses for each question
        prompt += "Users and their responses:\n"
        if question_data["user_responses"]:
            for response in question_data["user_responses"]:
                username = response["username"]
                user_response = response["response"]
                is_correct = "Correct" if any(fuzzy_match(user_response, answer) for answer in correct_answers) else "Incorrect"
                prompt += f"Username: {username} | Response: '{user_response}' | Result: {is_correct}\n"
        else:
            prompt += "No responses recorded for this question.\n"
        
        # Add scoreboard status after the question
        prompt += f"\nScoreboard after Question {question_number}:\n"
        if "scoreboard_after_question" in question_data:
            for user, score in question_data["scoreboard_after_question"].items():
                prompt += f"{user}: {score}\n"
        else:
            prompt += "No responses recorded.\n"
        
        prompt += "\n"

    # Add specific instructions for generating the ribbons
    prompt += (
        f"\nThe winner of the trivia round is {winner}. "
        "Insult their username and let them know that you're the best and you would crush them at trivia. Highlight any stupid or idiotic responses they gave during the round."
        "Create no more than 5 sentences in total. Be creative, sarcastic, and antagonistic. Use emojis in your response to make it engaging."
    )


    # Use OpenAI's API to generate the summary
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sarcastic trivia game host who roasts the winning trivia player using specifics from the trivia round."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.8,
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
                return image_data, image_width, image_height  # Successfully downloaded the image, return the binary data
                
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
    

def upload_image_to_matrix(image_data): #IMAGE CODE
    global max_retries, delay_between_retries

    """Upload an image to Matrix with retry logic."""
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
                print(bearer_token)
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

     message_headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/json"
    }

    message_params = {
    "dir": direction,
    "limit": limit
    }

    #print(f"target room id is: {target_room_id}")
    #print(f"bot user id is: {bot_user_id}")
    #print(f"bearer token is: {bearer_token}")
    #print(f"question time is: {question_time}")
    #print(f"questions per round is: {questions_per_round}")
    #print(f"time between rounds is: {time_between_rounds}")
    #print(f"time between questions is: {time_between_questions}")
    #print(f"questions module: {questions_module}")
    #print(f"max retries is: {max_retries}")
    #print(f"delay_between_retries is: {delay_between_retries}")
    
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


def send_image(room_id, image_mxc, image_width, image_height, image_size): #IMAGE CODE
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

def ask_question(trivia_question, trivia_url, trivia_answer_list, question_number):
    """Ask the trivia question."""
    # Define the numbered block emojis for questions 1 to 10
    numbered_blocks = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    number_block = numbered_blocks[question_number - 1]  # Get the corresponding numbered block
    new_solution = None #POLY

    # Store the question being asked in round_data
    round_data["questions"].append({
        "question_number": question_number,
        "question_text": trivia_question,
        "correct_answers": trivia_answer_list,  # Store the possible correct answers
        "user_responses": []  # Initialize a list to collect user responses
    })
    
    if is_valid_url(trivia_url): 
        if "polynomial" in trivia_url:
            image_mxc, image_width, image_height, new_solution = generate_and_render_polynomial_image() #POLY
            image_size = 100
        
        elif "scramble" in trivia_url:
            image_mxc, image_width, image_height = generate_scrambled_image(scramble_text(trivia_answer_list[0]))
            image_size = 100
    
        else:
            image_data, image_width, image_height = download_image_from_url(trivia_url) #FILE TYPE
            image_mxc = upload_image_to_matrix(image_data)
            image_size = len(image_data)
        message_body = f"\n{number_block}📷 Image Question 📷{number_block}\n{trivia_question}"
        message_response = send_message(target_room_id, message_body)

        if message_response is None:
            print("Error: Failed to send the message.")
            return None, None  # Exit the function if sending the message failed
        
        response = send_image(target_room_id, image_mxc, image_width, image_height, image_size)

        if response is None:                      
            print("Error: Failed to send image.")
            return None, None  # Exit the function if sending the image failed

    else:
        # Send the question to the chat
        message_body = f"\n{number_block} Question {number_block}\n{trivia_question}"
        initialize_sync()
        response = send_message(target_room_id, message_body)

        if response is None:
            print("Error: Failed to send the message.")
            return None, None  # Exit the function if sending the message failed

    # Extracting the 'Date' field
    response_time = response.headers.get('Date')

    # Convert the 'Date' field into a datetime object
    date_format = '%a, %d %b %Y %H:%M:%S %Z'
    response_time = datetime.datetime.strptime(response_time, date_format)

    # Set the timezone explicitly to UTC
    response_time = response_time.replace(tzinfo=pytz.UTC)

    # Convert the datetime object to epoch time in seconds and microseconds (float)
    response_time = response_time.timestamp()

    return response_time, new_solution

def initialize_sync():
    """Perform an initial sync to update the since_token without processing messages."""
    global since_token, filter_json, headers, params, max_retries, delay_between_retries
    #sync_url = f"{matrix_base_url}/sync"
        
    for attempt in range(max_retries):
        try:
            if since_token:
                params["since"] = since_token
                
            # Perform the sync to get the initial token
            response = requests.get(sync_url, headers=headers, params=params)
            
            if response.status_code == 200:
                sync_data = response.json()
                since_token = sync_data.get("next_batch")  # Update the since token without processing messages
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
    return points

# List of filler words to remove
filler_words = {'a', 'an', 'the', 'of', 'and', 'to', 'in', 'on', 'at', 'with', 'for', 'by'}

def remove_filler_words(input_str):
    """Remove filler words from the input string."""
    words = input_str.split()
    filtered_words = [word for word in words if word not in filler_words]
    return ' '.join(filtered_words)

def normalize_text(input_str):
    """Normalize text by removing diacritics, punctuation, whitespace, and filler words, and converting to lowercase."""
    # Strip leading/trailing whitespace
    text = input_str.strip()
    # Remove diacritics
    text = remove_diacritics(text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove filler words using the subfunction
    text = remove_filler_words(text)
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

def fuzzy_match(user_answer, correct_answer, threshold=0.90): #POLY
    # Check if the user's answer is 1, 2, or 3 characters long
    # Check if the correct answer is numeric
    # Check if either pre-normalized answer is empty
    
    if not user_answer or not correct_answer:           #POLY
         return user_answer == correct_answer  # Only accept exact match if either are empty        #POLY
    
    no_spaces_user = user_answer.replace(" ", "")       #POLY
    no_spaces_correct = correct_answer.replace(" ", "") #POLY

    if no_spaces_user == no_spaces_correct:     #POLY
        return True
    
    if is_number(correct_answer):
        return user_answer == correct_answer  # Only accept exact match if the correct answer is a number

    if len(user_answer) < 3 or len(correct_answer) < 3:
        return user_answer.lower() == correct_answer.lower()  # Only accept an exact match for short answers

    # Normalize both user and correct answers
    user_answer = normalize_text(user_answer)
    correct_answer = normalize_text(correct_answer)
    
    # Check if either post-normalized answer is empty
    if not user_answer or not correct_answer:
         return user_answer == correct_answer  # Only accept exact match if either are now empty
    
    no_spaces_user = user_answer.replace(" ", "")       #POLY
    no_spaces_correct = correct_answer.replace(" ", "") #POLY

    if no_spaces_user == no_spaces_correct:     #POLY
        return True

    # New Step: First 5 characters match
    if user_answer[:5] == correct_answer[:5]:
        return True
    
    # Remove filler words and split correct answer
    correct_answer_words = correct_answer.split()
    
    # Ensure correct_answer_words is not empty
    if correct_answer_words and user_answer == correct_answer_words[0] and len(correct_answer_words[0]) > 3:
        return True
    
    # Step 1: Exact match or Partial match
    if correct_answer in user_answer:
        return True
    
    # Step 2: Levenshtein similarity
    if levenshtein_similarity(user_answer, correct_answer) >= threshold:
        return True

    # Step 3: Jaccard similarity (Character level)
    if jaccard_similarity(user_answer, correct_answer) >= threshold:
        return True

    # Step 4: Token-based matching
    if token_based_matching(user_answer, correct_answer) >= threshold:
        return True

    return False  # No match found
                          
                
def check_correct_responses(question_ask_time, trivia_answer_list, question_number):
    """Check and respond to users who answered the trivia question correctly."""
    global since_token, params, filter_json, headers, max_retries, delay_between_retries, current_longest_answer_streak
    #sync_url = f"{matrix_base_url}/sync"
    
    # Define the first item in the list as trivia_answer
    trivia_answer = trivia_answer_list[0]  # The first item is the main answer
    correct_responses = []  # To store users who answered correctly
    has_responses = False  # Track if there are any responses
   
    for attempt in range(max_retries):
        try:
            fastest_correct_user = None
            fastest_response_time = None
            
            if since_token:
                message_params["from"] = since_token

            response = requests.get(messages_url, headers=message_headers, params=message_params)
            if response.status_code == 200:
                response_data = response.json()
                #print("Messages:", data.get("chunk", []))
                #print("Start token:", data.get("start"))
                #print("End token:", data.get("end"))
                since_token = response_data.get("end")
                messages = response_data.get("chunk", [])
                for message in messages:
                    content = message.get("content", {})
                    sender = message.get("sender", "Unknown sender")
                    body = content.get("body", "[No message body]")
            
                    # Print who said the message and the message content
                    print(f"{sender} said: {body}")
            else:
                print(f"Failed to fetch messages. Status code: {response.status_code}")
                print(response.text)
            
            
            
            
            
            '''
            if response.status_code == 200:
                sync_data = response.json()
                since_token = sync_data.get("next_batch")  # Update the since token
                
                # Process messages from the room
                responses.clear()
                for room_id, room_data in sync_data.get("rooms", {}).get("join", {}).items():
                    if room_id == target_room_id:  # Only process messages from the target room
                        for event in room_data.get("timeline", {}).get("events", []):
                            sender = event["sender"]
                            if sender == bot_user_id:
                                continue
                            display_name = get_display_name(event.get("content", {}).get("displayname", sender))  # Get the display name from content
                            
                            # Check if the user has already answered correctly, ignore if they have
                            if any(resp[0] == display_name for resp in correct_responses):
                                continue  # Ignore this response since the user has already answered correctly

                            # Log user submission (MongoDB operation)
                            log_user_submission(display_name)
                            
                            message_content = event.get("content", {}).get("body", "")
                            normalized_message_content = normalize_text(message_content)

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
                            if any(fuzzy_match(message_content, answer) for answer in trivia_answer_list):
                                
                                timestamp = event.get("origin_server_ts", None) / 1000  # Extract the timestamp
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
                message += f"\n✅ Answer ✅\n{trivia_answer}\n"
            
                # Notify the chat
                if correct_responses:    
                    correct_responses_length = len(correct_responses)
                    
                    # Loop through the responses and append to the message
                    for display_name, points, response_time, message_content in correct_responses:
                        time_diff = response_time - fastest_response_time
                        if time_diff == 0:
                            message += f"\n⚡ {display_name}: {points}"
                            if current_longest_answer_streak["streak"] > 1:
                                message += f"  🔥{current_longest_answer_streak['streak']}"
                            #message += f"\n💬 Answered: {message_content}"
                            #if correct_responses_length > 1:
                            #    message += f"\n\n👥 The Rest"
                        else:
                            message += f"\n🎉 {display_name}: {points} (+{round(time_diff, 1)}s)"
                elif has_responses:  # Only if there were responses but none correct
                    potential_messages = [
                        "\n🥴 We've got a bunch of geniuses here...\n",
                        "\n🤔 Did someone hide the thinking cap?\n",
                        "\n😅 Well, that went right...off a cliff!\n",
                        "\n🙃 The silence is deafening...\n",
                        "\n🤷‍♂️ Hello? Anyone here?\n",
                        "\n🧠 The brains have officially left the chat.\n",
                        "\n🤯 That clearly blew some minds...or all of them.\n",
                        "\n🦗 *Crickets*\n",
                        "\n🧐 Must’ve been a tough one...\n",
                        "\n🤡 At least we’re all equally clueless!\n",
                        "\n💤 ...that question was the nap break, right?\n",
                        "\n🙈 Looks like nobody saw that one coming!\n",
                        "\n😬 Well, yikes. That’s awkward...\n",
                        "\n😵‍💫 I think we all just forgot how to trivia.\n",
                        "\n💡 Lightbulb moment? More like a power outage!\n",
                        "\n🎯 Missed by a mile! Anyone aiming?\n",
                        "\n🚶‍♂️ Um wow. I'll just see myself out.\n",
                        "\n🤪 What a brain twister...yikes!\n",
                        "\n🎱 Outlook not so good...for all of us.\n"
                    ]
                    message += random.choice(potential_messages)
            
                # Send the entire message at once
                if message:
                    send_message(target_room_id, message)

                flush_submission_queue()
                
                return None
            else:
                print(f"Failed to fetch messages. Status code: {response.status_code}")
                return None
         
        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(delay_between_retries)
            else:
                print(f"Max retries reached. Failed to fetch messages.")
                return None  
                    
    return None  
'''

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
        summary = generate_round_summary(round_data, user)
        # Determine the message to send
        if current_longest_round_streak["streak"] > 1:
            message = f"\n🏆 Winner: @{user}...🔥{current_longest_round_streak['streak']} in a row!\n\n\n{summary}\n\n▶️ Live trivia stats available at https://livetriviastats.com\n"
        else:
            message = f"\n🏆 Winner: @{user}!\n\n\n{summary}\n\n▶️ Live trivia stats available at https://livetriviastats.com\n"

        # Send the message
        send_message(target_room_id, message)

    # Perform all MongoDB operations at the end
    for operation in mongo_operations:
        if operation["operation"] == "insert":
            insert_data_to_mongo(operation["collection"], operation["data"])
        elif operation["operation"] == "save":
            save_data_to_mongo(operation["collection"], operation["document_id"], operation["data"])
            

def determine_round_winner():
    """Determine the round winner based on points and response times."""
    if not scoreboard:
        return None

    # Find the user(s) with the most points
    max_points = max(scoreboard.values())
    potential_winners = [user for user, points in scoreboard.items() if points == max_points]

    # If there's a tie, return None (no clear-cut winner)
    if len(potential_winners) > 1:
        send_message(target_room_id, "No clear-cut winner this round due to a tie.")
        return None
    else:
        return potential_winners[0]  # Clear-cut winner


def show_standings():
    """Show the current standings after each question."""
    if scoreboard:
        standings = sorted(scoreboard.items(), key=lambda x: x[1], reverse=True)
        standing_message = "\n📈 Scoreboard"
        
        # Define the medals for the top 3 positions
        medals = ["🥇", "🥈", "🥉"]
        
        for rank, (user, points) in enumerate(standings, start=1):
            formatted_points = f"{points:,}"  # Format points with commas
            fastest_count = fastest_answers_count.get(user, 0)  # Get the user's fastest answer count, default to 0
            
            if rank <= 3:
                # Use medals for the top 3
                if fastest_count > 0:
                    standing_message += f"\n{medals[rank-1]} {user}: {formatted_points}  ⚡{fastest_count}"
                else:
                    standing_message += f"\n{medals[rank-1]} {user}: {formatted_points}"
            #elif rank == len(standings) and rank > 5:
            #    # For the last place person (who is not in the top 3), use the poop emoji
            #    if fastest_count > 0:
            #        standing_message += f"\n💩 {user}: {formatted_points} (*{fastest_count})"
            #    else:
            #        standing_message += f"\n💩 {user}: {formatted_points}"
            else:
                # Use numbers for 4th place and beyond
                if fastest_count > 0:
                    standing_message += f"\n{rank}. {user}: {formatted_points}  ⚡{fastest_count}"
                else:
                    standing_message += f"\n{rank}. {user}: {formatted_points}"
        
        send_message(target_room_id, standing_message)


def load_trivia_questions():
    #Dynamically load the trivia_questions module
    trivia_module = importlib.import_module(questions_module)
    importlib.reload(trivia_module)  # Reload in case the file has changed
    return trivia_module.trivia_questions


def generate_question_hash(question_tuple):  # Use the entire tuple for hashing
    """Generate a SHA-256 hash for the entire question tuple (question text, URL, and answers)."""
    # Convert the tuple to a string and encode it for hashing
    question_tuple_str = json.dumps(question_tuple, sort_keys=True)  # Convert tuple to a string, sort keys to ensure consistency
    return hashlib.sha256(question_tuple_str.encode('utf-8')).hexdigest()

def store_question_hashes_in_mongo(question_hashes):    #DEDUP
    """Store new question hashes in MongoDB and enforce a limit of stored hashes."""
    db = connect_to_mongodb()
    questions_collection = db["asked_questions"]

    # Insert the new question hashes
    for question_hash in question_hashes:
        questions_collection.insert_one({"hash": question_hash, "timestamp": time.time()})

    # Ensure the collection only contains the last 'limit' number of hashes
    total_hashes = questions_collection.count_documents({})
    if total_hashes > hash_limit:
        excess = total_hashes - hash_limit
        
        # Find the oldest excess entries
        oldest_entries = questions_collection.find().sort("timestamp", 1).limit(excess)

        # Delete those entries
        for entry in oldest_entries:
            questions_collection.delete_one({"_id": entry["_id"]})

def get_recent_question_hashes_from_mongo():    #DEDUP
    """Retrieve the most recent question hashes from MongoDB."""
    db = connect_to_mongodb()
    questions_collection = db["asked_questions"]

    # Retrieve the most recent hashes up to the limit
    recent_hashes = questions_collection.find().sort("timestamp", -1).limit(hash_limit)
    return {doc["hash"] for doc in recent_hashes}

def select_trivia_questions(questions_per_round):   #DEDUP
    """
    Select trivia questions, ensuring no recent duplicates based on stored hashes in MongoDB.
    """
    # Load and shuffle trivia questions
    trivia_questions = load_trivia_questions()
    random.shuffle(trivia_questions)

    # Get recent hashes from MongoDB
    recent_hashes = get_recent_question_hashes_from_mongo()

    # Filter out any questions whose hash is in the recent hashes
    selected_questions = []
    selected_question_hashes = []
    
    for question_tuple in trivia_questions:
        question_hash = generate_question_hash(question_tuple)

        # If the question's hash is not in the recent hashes, select it
        if question_hash not in recent_hashes:
            selected_questions.append(question_tuple)
            selected_question_hashes.append(question_hash)
        
        # Stop once we've selected enough questions
        if len(selected_questions) == questions_per_round:
            break

   # If we don't have enough unique questions, select additional questions
    if len(selected_questions) < questions_per_round:
        remaining_questions = [q for q in trivia_questions if generate_question_hash(q[0]) not in selected_question_hashes]
        
        # Check if we still have enough remaining questions
        if len(remaining_questions) >= questions_per_round - len(selected_questions):
            # Select enough remaining questions to fill the quota
            selected_questions += random.sample(remaining_questions, questions_per_round - len(selected_questions))
        else:
            # If not enough remaining questions, select all available and recycle some from recent hashes
            selected_questions += remaining_questions
            
            # Now recycle recently used questions by selecting from `recent_hashes`
            recycled_questions = [q for q in trivia_questions if generate_question_hash(q[0]) in recent_hashes and generate_question_hash(q[0]) not in selected_question_hashes]
            selected_questions += random.sample(recycled_questions, questions_per_round - len(selected_questions))

    # Store the selected question hashes in MongoDB
    store_question_hashes_in_mongo(selected_question_hashes)

    return selected_questions


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
        question = question_data[0]  # The question itself
        answers = question_data[2]  # List of acceptable answers
        # Format and print the question and answers
        print(f"{i}. {question} [{', '.join(answers)}]")


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
                send_message(target_room_id, f"👑  {username} is #1 across all leaderboards. They are our Sovereign. We must bow down.")
                time.sleep(5)
        else:
            # For users not in the Hall of Sovereigns, show all applicable messages
            if top_count == 6:
                send_message(target_room_id, f"👑  {username} is #1 across the board. They are our Sovereign. We must bow down.")
            elif top_count == 5:
                send_message(target_room_id, f"⚔️  {username} is getting too close. Only one leaderboard left. Who will mount the defense?")
            elif top_count == 4:
                send_message(target_room_id, f"🌡️  {username} is heating up. Only two leaderboards left. Stay safe.")
    return None

def generate_and_render_polynomial_image(): #POLY
    # Randomly select coefficients for a, b, and c
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    c = random.randint(1, 9)

    display_a = a
    display_b = b
    display_c = c

    if a == 1:
        display_a = ""
    if b == 1:
        display_b = ""

    # Create the polynomial string in the form ax^2 + bx + c
    polynomial = f"{display_a}x² + {display_b}x + {display_c}"

    # Calculate the derivative: derivative of ax^2 + bx + c is 2ax + b
    derivative = f"{2 * a}x + {b}"

    # Print the polynomial and its derivative
    print(f"Polynomial: {polynomial}")
    print(f"Derivative: {derivative}")

    # Define the font path relative to the current script
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")

    # Create a blank image
    img_width, img_height = 400, 150
    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load the font
    font_size = 48
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return

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
    content_uri = upload_image_to_matrix(image_buffer.read())
    if content_uri:
        return content_uri, img_width, img_height, derivative
    else:
        print("Failed to upload the image to Matrix.")


def start_trivia_round():
# Function to start the trivia round
    global target_room_id, bot_user_id, bearer_token, question_time, questions_per_round, time_between_rounds, time_between_questions, questions_module, filler_words
    global scoreboard, current_longest_round_streak, current_longest_answer_streak
    global headers, params, filter_json, since_token, round_count

    # Track the initial time for hourly re-login
    last_login_time = time.time()  # Store the current time when the script starts
    
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
            
            # Load existing streak data from the file
            load_streak_data()

            """Start a round of n trivia questions."""
            send_message(target_room_id, f"\n⏩ Starting a round of {questions_per_round} questions\n\n🏁 Get ready...\n")
            round_start_messages()
            time.sleep(5)

            # Reset the scoreboard and fastest answers at the start of each round
            scoreboard.clear()
            fastest_answers_count.clear()
            
            # Reset round data for the next round
            round_data["questions"] = []

            # Randomly select n questions
            selected_questions = select_trivia_questions(questions_per_round)  #DEDUP 
            print() 
            print_selected_questions(selected_questions)
            print()
            
            question_number = 1
            for trivia_question, trivia_url, trivia_answer_list in selected_questions:
                # Ask the trivia question and get start times
                question_ask_time, new_solution = ask_question(trivia_question, trivia_url, trivia_answer_list, question_number)
                time.sleep(question_time)  # Wait for n seconds for answers
                #fetch_all_responses(question_ask_time)  # Pass both local and server start times
                send_message(target_room_id, f"\n🛑 TIME 🛑\n")
                
                if new_solution is None:
                    check_correct_responses(question_ask_time, trivia_answer_list, question_number)  # Check the answers     # POLY
                else:
                    check_correct_responses(question_ask_time, [new_solution], question_number)  # Check the answers     # LATENCY
                
                show_standings()  # Show the standings after each question
                #save_trivia_data_answers()
                time.sleep(time_between_questions)  # Small delay before the next question
                question_number = question_number + 1
                
            #Determine the round winner
            round_winner = determine_round_winner()

            #Update round streaks
            update_round_streaks(round_winner)
            # Increment the round count

            round_count += 1
        
            time.sleep(7)
            if round_count % 5 == 0:
                send_message(target_room_id, f"\n🧘‍♂️ 60s breather. Meet your fellow trivians!\n\n🎨 This game has been a pure hobby effort.\n🛟 Help keep it going.\n☕ https://buymeacoffee.com/livetrivia\n👕 https://livetriviamerch.com\n")
                time.sleep(60)
            else:
                send_message(target_room_id, f"💡 Help me improve Live Trivia: https://forms.gle/iWvmN24pfGEGSy7n7\n\n⏳ Next round in ~{time_between_rounds} seconds...\n")
                time.sleep(time_between_rounds)  # Adjust this time to whatever delay you need between rounds

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

    # Call this function at the start of the script to initialize the sync
    #initialize_sync()
    messages_test()
    
    # Start the trivia round
    #start_trivia_round()

except Exception as e:
    #sentry_sdk.capture_exception(e)
    #print(f"Unhandled exception: {e}. Restarting in 5 seconds...")
    #traceback.print_exc()  # Print the stack trace for debugging
    time.sleep(5)
    #reddit_login()
    #login_to_chat()
    #load_global_variables()
    #initialize_sync()
    #start_trivia_round()  # Restart the bot
