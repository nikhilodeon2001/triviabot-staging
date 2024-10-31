
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
from PIL import Image, ImageDraw, ImageFont 
import openai

import main
import subprocess



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
target_room_id = os.getenv("target_room_id")
question_time = int(os.getenv("question_time"))
questions_per_round = int(os.getenv("questions_per_round"))
time_between_rounds = int(os.getenv("time_between_rounds"))
time_between_questions = int(os.getenv("time_between_questions"))
time_between_questions_default = time_between_questions
questions_module = os.getenv("questions_module", "trivia_questions")
max_retries = int(os.getenv("max_retries"))
delay_between_retries = int(os.getenv("delay_between_retries"))
id_limits = {"general": 20000, "mysterybox": 2000, "crossword": 50000, "jeopardy": 100000}
first_place_bonus = 0
magic_time = 10
magic_users = []


num_mysterybox_clues_default = 0
num_mysterybox_clues = num_mysterybox_clues_default
num_crossword_clues_default = 0
num_crossword_clues = num_crossword_clues_default
num_jeopardy_clues_default = 2
num_jeopardy_clues = num_jeopardy_clues_default
ghost_mode_default = False
ghost_mode = ghost_mode_default
god_mode_default = False
god_mode = god_mode_default
god_mode_points = 5000
yolo_mode_default = False
yolo_mode = yolo_mode_default
emoji_mode_default = True
emoji_mode = emoji_mode_default


question_categories = [
    "Mystery Box or Boat", "Famous People", "Anatomy", "Characters", "Music", "Art & Literature", 
    "Chemistry", "Geography", "Mathematics", "Physics", "Science & Nature", "Language", "English Grammar", 
    "Astronomy", "Logos", "The World", "Economics & Government", "Toys & Games", "Food & Drinks", "Geology", 
    "Tech & Video Games", "Flags", "Miscellaneous", "Biology", "Superheroes", "Television", "Pop Culture", 
    "History", "Movies", "Religion & Mythology", "Sports & Leisure", "World Culture", "General Knowledge"
]

categories_to_exclude = []  



def generate_magic_image(input_text):
    global since_token, params, headers, max_retries, delay_between_retries, magic_users

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
        
        message = "🔍❓\n"
        send_message(target_room_id, message)
    
        response = send_image(target_room_id, image_mxc, image_width, image_height, image_size)

        if response.status_code != 200:                      
            print("Error: Failed to send image.")
            print(response)

        sync_url = f"{matrix_base_url}/sync"
    
        collected_responses = []  # Store all responses
        
        processed_events = set()  # Track processed event IDs to avoid duplicates

        initialize_sync()
        start_time = time.time()  # Track when the question starts
        magic_message = "\n"
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
                    redact_message(event_id, target_room_id)
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

                        if sender == bot_user_id:
                            continue

                        if str(input_text).lower() in str(message_content).lower():
                            magic_users.append(sender_display_name)
                            print(f"{sender_display_name} sent {input_text}")  
                            print(magic_users)
                            magic_message += f"{sender_display_name} ✨\n"

            except requests.exceptions.RequestException as e:
                sentry_sdk.capture_exception(e)
                print(f"Error collecting responses: {e}")

        if magic_users:
            send_message(target_room_id, magic_message)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running main.py: {e}")
        print("Error output:", e.stderr)

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
    global since_token, time_between_questions, time_between_questions_default, ghost_mode, since_token, categories_to_exclude, num_crossword_clues, num_jeopardy_clues, num_mysterybox_clues, god_mode, yolo_mode
    time_between_questions = time_between_questions_default
    ghost_mode = ghost_mode_default
    categories_to_exclude.clear()
    num_crossword_clues = num_crossword_clues_default
    num_jeopardy_clues = num_jeopardy_clues_default
    num_mysterybox_clues = num_mysterybox_clues_default
    god_mode = god_mode_default
    yolo_mode = yolo_mode_default
    
    if round_winner is None:
        return
   

    # Notify the round winner about their award
    message = (
        f"\n🍔🍟 @{round_winner}, what's your order?\n\n"
        "⏱️⏳ <3 - 15>:  Time (s) between questions\n"
        "🟦✋ Jeopardy:  5 Jeopardy questions\n"
        "🟦❌ Trebek:  0 Jeopardy questions\n"
        "🚫👆 <Category>:  Exclude one category\n"
        "🔥🤘 Yolo:  No scores shown until the end\n"
        "👻🫥 Ghost: Responses and answers vanish\n"
    )

    if winner_points >= god_mode_points:
        message += "🎖🍆 Dicktator: Control the question order\n"

    message += "\nYou have 10 seconds."

    send_message(target_room_id, message)
    prompt_user_for_response(round_winner, winner_points)


def prompt_user_for_response(round_winner, winner_points):
    global since_token, time_between_questions, ghost_mode, num_jeopardy_clues, num_crossword_clues, num_mysterybox_clues, yolo_mode, god_mode
    
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
        
                    matched_category = cross_reference_category(message_content)
        
                    if matched_category:
                        categories_to_exclude[:1] = [matched_category]  # Add matched_category to exclude list
        
                        # Send message after handling special cases
                        send_message(target_room_id, f"🚫⛔ @{round_winner} has excluded {matched_category}.\n")
        
                    if "jeopardy" in message_content.lower():
                        num_jeopardy_clues = 5
                        send_message(target_room_id, f"🟦✋ @{round_winner} has added 5 Jeopardy questions.\n")
        
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
                        send_message(target_room_id, f"👻👻 @{round_winner} turned on 'Ghost Mode'. All answers will disappear.\n")

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
            model="gpt-3.5-turbo",
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
        return None

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
    return content_uri, img_width, img_height, str(median)




def generate_mean_question():
    """
    Generate a question asking for the mean of a set of random numbers.
    The set will contain 3 to 5 numbers between 1 and 10, and the image
    of the numbers will be sent to the user with the question.
    """
    # Generate a random n between 3 and 5
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
        return None

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
    content_uri = upload_image_to_matrix(image_buffer.read())

    # Return the integer mean for verification
    return content_uri, img_width, img_height, str(int(mean_value))





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

    # Construct the base prompt with different instructions if the winner is "username"
    if winner == "OkraStrut":
        prompt = (
            f"You are OkraStrut. You played trivia against everyone and dominated by a huge margin. "
            "Roast everyone except OkraStrut about how terrible they are and how easily you beat them. Rudely call out specific players except OkraStrut in your roast and call our specific wrong answers they gave. "
            "Create no more than 5 sentences in total. Be sarcastic, angry, offensive, and use emojis in your comments. End by complimenting yourself, OkraStrut, on how great you are. "
            "Here is a detailed summary of the trivia round with explicit mappings of user responses:\n"
            "Questions asked:\n"
        )
    elif winner in magic_users:
         prompt = (
            f"The winner of the trivia round is {winner}. "
            "Love bomb the winning player about their username and be very specific, positive, and loving. Compliment specific responses they gave during the round and talk about how much beter they are than eveyone else, including OkraStrut."
            "Create no more than 5 sentences in total. Be sweet, happy, positive, and use emojis in your response. "
            "Here is a detailed summary of the trivia round with explicit mappings of user responses:\n"
            "Questions asked:\n"
        )
    
    else:
        prompt = (
            f"The winner of the trivia round is {winner}. "
            "Roast the winning player about their username and be very specific and negative in your roast. Insult specific responses they gave during the round. "
            "Create no more than 5 sentences in total. Be sarcastic, very angry, offensive, and use emojis in your response. Deeply insult the winner using angry and rough language. "
            "Here is a detailed summary of the trivia round with explicit mappings of user responses:\n"
            "Questions asked:\n"
        )

    # Add questions, their correct answers, users' responses, and scoreboard status after each question
    for question_data in round_data["questions"]:
        question_number = question_data["question_number"]
        question_text = question_data["question_text"]
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


    # Use OpenAI's API to generate the summary
    try:
        if winner == "OkraStrut":
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are OkraStrut, an arrogant trivia master who always wins."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                n=1,
                stop=None,
                temperature=0.8,
            )

        elif winner in magic_users:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a loving old man who is completely in love with the winning trivia player."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                n=1,
                stop=None,
                temperature=0.8,
            )
            
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a bitter, dirty, and raunchy old curmudgeon who is completely fed up with all the trivia players."},
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



def send_disappearing_message(room_id, message):
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
                time.sleep(2)
                redact_message(message_id)
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
    
    if is_valid_url(trivia_url): 
        image_mxc, image_width, image_height = download_image_from_url(trivia_url) #FILE TYPE
        message_body = f"\n{number_block}📷 {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
        image_size = 100
        send_image_flag = True
        
    elif trivia_url == "polynomial":
        image_mxc, image_width, image_height, new_solution = generate_and_render_polynomial_image() #POLY
        message_body = f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
        image_size = 100
        send_image_flag = True
        
    elif trivia_url == "scramble":
        image_mxc, image_width, image_height = generate_scrambled_image(scramble_text(trivia_answer_list[0]))
        message_body = f"\n{number_block}🧩 {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
        image_size = 100
        send_image_flag = True

    elif trivia_url == "median":
        image_mxc, image_width, image_height, new_solution = generate_median_question()
        message_body = f"\n{number_block}📊 {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
        image_size = 100
        send_image_flag = True

    elif trivia_url == "mean":
        image_mxc, image_width, image_height, new_solution = generate_mean_question()
        message_body = f"\n{number_block}📊 {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
        image_size = 100
        send_image_flag = True


    elif trivia_url == "jeopardy":
        image_mxc, image_width, image_height = generate_jeopardy_image(trivia_question)
        message_body = f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\nAnd the answer is: \n"
        image_size = 100
        send_image_flag = True

    elif trivia_category == "Crossword":
        image_mxc, image_width, image_height = generate_crossword_image(trivia_answer_list[0])
        message_body = f"\n{number_block}✏️ {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"
        image_size = 100
        send_image_flag = True

    else:
         message_body = f"\n{number_block} {get_category_title(trivia_category, trivia_url)}\n\n{trivia_question}\n"

    response = send_message(target_room_id, message_body)

    if response is None:
        print("Error: Failed to send the message.")
        return None, None, None
            
    if send_image_flag:  
        response = send_image(target_room_id, image_mxc, image_width, image_height, image_size)

        if response is None:                      
            print("Error: Failed to send image.")
            return None, None, None
            
    initialize_sync()
    
    correct_answers = [new_solution] if new_solution else trivia_answer_list
    round_data["questions"].append({
        "question_number": question_number,
        "question_category": trivia_category,
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
    user_answer = str(user_answer)  # Ensure user_answer is also a string
    correct_answer = str(correct_answer)  # Convert to string
    
    if not user_answer or not correct_answer:           #POLY
         return user_answer == correct_answer  # Only accept exact match if either are empty        #POLY
    
    no_spaces_user = user_answer.replace(" ", "")       #POLY
    no_spaces_correct = correct_answer.replace(" ", "") #POLY

    if no_spaces_user == no_spaces_correct:     #POLY
        return True
    
    if is_number(correct_answer):
        return user_answer == correct_answer  # Only accept exact match if the correct answer is a number

    if len(user_answer) < 4 or len(correct_answer) < 4:
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

    #Check if user's answer is a substring of the correct answer after normalization
    if user_answer in correct_answer:
        return True
    
    # Step 1: Exact match or Partial match
    if correct_answer in user_answer:
        return True

    # Step 1: Exact match or Partial match
    if user_answer in correct_answer and len(user_answer) >= min(5, len(correct_answer)):
        return True
    
    # Step 2: Levenshtein similarity
    if levenshtein_similarity(user_answer, correct_answer) >= threshold:
        return True

    # Step 3: Jaccard similarity (Character level)
    #if jaccard_similarity(user_answer, correct_answer) >= threshold:
    #    return True

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

        except requests.exceptions.RequestException as e:
            sentry_sdk.capture_exception(e)
            print(f"Error collecting responses: {e}")

    return collected_responses


def check_correct_responses_delete(question_ask_time, trivia_answer_list, question_number, collected_responses):
    """Check and respond to users who answered the trivia question correctly."""
    global since_token, params, filter_json, headers, max_retries, delay_between_retries, current_longest_answer_streak
    
    # Define the first item in the list as trivia_answer
    trivia_answer = trivia_answer_list[0]  # The first item is the main answer
    correct_responses = []  # To store users who answered correctly
    has_responses = False  # Track if there are any responses

    fastest_correct_user = None
    fastest_response_time = None
    fastest_correct_event_id = None

    # Process collected responses
    for response in collected_responses:
        sender = response["user_id"]
        event_id = response["event_id"]
        display_name = get_display_name(sender)  # Get the display name from content
        
        # Check if the user has already answered correctly, ignore if they have
        if any(resp[0] == display_name for resp in correct_responses):
            continue  # Ignore this response since the user has already answered correctly
    
        # Log user submission (MongoDB operation)
        log_user_submission(display_name)
                            
        message_content = response.get("message_content", "")  # Use 'response' instead of 'event'
        normalized_message_content = normalize_text(message_content)
        
        if "okra" in message_content.lower() and emoji_mode == True:
            react_to_message(event_id, target_room_id, "okra1")
            
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
    #if ghost_mode == True:
    #    message = f"\n✅ Answer ✅\n👻👻👻👻👻\n"
    #else:
    #    message = f"\n✅ Answer ✅\n{trivia_answer}\n"

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
        send_disappearing_message(target_room_id, message)

    flush_submission_queue() 
    return None


def check_correct_responses(question_ask_time, trivia_answer_list, question_number):
    """Check and respond to users who answered the trivia question correctly."""
    global since_token, params, filter_json, headers, max_retries, delay_between_retries, current_longest_answer_streak
    sync_url = f"{matrix_base_url}/sync"
    
    # Define the first item in the list as trivia_answer
    trivia_answer = trivia_answer_list[0]  # The first item is the main answer
    correct_responses = []  # To store users who answered correctly
    has_responses = False  # Track if there are any responses
   
    for attempt in range(max_retries):
        try:
            fastest_correct_user = None
            fastest_response_time = None
            
            if since_token:
                params["since"] = since_token

            response = requests.get(sync_url, headers=headers, params=params)
            
            if response.status_code == 200:
                sync_data = response.json()
                since_token = sync_data.get("next_batch")  # Update the since token
                
                # Process messages from the room
                responses.clear()
                for room_id, room_data in sync_data.get("rooms", {}).get("join", {}).items():
                    if room_id == target_room_id:  # Only process messages from the target room
                        for event in room_data.get("timeline", {}).get("events", []):
                            sender = event["sender"]
                            display_name = get_display_name(event.get("content", {}).get("displayname", sender))  # Get the display name from content
                            
                            # Check if the user has already answered correctly, ignore if they have
                            if any(resp[0] == display_name for resp in correct_responses):
                                continue  # Ignore this response since the user has already answered correctly

                            # Log user submission (MongoDB operation)
                            log_user_submission(display_name)

                            emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟", "🛑"]
                            message_content = event.get("content", {}).get("body", "")
                            normalized_message_content = normalize_text(message_content)
                        
                            # Continue loop only if sender is bot_user_id and message contains one of the specified emojis
                            if sender == bot_user_id and any(emoji in message_content for emoji in emojis):
                                continue
                                
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
            message = f"\n🏆 Winner: @{user}...🔥{current_longest_round_streak['streak']} in a row!\n\n{summary}\n\n▶️ Live trivia stats available: https://stats.redditlivetrivia.com\n"
        else:
            message = f"\n🏆 Winner: @{user}!\n\n{summary}\n\n▶️ Live trivia stats available: https://stats.redditlivetrivia.com\n"

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

            if  user in magic_users:
                standing_message += " ✨"
        
        send_message(target_room_id, standing_message)


def load_trivia_questions():
    #Dynamically load the trivia_questions module
    trivia_module = importlib.import_module(questions_module)
    importlib.reload(trivia_module)  # Reload in case the file has changed
    return trivia_module.trivia_questions


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

        selected_questions = []



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
        remaining_needed = max(questions_per_round - len(mysterybox_questions) - len(crossword_questions) - len(jeopardy_questions), 0)

        if remaining_needed > 0:

            trivia_collection = db["trivia_questions"]
            # Define the maximum number of questions per category
            max_questions_per_category = 2
            
            pipeline_trivia = [
                {"$match": {"_id": {"$nin": list(recent_general_ids)}, "category": {"$nin": categories_to_exclude}}},
                {
                    "$group": {
                        "_id": "$category",
                        "questions": {"$push": "$$ROOT"}  # Push full document to each category group
                    }
                },
                {
                    "$project": {
                        "category": "$_id",
                        "questions": {"$slice": ["$questions", max_questions_per_category]}  # Limit number of questions per category
                    }
                },
                {"$unwind": "$questions"},  # Unwind the limited question list for each category back into individual documents
                {"$replaceRoot": {"newRoot": "$questions"}},  # Flatten to original document structure
                {"$sample": {"size": remaining_needed}}  # Sample from the resulting limited set
            ]
            
            trivia_questions = list(trivia_collection.aggregate(pipeline_trivia))
            selected_questions.extend(trivia_questions)

            # Store separate sets of IDs in MongoDB only if they are non-empty
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

    print("entering round start")
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
        max_questions_per_category = 2
            
        pipeline = [
            {"$match": {"_id": {"$nin": list(recent_general_ids)}, "category": {"$nin": categories_to_exclude}}},
            {
                "$group": {
                    "_id": "$category",
                    "questions": {"$push": "$$ROOT"}  # Push full document to each category group
                }
            },
            {
                "$project": {
                    "category": "$_id",
                    "questions": {"$slice": ["$questions", max_questions_per_category]}  # Limit number of questions per category
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
    global target_room_id, bot_user_id, bearer_token, question_time, questions_per_round, time_between_rounds, time_between_questions, questions_module, filler_words
    global scoreboard, current_longest_round_streak, current_longest_answer_streak
    global headers, params, filter_json, since_token, round_count, selected_questions

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
            
            # Load existing streak data from the file
            load_streak_data()

            # Reset the scoreboard and fastest answers at the start of each round
            scoreboard.clear()
            fastest_answers_count.clear()
            magic_users.clear()
            
            # Reset round data for the next round
            round_data["questions"] = []

            # Select a random GIF URL
            
                
            # Send the selected GIF
            #


            
            send_message(target_room_id, f"\n⏩ Starting a round of {questions_per_round} questions ⏩\n\n🏁 Get ready 🏁\n\n")
            round_start_messages()

            if random.random() < 1.0:  # random.random() generates a float between 0 and 1
                magic_number = random_number = random.randint(1000, 9999)
                print(f"Magic number is {magic_number}")
                generate_magic_image(magic_number)
            else:
                selected_gif_url = random.choice(okra_gif_urls)
                print(selected_gif_url)
                image_mxc, image_width, image_height = download_image_from_url(selected_gif_url)
                send_image(target_room_id, image_mxc, image_width, image_height, image_size=100)
                time.sleep(7)
                
            # Randomly select n questions
            print() 
            print_selected_questions(selected_questions)
            print()
            
            question_number = 1
            while question_number <= questions_per_round:
                
                if god_mode and round_winner and len(selected_questions)>1:
                    selected_question = selected_questions[get_player_selected_question(selected_questions, round_winner) - 1]
                    
                else:
                    # Normal mode - sequential questions
                    selected_question = selected_questions[0]

                trivia_category, trivia_question, trivia_url, trivia_answer_list = selected_question
                
                # Ask the trivia question and get start times
                question_ask_time, new_question, new_solution = ask_question(trivia_category, trivia_question, trivia_url, trivia_answer_list, question_number)
                
                collected_responses = collect_responses(question_time, question_number, question_time)
                
                send_message(target_room_id, f"\n🛑 TIME 🛑\n")
                
                solution_list = trivia_answer_list if new_solution is None else [new_solution]
                
                check_correct_responses_delete(question_ask_time, solution_list, question_number, collected_responses)
                
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
                send_message(target_room_id, f"🛟 Help Okra keep it up\n☕️ https://buymeacoffee.com/livetrivia\n👕 https://merch.redditlivetrivia.com\n")
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
