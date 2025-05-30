# SETUP OLLAMA FIRST

# https://github.com/moulish-dev/vita.git
# USE VITA FOR VOICE OUTPUT - 

# Using Gemma3:1b model

# remove old python version
# sudo rm -rf /usr/local/bin/python3.11
# sudo rm -rf /usr/local/lib/python3.11
# sudo rm -rf /usr/local/include/python3.11

# python version for this project
# sudo apt update
# sudo apt install software-properties-common -y
# sudo add-apt-repository ppa:deadsnakes/ppa -y
# sudo apt update
# sudo apt install python3.11 python3.11-venv python3.11-dev -y

import requests
import subprocess
import ollama

# KokoroTTS
from kokoro import KPipeline
import soundfile as sf
import torch
import os
import warnings
import time
import numpy as np
import uuid
import queue
from RealtimeSTT import AudioToTextRecorder

recorder = None  # Global so main.py can access

def get_voice_input():
    print("🎤 Speak now...")
    with AudioToTextRecorder() as recorder:
        text = recorder.text()
        print(f"📝 You said: {text}")
        return text


audio_queue = queue.Queue()

# to supress pytorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

model_name = "gemma3:1b"
debug = False

# Initialize conversation
messages = [
    {"role": "system","content": "You are a helpful assistant and a friend to me."},
    {"role": "user", "content": "Hello!"},
]

# First response from the bot
response = ollama.chat(
    model=model_name, 
    messages=messages,
    stream=True,
    )
for chunk in response:
  print(chunk['message']['content'], end='', flush=True)


from datetime import datetime

LOG_FILE = "conversation_log.txt"

def log_entry(role, text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {role.upper()}:\n{text.strip()}\n\n")

def speak(text):
    
    start = time.time()
    generator = pipeline(text, voice='af_heart')
    all_audio = []

    for _, _, audio in generator:
        all_audio.append(audio)

    # Concatenate all chunks
    combined_audio = np.concatenate(all_audio)
    sf.write('response.wav', combined_audio, 24000)

    duration = time.time() - start
    print(f"🕒 Kokoro processing time: {duration:.2f} seconds")

    # Play once
    os.system('aplay response.wav')

    os.remove("response.wav")

import re

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)

def split_into_chunks(text):
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

def audio_worker():
    while True:
        filename = audio_queue.get()
        if filename is None:
            break  # Allows clean shutdown if needed

        os.system(f"aplay {filename} > /dev/null 2>&1")
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
        audio_queue.task_done()

import threading
pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

def threaded_speak(text):
    def _speak():
        cleaned = remove_emojis(text.strip())
        chunks = split_into_chunks(cleaned)
        for chunk in chunks:
            
            generator = pipeline(chunk, voice='af_heart')
            all_audio = []
            for _, _, audio in generator:
                all_audio.append(audio)
            combined_audio = np.concatenate(all_audio)
            

            file_id = str(uuid.uuid4())[:8]  # unique short ID
            filename = f"response_{file_id}.wav"

            sf.write(filename, combined_audio, 24000)
            
            # Enqueue the audio file for playback
            audio_queue.put(filename)

            

            
    thread = threading.Thread(target=_speak)
    thread.start()

# Start background thread to process audio queue
playback_thread = threading.Thread(target=audio_worker, daemon=True)
playback_thread.start()

# Continue the conversation
while True:
    # 🚀 Main loop
    print("\n🎤 Speak now (or type if silent)...")
    user_prompt = get_voice_input()

    # If voice input is empty, fall back to typing
    if not user_prompt:
        user_prompt = input("⌨️  You (typed): ").strip()

    # If both are empty, exit
    if not user_prompt:
        print("👋 No input given. Exiting.")
        break # exit loop on empty input
    messages.append({"role": "user", "content": user_prompt})
    log_entry("User", user_prompt)
    response = ollama.chat(model=model_name, messages=messages, stream=True)
    full_reply = ""
    buffer = ""
    for chunk in response:
        piece = chunk['message']['content'] # as this is being used more time it is a variable
        print(piece, end='', flush=True)
        full_reply += piece # for the whole conversation to be saved
        buffer += piece     # for the audio streaming

        if any(p in buffer for p in [".", "!", "?"]) and len(buffer.strip()) > 15:
            threaded_speak(buffer.strip())
            buffer = ""
    # speak(answer)
    log_entry("Assistant: ", full_reply)
    messages.append({"role": "assistant", "content": full_reply})



audio_queue.put(None)
playback_thread.join()
