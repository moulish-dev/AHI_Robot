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

model_name = "gemma3:1b"

# Initialize conversation
messages = [
    {"role": "system","content": "You are a helpful assistant and a friend to me."},
    {"role": "user", "content": "Hello!"},
]

# First response from the bot
response = ollama.chat(model=model_name, messages=messages)
print("Bot:", response.message.content)

# def ask_ollama(prompt):
#     # response = requests.post(
#     #     "http://localhost:11434/api/generate",
#     #     json={"model": "gemma3:1b", "prompt": prompt},
#     #     stream=False
#     # )
#     # print("Full Ollama Response JSON:", response.json())  # Debug
#     # return response.json()["response"]
#     response = ollama.generate(model="gemma3:1b",prompt=prompt)
#     return response.get("response", "No response")

def speak(text):
    pipeline = KPipeline(lang_code='a',repo_id='hexgrad/Kokoro-82M')
    generator = pipeline(text, voice='af_heart')
    for i, (gs, ps, audio) in enumerate(generator):
        # print(f"Segment {i}: {gs} | {ps}")
        file_path = f'{i}.wav'
        sf.write(file_path, audio, 24000)
        os.system(f"aplay {file_path}")

# Continue the conversation
while True:
    # 🚀 Main loop
    user_prompt = input("You: ")
    if not user_input:
        break  # exit loop on empty input
    messages.append({"role": "user", "content": user_input})
    response = ollama.chat(model=model_name, messages=messages)
    answer = response.message.content
    print("Ollama:", answer)
    speak(answer)
    messages.append({"role": "assistant", "content": answer})
