# SETUP OLLAMA FIRST

# https://github.com/moulish-dev/vita.git
# USE VITA FOR VOICE OUTPUT - 

# Using Gemma3:1b model

import requests
import subprocess
import ollama

# KokoroTTS
from kokoro import KPipeline
import soundfile as sf
import torch
import os



def ask_ollama(prompt):
    # response = requests.post(
    #     "http://localhost:11434/api/generate",
    #     json={"model": "gemma3:1b", "prompt": prompt},
    #     stream=False
    # )
    # print("Full Ollama Response JSON:", response.json())  # Debug
    # return response.json()["response"]
    response = ollama.generate(model="gemma3:1b",prompt=prompt)
    return response.get("response", "No response")

def speak(text):
    pipeline = KPipeline(lang_code='a',repo_id='hexgrad/Kokoro-82M')
    generator = pipeline(text, voice='af_heart')
    for i, (gs, ps, audio) in enumerate(generator):
        # print(f"Segment {i}: {gs} | {ps}")
        file_path = f'{i}.wav'
        sf.write(file_path, audio, 24000)
        os.system(f"aplay {file_path}")

# 🚀 Main loop
user_prompt = input("You: ")
ollama_reply = ask_ollama(user_prompt)
print("Ollama:", ollama_reply)
speak(ollama_reply)