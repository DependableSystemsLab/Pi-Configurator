import speech_recognition as sr
from difflib import SequenceMatcher
import numpy as np
from scipy.io import wavfile


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


AUDIO_FILE = ''  # your input audio data

source = sr.AudioFile(AUDIO_FILE)
samplerate, audio_data = wavfile.read(AUDIO_FILE)
text = []
r = sr.Recognizer()
with source as file:
    sound = r.record(file)  # read the entire audio file
try:
    # text = r.recognize_sphinx(sound)
    text = r.recognize_google(sound)
    print(text)

    file1 = open("audioSpeech.txt", "w")
    file1.write(text)
    file1.close()
except Exception as e:
    print(e)

file2 = open("audioSpeech.txt", "r")
text2 = file2.read()
file2.close()

print(similar(text, text2))
