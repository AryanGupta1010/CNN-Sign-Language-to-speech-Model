from gtts import gTTS
import pygame
import time
import os

class SpeechModule:
    def __init__(self, cooldown=2.0):
        pygame.mixer.init()
        self.last_spoken = None
        self.last_time = 0
        self.cooldown = cooldown

    def speak(self, text):
        current_time = time.time()

        if text == self.last_spoken and (current_time - self.last_time < self.cooldown):
            return

        print(f"[SPEAKING]: {text}")

        filename = "temp.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        # wait till speech finishes
        while pygame.mixer.music.get_busy():
            continue

        pygame.mixer.music.unload()
        os.remove(filename)

        self.last_spoken = text
        self.last_time = current_time


CONFIDENCE_THRESHOLD = 0.6

def process_prediction(label, confidence, speech_module):
    if confidence < CONFIDENCE_THRESHOLD:
        return

    speech_module.speak(label)