""" Speech Recognition and Translation Script
This script recognizes speech from the microphone, translates it to a specified language, and plays the translated speech using text-to-speech synthesis.
"""
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import tempfile
import pygame

def recognize_speech(lang_code="en-US"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"🎤 Speak now ({lang_code})...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        print("🧠 Recognizing...")
        text = r.recognize_google(audio, language=lang_code)
        print(f"✅ Transcribed Text: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"❌ API error: {e}")
        return None

def translate_text(text, dest_lang='ta'):
    translator = Translator()
    try:
        translated = translator.translate(text, dest=dest_lang)
        print(f"🌐 Translated Text ({dest_lang}): {translated.text}")
        return translated.text
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return None

import uuid

def speak_text2(text, lang='ta'):
    try:
        tts = gTTS(text=text, lang=lang)
        
        # Create a temp .mp3 file with a unique name
        filename = f"temp_audio_{uuid.uuid4().hex}.mp3"
        tts.save(filename)

        print("🔊 Playing translated speech with pygame...")
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Let playback finish

        os.remove(filename)  # Remove after playback
    except Exception as e:
        print(f"❌ TTS or playback error: {e}")

import uuid
import traceback

def speak_text(text, lang='ta'):
    try:
        tts = gTTS(text=text, lang=lang)
        
        filename = f"temp_audio_{uuid.uuid4().hex}.mp3"
        tts.save(filename)

        print("🔊 Playing translated speech with pygame...")
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.quit()
        os.remove(filename)
    except Exception as e:
        print(f"❌ TTS or playback error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    source_lang_code = "en-US"  # Input speech language (e.g., English)
    target_lang_code = "ta"     # Output language (e.g., Tamil)

    original_text = recognize_speech(lang_code=source_lang_code)

    if original_text:
        translated_text = translate_text(original_text, dest_lang=target_lang_code)
        if translated_text:
            speak_text(translated_text, lang=target_lang_code)
