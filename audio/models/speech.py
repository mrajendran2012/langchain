""" Speech Recognition and Translation Script
Speech Recognition and Translation Script
This script provides functionality to:
- Recognize speech from the microphone in a specified language.
- Translate the recognized text to a target language.
- Synthesize the translated text into speech and play it back.
Modules Used:
- speech_recognition: For capturing and transcribing speech from the microphone.
- googletrans: For translating text between languages.
- gtts: For converting text to speech (TTS).
- pygame: For audio playback of the synthesized speech.
- os, tempfile, uuid, traceback: For file management and error handling.
Functions:
- recognize_speech(lang_code="en-US"):
    Captures audio from the microphone and transcribes it to text using Google's speech recognition API.
    Args:
        lang_code (str): The language code for speech recognition (default: "en-US").
    Returns:
        str or None: The transcribed text, or None if recognition fails.
- translate_text(text, dest_lang='ta'):
    Translates the given text to the specified destination language using Google Translate.
    Args:
        text (str): The text to translate.
        dest_lang (str): The target language code (default: 'ta' for Tamil).
    Returns:
        str or None: The translated text, or None if translation fails.
- speak_text2(text, lang='ta'):
    Converts the given text to speech using gTTS and plays it using pygame.
    Args:
        text (str): The text to synthesize.
        lang (str): The language code for TTS (default: 'ta').
    Notes:
        This function creates a temporary mp3 file for playback and deletes it after use.
- speak_text(text, lang='ta'):
    Similar to speak_text2, but includes additional error handling and ensures pygame mixer is properly quit.
    Args:
        text (str): The text to synthesize.
        lang (str): The language code for TTS (default: 'ta').
Usage:
- Run the script directly to:
    1. Prompt the user to speak in the source language.
    2. Transcribe the speech to text.
    3. Translate the text to the target language.
    4. Synthesize and play the translated speech.
Example:
    python input.py
Note:
- Ensure that the required libraries are installed and a microphone is available.
- The script uses Google APIs for speech recognition and translation, which require internet access.
This script recognizes speech from the microphone, translates it to a specified language, and plays the translated speech using text-to-speech synthesis.
"""
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import tempfile
import pygame
import uuid
import traceback
from io import BytesIO
import uuid
from gtts import gTTS
import os

class PygameAudio:
    def __init__(self):
        pygame.mixer.init()

    def play(self, filename):
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Let playback finish

    def quit(self):
        pygame.mixer.quit()

    def recognize_speech(self, lang_code="en-US"):
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

    def translate_text(self, text, dest_lang='ta'):
        translator = Translator()
        try:
            translated = translator.translate(text, dest=dest_lang)
            print(f"🌐 Translated Text ({dest_lang}): {translated.text}")
            return translated.text
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return None

    import uuid

    def speak_text2(self, text, lang='ta'):
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


    def speak_text(self, text, lang='ta') -> BytesIO:
        try:
            # Create TTS object and save to buffer
            tts = gTTS(text=text, lang=lang)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)

            # Save to a temp file for playback
            tmp_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{uuid.uuid4().hex}.mp3")
            tts.save(tmp_path)

            # Play audio with pygame
            print("🔊 Playing translated speech with pygame...")
            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.quit()

            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            return audio_buffer

        except Exception as e:
            print(f"❌ TTS or playback error: {e}")
            traceback.print_exc()
            return None

    def speak_text2(self, text, lang='ta'):
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