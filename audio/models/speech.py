# models/speech.py (assumed implementation)

import speech_recognition as sr
from gtts import gTTS
import io
import pygame
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
        self.recognizer = sr.Recognizer()

    def recognize_speech2(self, lang_code="en"):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            try:
                return self.recognizer.recognize_google(audio, language=lang_code)
            except sr.UnknownValueError:
                return None
            except sr.RequestError:
                return None

    def recognize_speech(self, lang_code="en-US"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print(f"🎤 Speak now ({lang_code})...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            try:
                return self.recognizer.recognize_google(audio, language=lang_code)
            except sr.UnknownValueError:
                return None
            except sr.RequestError:
                return None

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

    def speak_text2(self, text, lang="en"):
        try:
            tts = gTTS(text=text, lang=lang)
            audio_file = io.BytesIO()
            tts.write_to_fp(audio_file)
            audio_file.seek(0)
            return audio_file
        except Exception:
            return None