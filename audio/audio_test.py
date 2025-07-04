from models.speech import PygameAudio
speech = PygameAudio()
audio_bytes = speech.speak_text("Hello, this is a test", lang="en")
if audio_bytes:
    with open("test.wav", "wb") as f:
        f.write(audio_bytes.read())