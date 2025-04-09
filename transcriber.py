import sounddevice as sd
import numpy as np
import whisper
from scipy.io.wavfile import write

class Transcriber:
    def __init__(self):
        self.model_type = "turbo"
        self.model = self._load_model()
        self.DURATION = 60 
        self.SAMPLERATE = 16000 
        self.FILENAME = "recorded_audio_test.mp3"

    def _load_model(self):
        return whisper.load_model(self.model_type)
    
    def record_audio(self, filename=None):
        if not filename:
            filename = self.FILENAME
        print("Type Enter to start recording...")
        input()
        print("Recording has begun. Speak...")
        audio = sd.rec(int(self.DURATION * self.SAMPLERATE), samplerate=self.SAMPLERATE, channels=1, dtype='int16')
        sd.wait()
        write(filename, self.SAMPLERATE, audio)
        print(f"Recording completed and saved to file '{filename}'.")

    def transcribe_audio(self, filename=None):
        if not filename:
            filename = self.FILENAME
        result = self.model.transcribe(filename, word_timestamps=True,)
        print("\nTranscription of the record:")
        print(result["text"])
        return result


if __name__ == '__main__':
    transcriber = Transcriber()
    print("Starting audio recording. Press Enter to begin...")
    transcriber.record_audio()
    print("Recording finished. Starting transcription...")
    transcriber.transcribe_audio()
    