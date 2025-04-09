
import openwillis.speech as ows
import openwillis.transcribe as owt
import os
import nltk
import traceback

from transcriber import Transcriber

class VoiceCharacteristics(Transcriber):
    """
    A class to represent the characteristics of a voice.
    """

    def __init__(self):
        self.transcript_json = None
        self.language = None

    def get_transcript_json(self, filename=None):
        if filename is not None:
            self.transcribe_json = self.transcribe_audio(filename)
            self.language = self.transcript_json["language"]
        else:
            print("No filename provided for transcription.")
            return None

    def get_speech_characteristics(self, json_conf=None):
        """
        Retrieves speech characteristics based on the transcript JSON.
        If no JSON configuration (json_conf) is provided, the method uses self.transcribe_json.
        """
        json_conf = json_conf or self.transcribe_json
        if not json_conf:
            print("No transcript JSON available. Please run get_transcript_json() first.")
            return None

        try:
            return ows.speech_characteristics(json_conf=json_conf, 
                                              option='coherence', 
                                              language=self.language, 
                                              speaker_label='SPEAKER_A')
        except Exception as e:
            print("Error in getting speech characteristics:", e)
            traceback.print_exc()
            return None
