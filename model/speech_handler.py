"""
Speech recognition and transcription module
Handles audio recording, downloading from Misty, and transcription
"""
import speech_recognition as sr
import requests
import os

#whisper ai

class SpeechHandler:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.recognizer = sr.Recognizer()
        self.temp_audio_path = 'temp_audio.wav'
    
    def transcribe_audio_from_misty(self, audio_filename):
        """Download and transcribe audio file from Misty"""
        try:
            # Download the audio file from Misty
            audio_url = f"http://{self.robot_ip}/api/audio?FileName={audio_filename}"
            print(f"Downloading audio from: {audio_url}")
            
            response = requests.get(audio_url)
            if response.status_code != 200:
                print(f"Failed to download audio: {response.status_code}")
                return None
            
            # Save audio temporarily
            with open(self.temp_audio_path, 'wb') as f:
                f.write(response.content)
            
            # Transcribe using speech_recognition
            with sr.AudioFile(self.temp_audio_path) as source:
                audio = self.recognizer.record(source)
            
            # Use Google Speech Recognition (free)
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"Transcribed text: {text}")
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition; {e}")
                return None
                
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Clean up temporary audio files"""
        if os.path.exists(self.temp_audio_path):
            os.remove(self.temp_audio_path)