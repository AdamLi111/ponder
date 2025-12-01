"""
Speech recognition and transcription module
Handles audio recording, downloading from Misty, and transcription
Now using sounddevice (better Apple Silicon support than PyAudio)
"""
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import requests
import os
import threading
import io
import wave


class SpeechHandler:
    def __init__(self, robot_ip, use_laptop_mic=False):
        """
        Initialize speech handler
        
        Args:
            robot_ip: IP address of Misty robot
            use_laptop_mic: If True, use laptop microphone instead of Misty's
        """
        self.robot_ip = robot_ip
        self.recognizer = sr.Recognizer()
        self.temp_audio_path = 'temp_audio.wav'
        self.use_laptop_mic = use_laptop_mic
        self.is_listening = False
        self.callback = None
        
        # Audio parameters
        self.sample_rate = 16000  # 16kHz for speech recognition
        self.channels = 1  # Mono
        
        if use_laptop_mic:
            print("Using laptop microphone for audio input")
            print(f"Default input device: {sd.query_devices(kind='input')['name']}")
            
            # Test recording to calibrate
            print("Calibrating ambient noise... Please wait.")
            self._calibrate_ambient_noise()
            print("Calibration complete!")
    
    def _calibrate_ambient_noise(self):
        """Calibrate for ambient noise by recording silence"""
        try:
            # Record 1 second of ambient noise
            duration = 1.0
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16'
            )
            sd.wait()
            
            # Convert to AudioData for speech_recognition
            audio_data = self._numpy_to_audio_data(recording)
            
            # Adjust recognizer for ambient noise
            with io.BytesIO(audio_data) as source_file:
                with sr.AudioFile(source_file) as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    
        except Exception as e:
            print(f"Warning: Ambient noise calibration failed: {e}")
    
    def _numpy_to_audio_data(self, audio_array):
        """Convert numpy array to WAV bytes for speech_recognition"""
        # Ensure audio is int16
        if audio_array.dtype != np.int16:
            audio_array = (audio_array * 32767).astype(np.int16)
        
        # Flatten if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.flatten()
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_array.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    def record_audio(self, duration=5.0, wait_for_sound=True):
        """
        Record audio from laptop microphone
        
        Args:
            duration: Maximum duration to record in seconds
            wait_for_sound: If True, wait for sound above threshold before recording
            
        Returns:
            numpy array of audio data
        """
        if wait_for_sound:
            print("Listening for sound...")
            # Wait for sound above threshold
            threshold = 0.005  # Adjust based on environment
            while self.is_listening:
                # Record short chunk
                chunk = sd.rec(
                    int(0.1 * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='float32'
                )
                sd.wait()
                
                # Check if sound detected
                if np.abs(chunk).mean() > threshold:
                    print("Sound detected, recording...")
                    break
        
        # Record the actual audio
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16'
        )
        sd.wait()
        
        return recording
    
    def transcribe_recording(self, audio_array):
        """
        Transcribe a numpy audio array using Google Speech Recognition
        
        Args:
            audio_array: numpy array of audio data
            
        Returns:
            Transcribed text or None
        """
        try:
            # Convert to WAV bytes
            audio_bytes = self._numpy_to_audio_data(audio_array)
            
            # Use speech_recognition to transcribe
            with io.BytesIO(audio_bytes) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio = self.recognizer.record(source)
            
            # Transcribe with Google Speech Recognition
            text = self.recognizer.recognize_google(audio)
            return text
            
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            print(f"Error transcribing: {e}")
            return None
    
    def listen_for_wake_word(self, wake_word="hey misty", timeout=None):
        """
        Listen for wake word and return the command after it
        
        Args:
            wake_word: Wake phrase to listen for
            timeout: Maximum time to listen (None for indefinite)
            
        Returns:
            Command text after wake word, or None
        """
        print(f"Listening for wake word: '{wake_word}'...")
        
        start_time = None
        if timeout:
            import time
            start_time = time.time()
        
        while self.is_listening:
            # Check timeout
            if timeout and start_time:
                import time
                if time.time() - start_time > timeout:
                    return None
            
            # Record audio chunk
            try:
                audio = self.record_audio(duration=5.0, wait_for_sound=True)
                
                # Transcribe
                text = self.transcribe_recording(audio)
                
                if text:
                    text_lower = text.lower()
                    print(f"Heard: {text}")
                    
                    # Check for wake word
                    if wake_word.lower() in text_lower:
                        print(f"✓ Wake word '{wake_word}' detected!")
                        
                        # Extract command after wake word
                        wake_idx = text_lower.find(wake_word.lower())
                        command = text[wake_idx + len(wake_word):].strip()
                        
                        if command:
                            return command
                        else:
                            # No command after wake word, listen for next phrase
                            print("Listening for command...")
                            audio = self.record_audio(duration=5.0, wait_for_sound=True)
                            command = self.transcribe_recording(audio)
                            if command:
                                return command
                            
            except Exception as e:
                print(f"Error in listen loop: {e}")
                continue
        
        return None
    
    def listen_for_command(self, duration=5.0):
        """
        Listen for a command (without wake word requirement)
        Used after wake word has been detected once
        
        Args:
            duration: Maximum duration to record
            
        Returns:
            Command text or None
        """
        print("Listening for command...")
        
        try:
            audio = self.record_audio(duration=duration, wait_for_sound=True)
            text = self.transcribe_recording(audio)
            
            if text:
                print(f"Command: {text}")
                return text
            else:
                print("No speech detected")
                return None
                
        except Exception as e:
            print(f"Error listening for command: {e}")
            return None
    
    def listen_continuously(self, callback_func):
        """
        Start continuous listening with laptop microphone
        Waits for wake word once, then all subsequent audio is treated as commands
        Calls callback_func(command_text) for each command
        """
        if not self.use_laptop_mic:
            print("Error: Continuous listening only available with laptop microphone")
            return
        
        self.is_listening = True
        self.callback = callback_func
        
        def listen_loop():
            # Wait for wake word first
            wake_detected = False
            
            while self.is_listening:
                if not wake_detected:
                    # First time - wait for wake word
                    command = self.listen_for_wake_word()
                    if command:
                        wake_detected = True
                        print("✓ Wake word detected! Now listening for all commands (no wake word needed)")
                        # Process the command that came with wake word
                        if self.callback:
                            self.callback(command)
                else:
                    # After wake word - just listen for commands
                    command = self.listen_for_command()
                    if command and self.callback:
                        self.callback(command)
        
        # Start listening in a separate thread
        self.listen_thread = threading.Thread(target=listen_loop, daemon=True)
        self.listen_thread.start()
        print("Started continuous listening on laptop microphone")
        print("Say 'Hey Misty' followed by your first command to activate")
    
    def stop_listening(self):
        """Stop continuous listening"""
        self.is_listening = False
        print("Stopped listening")
    
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
        self.stop_listening()
        if os.path.exists(self.temp_audio_path):
            os.remove(self.temp_audio_path)