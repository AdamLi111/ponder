from mistyPy.Robot import Robot
from mistyPy.Events import Events
import time
import re
import speech_recognition as sr
import requests
from io import BytesIO

def process_voice_command(command):
    """Process voice commands and execute corresponding actions"""
    command = command.lower()
    print(f"Processing command: {command}")
    
    # Extract distance if mentioned (default to 1 meter)
    distance_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:meter|metre)', command)
    distance = float(distance_match.group(1)) if distance_match else 1.0
    
    # Calculate drive time (approximate: 0.5 meters per second at speed 50)
    drive_time_ms = int((distance / 0.5) * 1000)
    
    if 'forward' in command in command:
        print(f"Moving forward {distance} meter(s)")
        misty.speak(f"Moving forward {distance} meters")
        misty.drive_time(50, 0, drive_time_ms)
    else:
        print("Command not recognized")
        misty.speak("Sorry, I didn't understand that command")

def transcribe_audio_from_misty(audio_filename):
    """Download and transcribe audio file from Misty"""
    try:
        # Download the audio file from Misty
        audio_url = f"http://{ip_address}/api/audio?FileName={audio_filename}"
        print(f"Downloading audio from: {audio_url}")
        
        response = requests.get(audio_url)
        if response.status_code == 200:
            # Use speech_recognition library to transcribe
            recognizer = sr.Recognizer()
            
            # Save audio temporarily
            audio_data = BytesIO(response.content)
            
            # Convert to AudioFile format
            with open('temp_audio.wav', 'wb') as f:
                f.write(response.content)
            
            with sr.AudioFile('temp_audio.wav') as source:
                audio = recognizer.record(source)
                
            # Transcribe using Google Speech Recognition (free)
            try:
                text = recognizer.recognize_google(audio)
                print(f"Transcribed text: {text}")
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return None
        else:
            print(f"Failed to download audio: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def Voice_Record_Callback(data):
    """Callback for voice recording events"""
    print("Voice_Record_Callback triggered!")
    print(f"Full data received: {data}")
    
    try:
        if isinstance(data, dict) and 'message' in data:
            message = data['message']
            filename = message.get('filename', '')
            
            print(f"Audio filename: {filename}")
            
            # Skip the key phrase recording
            if 'HeyMisty' in filename:
                print("Skipping key phrase audio")
                return
            
            # Transcribe the audio using Google Speech Recognition
            print("Transcribing audio...")
            misty.speak("Processing your command")
            
            speech_text = transcribe_audio_from_misty(filename)
            
            if speech_text:
                print(f"Heard: {speech_text}")
                process_voice_command(speech_text)
            else:
                print("Could not transcribe audio")
                misty.speak("Sorry, I couldn't understand that")
                
    except Exception as e:
        print(f"Error processing voice: {e}")
        import traceback
        traceback.print_exc()

def Key_Phrase_Recognized(data):
    """Callback when key phrase is recognized"""
    print("="*50)
    print("Key phrase detected! Listening for command...")
    print("="*50)
    
    misty.display_image("e_Surprise.jpg", 1)
    misty.change_led(0, 255, 0)  # Green LED to show listening
    
    # Start speech capture with minimal parameters
    print("Starting speech capture...")
    try:
        result = misty.capture_speech()
        print(f"Capture speech result: {result}")
    except Exception as e:
        print(f"Error starting speech capture: {e}")
    
    time.sleep(5)
    misty.change_led(0, 0, 255)

if __name__ == "__main__":
    ip_address = "172.20.10.2"
    misty = Robot(ip_address)

    print("Setting up Misty...")
    misty.display_image("e_SleepingZZZ.jpg", 1)
    misty.move_head(60, 0, 0, 80)
    misty.move_arms(85, 85, 80, 80)
    misty.change_led(0, 0, 255)
    
    # Start key phrase recognition
    print("Starting key phrase recognition...")
    misty.start_key_phrase_recognition()
    
    # Register events
    print("Registering Key Phrase event...")
    misty.register_event(
        event_name='Key_Phrase_Recognized',
        event_type=Events.KeyPhraseRecognized,
        callback_function=Key_Phrase_Recognized,
        keep_alive=True
    )
    
    print("Registering VoiceRecord event...")
    misty.register_event(
        event_name='VoiceRecord',
        event_type=Events.VoiceRecord,
        callback_function=Voice_Record_Callback,
        keep_alive=True
    )
    
    print("\n" + "="*50)
    print("Misty is ready!")
    print("Testing feasibility, currently only support the following command:")
    print("  - 'move forward 1 meter'")
    print("="*50 + "\n")
    
    misty.keep_alive()