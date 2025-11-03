from PythonSDKmain.mistyPy.Robot import Robot
from PythonSDKmain.mistyPy.Events import Events
import time
import re
import speech_recognition as sr
import requests
from io import BytesIO
import json

# TEST MODE: Set to 1 to type commands instead of using voice
TEST_MODE = 0  # Change to 0 for normal voice mode

def parse_intent_with_llm(user_speech):
    """Use Ollama (local) to parse user's actual intent from their speech"""
    print(f"Parsing intent for: '{user_speech}'")
    
    try:
        print("connecting to LLM")
        response = requests.post(
            'http://10.200.206.195:11434/api/generate',
            json={
                'model': 'llama3.2',  # or 'llama3.2', 'mistral', 'phi3', etc.
                'prompt': f"""You are a command parser. Extract the robot command.

User said: "{user_speech}"

IMPORTANT: Reply with ONLY a JSON object. Do NOT write code. Do NOT explain.

Format: {{"action": "forward", "distance": 1}}

Valid actions: forward, backward, left, right, stop, unknown

JSON:""",
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'num_predict': 100
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            llm_output = response.json()['response'].strip()
            print(f"LLM raw output: {llm_output}")
            
            # Extract JSON from response (sometimes models add extra text)
            json_start = llm_output.find('{')
            json_end = llm_output.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = llm_output[json_start:json_end]
                intent = json.loads(json_str)
                action = intent.get('action', 'unknown')
                distance = intent.get('distance', 1.0)
                
                print(f"Parsed intent - Action: {action}, Distance: {distance}")
                return action, distance
            else:
                print("Could not find JSON in response")
                return "unknown", 0
        else:
            print(f"Error from Ollama: {response.status_code}")
            return "unknown", 0
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Make sure Ollama is running with 'ollama serve'")
        return "unknown", 0
    except Exception as e:
        print(f"Error calling LLM: {e}")
        import traceback
        traceback.print_exc()
        return "unknown", 0

def process_voice_command(action, distance):
    """Process voice commands based on parsed action and distance"""
    print(f"Executing command - Action: {action}, Distance: {distance} meter(s)")
    
    if action == "unknown":
        print("Command not recognized")
        #misty.speak("Sorry, I didn't understand that command")
        return
    
    # Calculate drive time (approximate: 0.5 meters per second at speed 50)
    drive_time_ms = int((distance / 0.5) * 1000)
    
    if action == 'forward':
        print(f"Moving forward {distance} meter(s)")
        #misty.speak(f"Moving forward {distance} meters")
        misty.drive_time(50, 0, drive_time_ms)
        
    elif action == 'backward':
        print(f"Moving backward {distance} meter(s)")
        #misty.speak(f"Moving backward {distance} meters")
        misty.drive_time(-50, 0, drive_time_ms)
        
    elif action == 'left':
        print(f"Going left {distance} meter(s)")
        #misty.speak(f"Going left {distance} meters")
        misty.drive_time(0, -50, 1000)
        time.sleep(1.5)
        misty.drive_time(50, 0, drive_time_ms)
        
    elif action == 'right':
        print(f"Going right {distance} meter(s)")
        #misty.speak(f"Going right {distance} meters")
        misty.drive_time(0, 50, 1000)
        time.sleep(1.5)
        misty.drive_time(50, 0, drive_time_ms)
        
    elif action == 'stop':
        print("Stopping")
        #misty.speak("Stopping")
        misty.stop()
        
    else:
        print("Command not recognized")
        #misty.speak("Sorry, I didn't understand that command")

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
            #misty.speak("Let me think about that")
            
            speech_text = transcribe_audio_from_misty(filename)
            
            if speech_text:
                print(f"Heard: {speech_text}")
                
                # Use LLM to parse the intent
                action, distance = parse_intent_with_llm(speech_text)
                
                # Execute the parsed command
                if action != "unknown":
                    process_voice_command(action, distance)
                else:
                    misty.speak("I'm not sure what you want me to do")
            else:
                print("Could not transcribe audio")
                misty.speak("Sorry, I couldn't hear you clearly")
                
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

def test_mode_loop():
    """Loop for testing with typed commands - robot still moves!"""
    print("\n" + "="*50)
    print("TEST MODE ACTIVE - Type commands instead of speaking")
    print("Robot WILL execute movements based on your typed commands")
    print("Using Ollama (local LLM) for parsing")
    print("Type commands to test (or 'quit' to exit)")
    print("="*50 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter command: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting test mode...")
                break
            
            if not user_input:
                continue
            
            print(f"\n--- Processing: '{user_input}' ---")
            
            # Parse intent with LLM
            action, distance = parse_intent_with_llm(user_input)
            
            # Execute command (robot will actually move!)
            process_voice_command(action, distance)
            
            print("--- Done ---\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting test mode...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    ip_address = "172.20.10.2"
    print("setting up misty...")
    misty = Robot(ip_address)

    print("misty setup sucessfully")
    misty.display_image("e_SleepingZZZ.jpg", 1)
    misty.move_head(60, 0, 0, 80)
    misty.move_arms(85, 85, 80, 80)
    misty.change_led(0, 0, 255)
    
    if TEST_MODE:
        print("entered test mode")
        # Test mode: Type commands, but robot still executes them
        test_mode_loop()
    else:
        # Normal mode: Voice commands
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
        print("Misty is ready with LLM-powered intent parsing!")
        print("Say 'Hey Misty' then give a command like:")
        print("  - 'move forward 1 meter'")
        print("  - 'go left 2 meters, wait no, go right 3 meters'")
        print("  - 'move backward... actually move forward 1 meter'")
        print("="*50 + "\n")
        
        misty.keep_alive()