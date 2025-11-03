"""
Main control loop for Misty robot
Handles voice interaction loop and coordinates all modules
"""
from PythonSDKmain.mistyPy.Robot import Robot
from PythonSDKmain.mistyPy.Events import Events
import time

from model import SpeechHandler, LLMLayer, VisionHandler, ActionExecutor


class MistyController:
    def __init__(self, ip_address, gemini_api_key):
        self.ip_address = ip_address
        
        # Initialize robot
        self.robot = Robot(ip_address)
        
        # Initialize modules
        self.speech_handler = SpeechHandler(ip_address)
        self.llm_layer = LLMLayer(gemini_api_key)
        self.vision_handler = VisionHandler(self.robot, ip_address)
        self.action_executor = ActionExecutor(
            self.robot, 
            self.vision_handler, 
            self.llm_layer
        )
        
        print("Misty controller initialized")
    
    def setup_robot(self):
        """Initial robot setup"""
        print("Setting up Misty...")
        self.robot.display_image("e_SleepingZZZ.jpg", 1)
        self.robot.move_head(60, 0, 0, 80)
        self.robot.move_arms(85, 85, 80, 80)
        self.robot.change_led(0, 0, 255)
        print("Setup complete")
    
    def voice_record_callback(self, data):
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
                
                # Transcribe the audio
                print("Transcribing audio...")
                self.robot.speak("Let me think about that")
                
                speech_text = self.speech_handler.transcribe_audio_from_misty(filename)
                
                if speech_text:
                    print(f"Heard: {speech_text}")
                    
                    # Parse intent with LLM
                    intent = self.llm_layer.parse_intent(speech_text)
                    
                    # Execute the action
                    if intent['action'] != "unknown":
                        self.action_executor.execute(intent)
                    else:
                        self.robot.speak("I'm not sure what you want me to do")
                else:
                    print("Could not transcribe audio")
                    self.robot.speak("Sorry, I couldn't hear you clearly")
                    
        except Exception as e:
            print(f"Error processing voice: {e}")
            import traceback
            traceback.print_exc()
    
    def key_phrase_callback(self, data):
        """Callback when key phrase is recognized"""
        print("="*50)
        print("Key phrase detected! Listening for command...")
        print("="*50)
        
        self.robot.display_image("e_Surprise.jpg", 1)
        self.robot.change_led(0, 255, 0)  # Green LED to show listening
        
        # Start speech capture
        print("Starting speech capture...")
        try:
            result = self.robot.capture_speech()
            print(f"Capture speech result: {result}")
        except Exception as e:
            print(f"Error starting speech capture: {e}")
        
        time.sleep(5)
        self.robot.change_led(0, 0, 255)
    
    def start_voice_mode(self):
        """Start voice-controlled interaction mode"""
        print("Starting key phrase recognition...")
        self.robot.start_key_phrase_recognition()
        
        # Register events
        print("Registering Key Phrase event...")
        self.robot.register_event(
            event_name='Key_Phrase_Recognized',
            event_type=Events.KeyPhraseRecognized,
            callback_function=self.key_phrase_callback,
            keep_alive=True
        )
        
        print("Registering VoiceRecord event...")
        self.robot.register_event(
            event_name='VoiceRecord',
            event_type=Events.VoiceRecord,
            callback_function=self.voice_record_callback,
            keep_alive=True
        )
        
        print("\n" + "="*50)
        print("Misty is ready with LLM-powered intent parsing and vision!")
        print("Say 'Hey Misty' then give a command like:")
        print("  - 'move forward 1 meter'")
        print("  - 'what do you see?'")
        print("  - 'say hello to everyone'")
        print("="*50 + "\n")
        
        self.robot.keep_alive()
    
    def test_mode_loop(self):
        """Interactive testing mode with typed commands"""
        print("\n" + "="*50)
        print("TEST MODE ACTIVE - Type commands instead of speaking")
        print("Robot WILL execute movements based on your typed commands")
        print("Using Google Gemini (free) for parsing and vision")
        print("Type commands to test (or 'quit' to exit)")
        print("="*50)
        print("\nAvailable commands:")
        print("  - Movement: 'move forward 1 meter', 'go left', 'stop'")
        print("  - Vision: 'what do you see', 'describe what's in front of you'")
        print("  - Speech: 'say hello', 'tell me a joke'")
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
                intent = self.llm_layer.parse_intent(user_input)
                
                # Execute action
                self.action_executor.execute(intent)
                
                print("--- Done ---\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting test mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    def run(self):
        """Main execution method"""
        self.setup_robot()
        self.start_voice_mode()
    
    def cleanup(self):
        """Clean up resources"""
        self.speech_handler.cleanup()
        self.vision_handler.cleanup()


if __name__ == "__main__":
    # Configuration
    IP_ADDRESS = "172.20.10.2"
    GEMINI_API_KEY = "AIzaSyDmIcNnlfYB39w4W2d76hq3C1nGI9sdmK0"
    
    # Create and run controller
    controller = MistyController(
        ip_address=IP_ADDRESS,
        gemini_api_key=GEMINI_API_KEY
    )
    
    try:
        controller.run()
    finally:
        controller.cleanup()