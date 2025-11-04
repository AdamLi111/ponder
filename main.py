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
    
    def _voice_record_callback(self, data):
        """Internal callback wrapper for voice recording events"""
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
                
                # Reset LED and wait for next command
                print("\nReady for next command. Say 'Hey Misty' to continue...")
                self.robot.change_led(0, 0, 255)  # Blue LED - ready state
                    
        except Exception as e:
            print(f"Error processing voice: {e}")
            import traceback
            traceback.print_exc()
            # Make sure we're ready for next command even if there's an error
            self.robot.change_led(0, 0, 255)
    
    def _key_phrase_callback(self, data):
        """Internal callback wrapper when key phrase is recognized"""
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
        
        # Create wrapper functions that only take data parameter
        def key_phrase_wrapper(data):
            self._key_phrase_callback(data)
        
        def voice_record_wrapper(data):
            self._voice_record_callback(data)
        
        # Register events with wrapper functions
        print("Registering Key Phrase event...")
        self.robot.register_event(
            event_name='Key_Phrase_Recognized',
            event_type=Events.KeyPhraseRecognized,
            callback_function=key_phrase_wrapper,
            keep_alive=True
        )
        
        print("Registering VoiceRecord event...")
        self.robot.register_event(
            event_name='VoiceRecord',
            event_type=Events.VoiceRecord,
            callback_function=voice_record_wrapper,
            keep_alive=True
        )
        
        print("\n" + "="*50)
        print("Misty is ready with LLM-powered intent parsing and vision!")
        print("Say 'Hey Misty' then give a command like:")
        print("  - 'move forward 1 meter'")
        print("  - 'what do you see?'")
        print("  - 'say hello to everyone'")
        print("="*50 + "\n")
        
        # Keep the event loop running
        try:
            self.robot.keep_alive()
        except KeyboardInterrupt:
            print("\nShutting down Misty...")
            self.robot.unregister_all_events()
            print("Goodbye!")
    
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