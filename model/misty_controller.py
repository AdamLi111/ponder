"""
MistyController module
Main controller coordinating all robot modules and interaction loops
"""
from PythonSDKmain.mistyPy.Robot import Robot
from PythonSDKmain.mistyPy.Events import Events
import time

from . import SpeechHandler, LLMLayer, VisionHandler, ActionExecutor


class MistyController:
    """
    Main controller for Misty robot that coordinates all modules
    and handles both voice-based and test mode interactions
    """
    
    def __init__(self, ip_address, groq_api_key, claude_api_key=None):
        """
        Initialize the Misty controller with all necessary modules
        
        Args:
            ip_address: IP address of Misty robot
            groq_api_key: API key for Groq LLM service
            claude_api_key: Optional API key for Claude vision service
        """
        self.ip_address = ip_address
        
        # Initialize robot
        self.robot = Robot(ip_address)
        
        # Initialize modules
        self.speech_handler = SpeechHandler(ip_address)
        self.llm_layer = LLMLayer(groq_api_key, claude_api_key)
        self.vision_handler = VisionHandler(self.robot, ip_address)
        self.action_executor = ActionExecutor(
            self.robot, 
            self.vision_handler, 
            self.llm_layer
        )
        
        # State management for voice processing
        self.processing_audio = False
        self.last_processed_file = None
        
        print("Misty controller initialized with Groq LLM + Claude Vision")
    
    def setup_robot(self):
        """Initial robot setup - configure default state"""
        print("Setting up Misty...")
        self.robot.display_image("e_SleepingZZZ.jpg", 1)
        # Uncomment if you want initial positioning:
        # self.robot.move_head(60, 0, 0, 80)
        # self.robot.move_arms(85, 85, 80, 80)
        # self.robot.change_led(0, 0, 255)
        print("Setup complete")
    
    def _process_command(self, speech_text):
        """
        Common command processing logic for both voice and typed input
        
        Args:
            speech_text: The transcribed or typed command text
        """
        print(f"Heard: {speech_text}")
        
        # Parse intent with LLM
        try:
            intent = self.llm_layer.parse_intent(speech_text)
            
            # Execute the action
            if intent['action'] != "unknown":
                self.action_executor.execute(intent)
            else:
                print("[DEBUG] About to speak: I'm not sure what you want me to do")
                self.robot.speak("I'm not sure what you want me to do")
        except Exception as llm_error:
            print(f"LLM error: {llm_error}")
            print("[DEBUG] About to speak: Sorry, I had trouble understanding that")
            self.robot.speak("Sorry, I had trouble understanding that")
    
    def _voice_record_callback(self, data):
        """
        Internal callback for voice recording events
        Handles audio transcription and command processing
        """
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
                
                # Prevent duplicate processing
                if filename == self.last_processed_file:
                    print(f"Already processed this file: {filename}")
                    return
                
                if self.processing_audio:
                    print(f"Already processing another audio file")
                    return
                
                # Set lock and mark file as being processed
                self.processing_audio = True
                self.last_processed_file = filename
                
                print(f"[DEBUG] About to transcribe. Lock set.")
                
                # Transcribe the audio
                print("Transcribing audio...")
                speech_text = self.speech_handler.transcribe_audio_from_misty(filename)
                
                print(f"[DEBUG] Transcription result: {speech_text}")
                
                if speech_text:
                    self._process_command(speech_text)
                else:
                    print("Could not transcribe audio")
                    print("[DEBUG] About to speak: Sorry, I couldn't hear you clearly")
                    self.robot.speak("Sorry, I couldn't hear you clearly")
                
                print("\nReady for next command. Say 'Hey Misty' to continue...")
                self.robot.change_led(0, 0, 255)
                
        except Exception as e:
            print(f"Error processing voice: {e}")
            import traceback
            traceback.print_exc()
            self.robot.change_led(0, 0, 255)
        finally:
            # Always release the lock
            print(f"[DEBUG] Releasing lock")
            self.processing_audio = False
    
    def _key_phrase_callback(self, data):
        """
        Internal callback when wake phrase ('Hey Misty') is recognized
        Initiates speech capture for command
        """
        print("="*50)
        print("Key phrase detected! Listening for command...")
        print("="*50)
        
        self.robot.display_image("e_Surprise.jpg", 1)
        self.robot.change_led(0, 255, 0)  # Green LED to show listening
        
        # Start speech capture with longer timeout
        print("Starting speech capture...")
        try:
            # Give user more time to speak before detecting silence
            result = self.robot.capture_speech(
                overwriteExisting=True,
                silenceTimeout=5000,  # 5 seconds of silence
                maxSpeechLength=10000  # Max 10 seconds of speech
            )
            print(f"Capture speech result: {result}")
        except Exception as e:
            print(f"Error starting speech capture: {e}")
            self.robot.change_led(0, 0, 255)  # Reset to blue on error
    
    def start_voice_mode(self):
        """
        Start voice-controlled interaction mode
        Registers event handlers and begins listening for wake phrase
        """
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
        print("Misty is ready with Groq-powered intent parsing and Claude vision!")
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
    
    def start_test_mode(self):
        """
        Test mode - type commands in terminal instead of using voice
        Useful for debugging and development
        """
        print("\n" + "="*50)
        print("TEST MODE: Type commands directly")
        print("="*50)
        print("Available commands:")
        print("  - 'move forward 2 meters'")
        print("  - 'move backward 1 meter'")
        print("  - 'turn left'")
        print("  - 'turn right'")
        print("  - 'what do you see?'")
        print("  - 'say hello everyone'")
        print("  - 'stop'")
        print("  - Type 'quit' to exit")
        print("="*50 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("Enter command: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Exiting...")
                    break
                
                if not user_input:
                    continue
                
                # Process the command using the same logic as voice
                self._process_command(user_input)
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    def run(self, test_mode=False):
        """
        Main execution method
        
        Args:
            test_mode: If True, run in terminal test mode. If False, use voice mode.
        """
        self.setup_robot()
        
        if test_mode:
            self.start_test_mode()
        else:
            self.start_voice_mode()
    
    def cleanup(self):
        """Clean up resources"""
        self.speech_handler.cleanup()
        self.vision_handler.cleanup()