"""
MistyController module
Main controller coordinating all robot modules and interaction loops
Uses Groq for text parsing and OpenAI GPT-5 nano for vision
"""
from PythonSDKmain.mistyPy.Robot import Robot
from model.speech_handler import SpeechHandler
from model.llm_layer import LLMLayer
from model.vision_handler import VisionHandler
from model.action_executor import ActionExecutor
import time


class MistyController:
    """
    Main controller for Misty robot that coordinates all modules
    and handles both voice-based and test mode interactions
    """
    
    def __init__(self, ip_address, groq_api_key, openai_api_key=None):
        """
        Initialize the Misty controller with all necessary modules
        
        Args:
            ip_address: IP address of Misty robot
            groq_api_key: API key for Groq LLM service (text/intent parsing)
            openai_api_key: Optional API key for OpenAI (GPT-5 nano vision)
        """
        self.ip_address = ip_address
        
        # Initialize robot
        self.robot = Robot(ip_address)
        
        # Initialize modules
        self.speech_handler = SpeechHandler(ip_address)
        self.llm_layer = LLMLayer(groq_api_key, openai_api_key)  # Now takes openai_api_key
        self.vision_handler = VisionHandler(self.robot, ip_address)
        self.action_executor = ActionExecutor(
            self.robot, 
            self.vision_handler, 
            self.llm_layer
        )
        
        # State management for voice processing
        self.processing_audio = False
        self.last_processed_file = None
        
        print("Misty controller initialized with Groq LLM + GPT-5 nano Vision")
    
    def setup_robot(self):
        """Initial robot setup - configure default state"""
        print("Setting up Misty...")
        self.robot.display_image("e_SleepingZZZ.jpg", 1)
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
            if intent.get('action') != "unknown" or 'actions' in intent:
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
        
        # Check if we're already processing
        if self.processing_audio:
            print("Already processing audio, ignoring this callback")
            return
        
        try:
            # Set processing flag
            self.processing_audio = True
            
            # Get the audio file path
            audio_file = data['message']['filename']
            
            # Skip if we just processed this file
            if audio_file == self.last_processed_file:
                print(f"Already processed {audio_file}, skipping")
                return
            
            self.last_processed_file = audio_file
            
            # Transcribe the audio
            text = self.speech_handler.transcribe_audio(audio_file)
            
            if text:
                # Process the command
                self._process_command(text)
            else:
                print("No text transcribed")
                
        except Exception as e:
            print(f"Error in voice callback: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always clear processing flag
            self.processing_audio = False
    
    def run(self, test_mode=False):
        """
        Run the controller in either voice mode or test mode
        
        Args:
            test_mode: If True, accepts typed commands instead of voice
        """
        self.setup_robot()
        
        if test_mode:
            print("\n=== TEST MODE ===")
            print("Type commands to test (or 'quit' to exit)")
            print("Examples:")
            print("  - move forward 2 meters")
            print("  - turn left")
            print("  - what do you see")
            print("  - go to the plant")
            print("  - move towards that chair")
            print("  - go forward 1 meter and then turn right")
            print()
            
            while True:
                try:
                    command = input("Command: ").strip()
                    if command.lower() in ['quit', 'exit', 'q']:
                        break
                    if command:
                        self._process_command(command)
                except KeyboardInterrupt:
                    break
        else:
            print("\n=== VOICE MODE ===")
            print("Say 'Hey Misty' followed by your command")
            print("Press Ctrl+C to stop")
            
            # Start listening for wake phrase
            self.speech_handler.start_listening(self._voice_record_callback)
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    
    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")
        if hasattr(self, 'speech_handler'):
            self.speech_handler.stop_listening()
        print("Cleanup complete")