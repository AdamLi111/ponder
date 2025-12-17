"""
MistyController module
Main controller coordinating all robot modules and interaction loops
Uses GPT-5 nano as unified VLM for both text and vision
Includes conversation logging
"""
from PythonSDKmain.mistyPy.Robot import Robot
from model.speech_handler import SpeechHandler
from model.llm_layer import LLMLayer
from model.llm_without_friction import LLMLayerNoFriction
from model.vision_handler import VisionHandler
from model.action_executor import ActionExecutor
from model.conversation_logger import ConversationLogger
import time


class MistyController:
    """
    Main controller for Misty robot that coordinates all modules
    and handles both voice-based and test mode interactions
    """
    
    def __init__(self, ip_address, openai_api_key, use_laptop_mic=False, friction_enabled=True, enable_logging=True, session_name=None):
        """
        Initialize the Misty controller with all necessary modules
        
        Args:
            ip_address: IP address of Misty robot
            openai_api_key: API key for OpenAI (GPT-5 nano)
            use_laptop_mic: If True, use laptop microphone instead of Misty's mic
            friction_enabled: If True, use LLM with positive friction; if False, use no-friction version
            enable_logging: If True, log all conversations to file
            session_name: Optional custom name for log file
        """
        self.ip_address = ip_address
        self.use_laptop_mic = use_laptop_mic
        self.friction_enabled = friction_enabled
        
        # Initialize conversation logger
        self.logger = None
        if enable_logging:
            self.logger = ConversationLogger(session_name=session_name)
        
        # Initialize robot FIRST
        self.robot = Robot(ip_address)
        
        # Initialize modules - PASS robot instance to speech_handler
        self.speech_handler = SpeechHandler(
            ip_address, 
            robot=self.robot,
            use_laptop_mic=use_laptop_mic
        )
        
        # Choose LLM layer based on friction setting and pass logger
        if friction_enabled:
            self.llm_layer = LLMLayer(openai_api_key, logger=self.logger)
            print("Using LLM with positive friction enabled")
        else:
            self.llm_layer = LLMLayerNoFriction(openai_api_key, logger=self.logger)
            print("Using LLM without friction (control mode)")
        
        self.vision_handler = VisionHandler(self.robot, ip_address)
        self.action_executor = ActionExecutor(
            self.robot, 
            self.vision_handler, 
            self.llm_layer,
            logger=self.logger
        )
        
        # State management for voice processing
        self.processing_audio = False
        self.last_processed_file = None
        
        mic_mode = "Laptop Microphone" if use_laptop_mic else "Misty's Microphone"
        friction_mode = "with friction" if friction_enabled else "without friction (control)"
        logging_status = "enabled" if enable_logging else "disabled"
        print(f"Misty controller initialized with GPT-5 nano VLM ({friction_mode}) + {mic_mode}")
        print(f"Conversation logging: {logging_status}")
    
    def setup_robot(self):
        """Initial robot setup - configure default state"""
        print("Setting up Misty...")
        self.robot.move_head(0, 0, 0)
        self.robot.display_image("e_Amazement.jpg")
        print("Setup complete")
    
    def _seems_like_spatial_command(self, speech_text):
        """
        Check if the text seems like it might be a spatial navigation command
        This is a simple heuristic check before calling the LLM
        """
        spatial_keywords = [
            'go to', 'go over to', 'move to', 'move towards', 
            'navigate to', 'walk to', 'approach', 'head to',
            'get to', 'move', 'turn'
        ]
        
        text_lower = speech_text.lower()
        return any(keyword in text_lower for keyword in spatial_keywords)
    
    def _seems_like_find_command(self, speech_text):
        """
        Check if the text seems like a find/search command
        Find commands handle their own image capture (360-degree scan)
        """
        find_keywords = [
            'find', 'find my', 'find the',
            'look for', 'search for',
            'where is', 'where is my', 'where is the',
            'locate', 'locate my', 'locate the'
        ]
        
        text_lower = speech_text.lower()
        return any(keyword in text_lower for keyword in find_keywords)
    
    def _process_command(self, speech_text):
        """
        Common command processing logic for both voice and typed input
        
        Args:
            speech_text: The transcribed or typed command text
        """
        print(f"Heard: {speech_text}")
        
        # Start logging turn
        if self.logger:
            self.logger.start_turn(speech_text)
        
        self.robot.speak("OK, let me think about that")
        if self.logger:
            self.logger.log_misty_speech("OK, let me think about that")
        
        # Check if this is a find command - these handle their own image capture
        if self._seems_like_find_command(speech_text):
            print("Find command detected - will do 360-degree scan later")
            try:
                intent = self.llm_layer.parse_intent_with_vision(speech_text, image_data_base64=None)
            except Exception as llm_error:
                print(f"LLM error: {llm_error}")
                import traceback
                traceback.print_exc()
                print("[DEBUG] About to speak: Sorry, I had trouble understanding that")
                self.robot.speak("Sorry, I had trouble understanding that")
                if self.logger:
                    self.logger.log_misty_speech("Sorry, I had trouble understanding that")
                    self.logger.end_turn()
                return
        
        # Check if this seems like a spatial command using simple keyword matching
        elif self._seems_like_spatial_command(speech_text):
            print("Spatial command detected (keyword match) - capturing image for vision analysis")
            
            # Capture image FIRST
            image_data = self.vision_handler.capture_and_encode()
            
            if image_data:
                # Parse with vision included
                try:
                    intent = self.llm_layer.parse_intent_with_vision(speech_text, image_data_base64=image_data)
                except Exception as llm_error:
                    print(f"LLM error: {llm_error}")
                    import traceback
                    traceback.print_exc()
                    print("[DEBUG] About to speak: Sorry, I had trouble understanding that")
                    self.robot.speak("Sorry, I had trouble understanding that")
                    if self.logger:
                        self.logger.log_misty_speech("Sorry, I had trouble understanding that")
                        self.logger.end_turn()
                    return
            else:
                print("Failed to capture image, proceeding without vision")
                try:
                    intent = self.llm_layer.parse_intent_with_vision(speech_text, image_data_base64=None)
                except Exception as llm_error:
                    print(f"LLM error: {llm_error}")
                    import traceback
                    traceback.print_exc()
                    print("[DEBUG] About to speak: Sorry, I had trouble understanding that")
                    self.robot.speak("Sorry, I had trouble understanding that")
                    if self.logger:
                        self.logger.log_misty_speech("Sorry, I had trouble understanding that")
                        self.logger.end_turn()
                    return
        else:
            # Not a spatial or find command - parse without vision
            try:
                intent = self.llm_layer.parse_intent_with_vision(speech_text, image_data_base64=None)
            except Exception as llm_error:
                print(f"LLM error: {llm_error}")
                import traceback
                traceback.print_exc()
                print("[DEBUG] About to speak: Sorry, I had trouble understanding that")
                self.robot.speak("Sorry, I had trouble understanding that")
                if self.logger:
                    self.logger.log_misty_speech("Sorry, I had trouble understanding that")
                    self.logger.end_turn()
                return
        
        # Execute the action
        if intent.get('action') != "unknown" or 'actions' in intent:
            self.action_executor.execute(intent)
        else:
            print("[DEBUG] About to speak: I'm not sure what you want me to do")
            self.robot.speak("I'm not sure what you want me to do")
            if self.logger:
                self.logger.log_misty_speech("I'm not sure what you want me to do")
        
        # End logging turn
        if self.logger:
            self.logger.end_turn()
    
    def _voice_record_callback(self, data):
        """
        Internal callback for voice recording events from Misty's microphone
        Handles audio transcription and command processing
        """
        print("Voice_Record_Callback triggered!")
        print(f"Event data: {data}")
        
        # Check if we're already processing
        if self.processing_audio:
            print("Already processing audio, ignoring this callback")
            return
        
        try:
            # Set processing flag
            self.processing_audio = True
            
            # Get the audio file path from VoiceRecord event
            audio_file = data['message']['filename']
            print(f"Audio filename from VoiceRecord event: {audio_file}")
            
            # Skip if we just processed this file
            if audio_file == self.last_processed_file:
                print(f"Already processed {audio_file}, skipping")
                return
            
            self.last_processed_file = audio_file
            
            # Small delay to ensure file is ready on Misty's filesystem
            print("Waiting for file to be ready...")
            time.sleep(0.3)
            
            # Transcribe the audio
            text = self.speech_handler.transcribe_audio_from_misty(audio_file)
            
            if text:
                # Process the command
                self._process_command(text)
            else:
                print("No text transcribed - file may be empty or corrupted")
                self.robot.speak("Sorry, I didn't catch that. Please try again.")
            
            # CRITICAL: Restart listening for the next command
            # This allows multi-turn conversation
            print("\n" + "="*50)
            print("Command processed. Ready for next command...")
            print("Say 'Hey Misty' to give another command")
            print("="*50 + "\n")
            
            if not self.use_laptop_mic:
                # Visual feedback - change LED to indicate ready for next command
                self.robot.change_led(0, 255, 0)  # Green = ready
                
                # Restart speech capture for Misty's mic
                self.speech_handler.restart_listening()
                
        except Exception as e:
            print(f"Error in voice callback: {e}")
            import traceback
            traceback.print_exc()
            
            # Still try to restart listening even if there was an error
            if not self.use_laptop_mic:
                try:
                    self.robot.change_led(255, 0, 0)  # Red = error
                    self.speech_handler.restart_listening()
                except:
                    pass
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
            friction_status = "ENABLED" if self.friction_enabled else "DISABLED (control)"
            print(f"Friction: {friction_status}")
            print("Type commands to test (or 'quit' to exit)")
            print("Examples:")
            print("  - move forward 2 meters")
            print("  - turn left")
            print("  - what do you see")
            print("  - go to the plant")
            print("  - move towards that chair")
            print("  - go forward 1 meter and then turn right")
            print("  - find my bag")
            print("  - where is my laptop")
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
            
            if self.use_laptop_mic:
                print("Using laptop microphone")
                print("Say 'Hey Misty' followed by your command")
                print("Press Ctrl+C to stop")
                
                # Start continuous listening with laptop mic
                self.speech_handler.listen_continuously(self._laptop_mic_callback)
                
                # Keep running
                try:
                    while True:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    pass
            else:
                print("Using Misty's built-in microphone")
                print("Say 'Hey Misty' followed by your command")
                print("Press Ctrl+C to stop")
                
                # Start listening for wake phrase from Misty
                self.speech_handler.start_listening(self._voice_record_callback)
                
                # Keep running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
    
    def _laptop_mic_callback(self, command_text):
        """
        Callback for laptop microphone input
        Receives transcribed command text directly
        """
        print(f"\n[Laptop Mic] Received command: {command_text}")
        
        # Check if we're already processing
        if self.processing_audio:
            print("Already processing audio, ignoring this command")
            return
        
        try:
            # Set processing flag
            self.processing_audio = True
            
            # Process the command directly
            self._process_command(command_text)
                
        except Exception as e:
            print(f"Error in laptop mic callback: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always clear processing flag
            self.processing_audio = False
    
    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")
        if hasattr(self, 'speech_handler'):
            self.speech_handler.stop_listening()
        if self.logger:
            self.logger.finalize()
        print("Cleanup complete")