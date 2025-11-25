"""
Action executor module
Defines and executes all robot actions with positive friction support
"""
import time


class ActionExecutor:
    def __init__(self, robot, vision_handler=None, llm_layer=None):
        self.robot = robot
        self.vision_handler = vision_handler
        self.llm_layer = llm_layer
        
        # Action mapping
        self.actions = {
            'forward': self.move_forward,
            'backward': self.move_backward,
            'left': self.move_left,
            'right': self.move_right,
            'turn_left': self.turn_left,
            'turn_right': self.turn_right,
            'stop': self.stop,
            'speak': self.speak,
            'clarify': self.clarify,
            'describe_vision': self.describe_vision,
            'unknown': self.unknown_action
        }
    
    def execute(self, intent_or_list):
        """
        Execute a single action or a sequence of actions
        
        Args:
            intent_or_list: Either a single intent dict or a list of intent dicts
        """
        # Handle both single action and list of actions
        if isinstance(intent_or_list, list):
            self.execute_sequence(intent_or_list)
        else:
            self.execute_single(intent_or_list)
    
    def execute_sequence(self, action_list):
        """Execute a sequence of actions"""
        print(f"\n{'='*50}")
        print(f"Executing sequence of {len(action_list)} action(s)")
        print(f"{'='*50}")
        
        for i, intent in enumerate(action_list):
            print(f"\n>>> Action {i+1}/{len(action_list)} <<<")
            self.execute_single(intent)
            
            # Small delay between actions for stability
            # Skip delay after the last action
            if i < len(action_list) - 1:
                print("Pausing briefly before next action...")
                time.sleep(0.5)
        
        print(f"\n{'='*50}")
        print(f"Sequence complete!")
        print(f"{'='*50}\n")
    
    def execute_single(self, intent):
        """Execute a single action based on parsed intent with friction handling"""
        friction_type = intent.get('friction_type', 'none')
        action = intent.get('action', 'unknown')
        distance = intent.get('distance', 1.0)
        text = intent.get('text', '')
        
        print(f"Executing - Friction: {friction_type}, Action: {action}")
        
        # Handle friction first if present
        if friction_type != 'none' and text:
            self.robot.speak(text)
            # If action is clarify, we stop here and wait for user response
            if action == 'clarify':
                return
        
        # Get the action function and execute it
        action_func = self.actions.get(action, self.unknown_action)
        action_func(distance=distance, text=text)
    
    def move_forward(self, distance, **kwargs):
        """Move robot forward"""
        drive_time_ms = self._calculate_drive_time(distance)
        print(f"Moving forward {distance} meter(s)")
        self.robot.speak(f"Moving forward {distance} meters")
        self.robot.drive_time(50, 0, drive_time_ms)
        time.sleep(2)
    
    def move_backward(self, distance, **kwargs):
        """Move robot backward"""
        drive_time_ms = self._calculate_drive_time(distance)
        print(f"Moving backward {distance} meter(s)")
        self.robot.speak(f"Moving backward {distance} meters")
        self.robot.drive_time(-50, 0, drive_time_ms)
        time.sleep(2)
    
    def move_left(self, distance, **kwargs):
        """Turn left 90 degrees then move forward"""
        print(f"Turning left and moving {distance} meter(s)")
        self.robot.speak(f"Turning left and moving {distance} meters")
        
        # Turn left 90 degrees
        self.robot.drive_time(0, 100, 4300)
        time.sleep(4.5)  # Wait for turn to complete
        
        # Move forward
        drive_time_ms = self._calculate_drive_time(distance)
        self.robot.drive_time(50, 0, drive_time_ms)
    
    def move_right(self, distance, **kwargs):
        """Turn right 90 degrees then move forward"""
        print(f"Turning right and moving {distance} meter(s)")
        self.robot.speak(f"Turning right and moving {distance} meters")
        
        # Turn right 90 degrees
        self.robot.drive_time(0, -100, 4300)
        time.sleep(4.5)  # Wait for turn to complete
        
        # Move forward
        drive_time_ms = self._calculate_drive_time(distance)
        self.robot.drive_time(50, 0, drive_time_ms)
    
    def turn_left(self, distance, **kwargs):
        """Turn left by specified degrees (distance parameter represents degrees)"""
        degrees = distance if distance != 1.0 else 90  # Default to 90 if not specified
        turn_time_ms = self._calculate_turn_time(degrees)
        print(f"Turning left {degrees} degrees")
        self.robot.speak(f"Turning left {degrees} degrees")
        self.robot.drive_time(0, 100, turn_time_ms)
        time.sleep(turn_time_ms / 1000 + 0.2)  # Wait for turn to complete with small buffer
    
    def turn_right(self, distance, **kwargs):
        """Turn right by specified degrees (distance parameter represents degrees)"""
        degrees = distance if distance != 1.0 else 90  # Default to 90 if not specified
        turn_time_ms = self._calculate_turn_time(degrees)
        print(f"Turning right {degrees} degrees")
        self.robot.speak(f"Turning right {degrees} degrees")
        self.robot.drive_time(0, -100, turn_time_ms)
        time.sleep(turn_time_ms / 1000 + 0.2)  # Wait for turn to complete with small buffer
    
    def stop(self, **kwargs):
        """Stop robot movement"""
        print("Stopping")
        self.robot.speak("Stopping")
        self.robot.stop()
    
    def speak(self, text, **kwargs):
        """Make robot speak the provided text"""
        if not text:
            text = "I don't know what to say"
        print(f"Speaking: {text}")
        self.robot.speak(text)
    
    def clarify(self, text, **kwargs):
        """Ask clarifying question - same as speak but semantically different"""
        if not text:
            text = "Could you please clarify what you'd like me to do?"
        print(f"Requesting clarification: {text}")
        self.robot.speak(text)
    
    def describe_vision(self, **kwargs):
        """Capture image and describe what robot sees"""
        if not self.vision_handler or not self.llm_layer:
            print("Vision or LLM handler not available")
            self.robot.speak("Sorry, my vision system is not available")
            return
        
        print("Processing vision request...")
        self.robot.speak("Let me take a look")
        
        # Capture and encode image
        image_data = self.vision_handler.capture_and_encode()
        
        if image_data:
            # Get description from LLM
            description = self.llm_layer.describe_image(image_data)
            
            if description:
                print(f"Vision description: {description}")
                self.robot.speak(description)
            else:
                self.robot.speak("Sorry, I couldn't analyze the image")
        else:
            self.robot.speak("Sorry, I couldn't capture an image")
    
    def unknown_action(self, **kwargs):
        """Handle unknown or unrecognized actions"""
        print("Command not recognized")
        self.robot.speak("Sorry, I didn't understand that command")
    
    def _calculate_drive_time(self, distance):
        """Calculate drive time in milliseconds based on distance
        
        Approximate: 0.5 meters per second at speed 50
        """
        return int((distance / 0.5) * 1000)
    
    def _calculate_turn_time(self, degrees):
        """Calculate turn time in milliseconds based on degrees
        
        Calibration: 4300ms at angular speed 100 = 90 degrees
        Therefore: time = (degrees / 90) * 4300
        """
        return int((degrees / 90.0) * 4300)