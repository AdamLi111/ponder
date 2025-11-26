"""
Action executor module
Defines and executes all robot actions with positive friction support and spatial navigation
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
            'spatial_navigate': self.spatial_navigate,
            'unknown': self.unknown_action
        }
    
    def execute(self, intent):
        """Execute an action based on parsed intent with friction handling"""
        # Handle multi-action sequences
        if 'actions' in intent:
            print(f"Executing multi-action sequence")
            for action_item in intent['actions']:
                action = action_item.get('action', 'unknown')
                distance = action_item.get('distance', 1.0)
                
                action_func = self.actions.get(action, self.unknown_action)
                action_func(distance=distance)
                
                # Small delay between actions
                time.sleep(0.5)
            return
        
        # Single action handling
        friction_type = intent.get('friction_type', 'none')
        action = intent.get('action', 'unknown')
        distance = intent.get('distance', 1.0)
        text = intent.get('text', '')
        target_object = intent.get('target_object', None)
        
        print(f"Executing - Friction: {friction_type}, Action: {action}")
        
        # Handle friction types
        if friction_type != 'none' and text:
            print(f"Applying {friction_type} friction: {text}")
            self.robot.speak(text)
            
            # Probing stops execution to wait for user response
            if friction_type == 'probing' or action == 'clarify':
                return
            
            # Reflective pause with delay before continuing
            if friction_type == 'reflective_pause':
                time.sleep(1.5)
            
            # For other friction types (assumption_reveal, overspecification, reinforcement),
            # we announce and then continue with the action
            time.sleep(0.5)
        
        # Get the action function and execute it
        action_func = self.actions.get(action, self.unknown_action)
        if action == 'spatial_navigate':
            action_func(target_object=target_object)
        else:
            action_func(distance=distance, text=text)
    
    def spatial_navigate(self, target_object=None, **kwargs):
        """Navigate to a target object using vision-based spatial reasoning"""
        if not self.vision_handler or not self.llm_layer:
            print("Vision or LLM handler not available")
            self.robot.speak("Sorry, my vision system is not available")
            return
        
        if not target_object:
            print("No target object specified")
            self.robot.speak("I need to know what object to navigate towards")
            return
        
        print(f"Starting spatial navigation towards: {target_object}")
        self.robot.speak(f"Let me find the {target_object}")
        
        # Capture and encode image
        image_data = self.vision_handler.capture_and_encode()
        
        if not image_data:
            self.robot.speak("Sorry, I couldn't capture an image")
            return
        
        # Analyze the scene with GPT-5 nano
        analysis = self.llm_layer.analyze_spatial_scene(image_data, target_object)
        
        if not analysis:
            self.robot.speak("Sorry, I had trouble analyzing the scene")
            return
        
        # Handle the results
        if not analysis['found']:
            # Object not visible
            print(f"Object not found: {analysis['reasoning']}")
            self.robot.speak(f"I don't see the {target_object} in front of me. Could you let me know which direction it is?")
            return
        
        # Object found - execute navigation
        direction = analysis['direction']
        turn_degrees = analysis.get('turn_degrees', 0)
        distance = analysis['distance']
        confidence = analysis['confidence']
        
        print(f"Navigation plan - Direction: {direction}, Turn: {turn_degrees}°, Distance: {distance}m, Confidence: {confidence}")
        
        # Announce the plan
        if direction == "straight":
            self.robot.speak(f"I see the {target_object} straight ahead, about {distance:.1f} meters away. Moving towards it.")
        elif direction == "left":
            self.robot.speak(f"I see the {target_object} to my left. Turning and then moving towards it.")
        elif direction == "right":
            self.robot.speak(f"I see the {target_object} to my right. Turning and then moving towards it.")
        
        time.sleep(1)
        
        # Execute the navigation
        if direction != "straight":
            # Turn towards the object first
            turn_time = self._calculate_turn_time(abs(turn_degrees))
            if direction == "left":
                print(f"Turning left {abs(turn_degrees)} degrees")
                self.robot.drive_time(0, -100, turn_time)
            else:  # right
                print(f"Turning right {turn_degrees} degrees")
                self.robot.drive_time(0, 100, turn_time)
            
            time.sleep(turn_time / 1000 + 0.5)
        
        # Move forward towards the object
        # Reduce distance slightly to avoid collision
        safe_distance = max(0.3, distance - 0.5)
        drive_time = self._calculate_drive_time(safe_distance)
        print(f"Moving forward {safe_distance:.1f} meters")
        self.robot.drive_time(50, 0, drive_time)
        
        time.sleep(drive_time / 1000 + 0.5)
        
        self.robot.speak(f"I've reached the {target_object}")
    
    def move_forward(self, distance=1.0, **kwargs):
        """Move forward by specified distance"""
        print(f"Moving forward {distance} meter(s)")
        self.robot.speak(f"Moving forward {distance} meters")
        drive_time = self._calculate_drive_time(distance)
        self.robot.drive_time(50, 0, drive_time)
    
    def move_backward(self, distance=1.0, **kwargs):
        """Move backward by specified distance"""
        print(f"Moving backward {distance} meter(s)")
        self.robot.speak(f"Moving backward {distance} meters")
        drive_time = self._calculate_drive_time(distance)
        self.robot.drive_time(-50, 0, drive_time)
    
    def move_left(self, distance=1.0, **kwargs):
        """Strafe left by specified distance"""
        print(f"Going left {distance} meter(s)")
        self.robot.speak(f"Going left {distance} meters")
        # Turn left, move forward, turn back right
        self.robot.drive_time(0, -100, 2150)  # ~45 degrees
        time.sleep(2.5)
        drive_time = self._calculate_drive_time(distance)
        self.robot.drive_time(50, 0, drive_time)
        time.sleep(drive_time / 1000 + 0.5)
        self.robot.drive_time(0, 100, 2150)  # Turn back
    
    def move_right(self, distance=1.0, **kwargs):
        """Strafe right by specified distance"""
        print(f"Going right {distance} meter(s)")
        self.robot.speak(f"Going right {distance} meters")
        # Turn right, move forward, turn back left
        self.robot.drive_time(0, 100, 2150)  # ~45 degrees
        time.sleep(2.5)
        drive_time = self._calculate_drive_time(distance)
        self.robot.drive_time(50, 0, drive_time)
        time.sleep(drive_time / 1000 + 0.5)
        self.robot.drive_time(0, -100, 2150)  # Turn back
    
    def turn_left(self, distance=1.0, **kwargs):
        """Turn left 90 degrees"""
        print("Turning left 90 degrees")
        self.robot.speak("Turning left")
        self.robot.drive_time(0, -100, 4300)
    
    def turn_right(self, distance=1.0, **kwargs):
        """Turn right 90 degrees"""
        print("Turning right 90 degrees")
        self.robot.speak("Turning right")
        self.robot.drive_time(0, 100, 4300)
    
    def stop(self, **kwargs):
        """Stop all movement"""
        print("Stopping")
        self.robot.speak("Stopping")
        self.robot.stop()
    
    def speak(self, text='', **kwargs):
        """Speak the provided text"""
        if text:
            print(f"Speaking: {text}")
            self.robot.speak(text)
    
    def clarify(self, text='', **kwargs):
        """Request clarification from user"""
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