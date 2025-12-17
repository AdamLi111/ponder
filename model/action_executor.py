"""
Action executor module
Defines and executes all robot actions with positive friction support and spatial navigation
Now simplified since vision analysis happens before action execution
"""
import time
import math


class ActionExecutor:
    def __init__(self, robot, vision_handler=None, llm_layer=None):
        self.robot = robot
        self.vision_handler = vision_handler
        self.llm_layer = llm_layer
        
        # Action mapping
        self.actions = {
            # Regular movements
            'forward': self.move_forward,
            'backward': self.move_backward,
            'left': self.move_left,
            'right': self.move_right,
            'turn_left': self.turn_left,
            'turn_right': self.turn_right,
            'stop': self.stop,

            # Spoken actions
            'speak': self.speak,
            'clarify': self.clarify,

            # Visual parsing
            'describe_vision': self.describe_vision,

            # Spatial navigation
            'spatial_navigate': self.spatial_navigate,
            'unknown': self.unknown_action

            # 
        }
    
    def execute(self, intent):
        """Execute an action based on parsed intent with friction handling"""
        # Handle multi-action sequences (only if array is non-empty)
        if 'actions' in intent and len(intent.get('actions', [])) > 0:
            friction_type = intent.get('friction_type', 'none')
            text = intent.get('text', '')
            
            # Announce the plan if there's text
            if text:
                print(f"Announcing plan: {text}")
                self.robot.speak(text)
                time.sleep(0.5)
            
            print(f"Executing multi-action sequence")
            for action_item in intent['actions']:
                action = action_item.get('action', 'unknown')
                distance = action_item.get('distance', 1.0)
                turn_degrees = action_item.get('turn_degrees', 0)
                
                action_func = self.actions.get(action, self.unknown_action)
                
                # Call the appropriate function based on action type
                if action == 'turn_left' or action == 'turn_right':
                    execution_time = action_func(degrees=turn_degrees, speak=True)  # Don't speak each step
                else:
                    execution_time = action_func(distance=distance, turn_degrees=turn_degrees)
                
                # Small delay between actions
                time.sleep(execution_time / 1000 + 0.5)
            return
        
        # Single action handling
        friction_type = intent.get('friction_type', 'none')
        action = intent.get('action', 'unknown')
        distance = intent.get('distance', 1.0)
        text = intent.get('text', '')
        target_object = intent.get('target_object', None)
        turn_degrees = intent.get('turn_degrees', 0)
        
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
            # Pass all spatial navigation parameters
            action_func(
                target_object=target_object,
                distance=distance,
                turn_degrees=turn_degrees
            )
        elif action == 'turn_left' or action == 'turn_right':
            action_func(degrees=turn_degrees, speak=True)
        else:
            action_func(distance=distance, text=text)
    
    def spatial_navigate(self, target_object=None, distance=0, turn_degrees=0, **kwargs):
        """
        Navigate to a target object using pre-analyzed vision data
        Vision analysis has already been done by the LLM layer, so we just execute the plan
        """
        if not target_object:
            print("No target object specified")
            self.robot.speak("I need to know what object to navigate towards")
            return
        
        print(f"Executing spatial navigation towards: {target_object}")
        print(f"Navigation plan - Turn: {turn_degrees}°, Distance: {distance}m")
        
        # Announce the plan
        if turn_degrees == 0:
            self.robot.speak(f"Moving towards the {target_object} straight ahead")
        elif turn_degrees < 0:
            self.robot.speak(f"Turning left and moving towards the {target_object}")
        else:
            self.robot.speak(f"Turning right and moving towards the {target_object}")
        
        time.sleep(1)
        
        # Execute the turn if needed using refactored turn methods
        if turn_degrees != 0:
            if turn_degrees < 0:  # Left turn
                self.turn_left(degrees=abs(turn_degrees), speak=False)
            else:  # Right turn
                self.turn_right(degrees=turn_degrees, speak=False)
        
        # Move forward if distance > 0
        if distance > 0:
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
        return drive_time
    
    def move_backward(self, distance=1.0, **kwargs):
        """Move backward by specified distance"""
        print(f"Moving backward {distance} meter(s)")
        self.robot.speak(f"Moving backward {distance} meters")
        drive_time = self._calculate_drive_time(distance)
        self.robot.drive_time(-50, 0, drive_time)
        return drive_time
    
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
        return drive_time
    
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
        return drive_time
    
    def turn_left(self, degrees, speak=True, **kwargs):
        if not degrees:
            print("no turn degree specified")
        """Turn left by specified degrees (default 90)"""
        print(f"Turning left {degrees} degrees")
        if speak:
            self.robot.speak(f"Turning left {degrees} degrees")
        
        turn_time = self._calculate_turn_time(degrees)
        self.robot.drive_time(0, 100, turn_time)
        return turn_time
    
    def turn_right(self, degrees, speak=True, **kwargs):
        if not degrees:
            print("no turn degree specified")
        """Turn right by specified degrees (default 90)"""
        print(f"Turning right {degrees} degrees")
        if speak:
            self.robot.speak(f"Turning right {degrees} degrees")
        
        turn_time = self._calculate_turn_time(degrees)
        self.robot.drive_time(0, -100, turn_time)
        return turn_time
    
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
        
        The robot's driving acceleration is non-linear - it starts slower and 
        speeds up over time. Need to balance giving more time for both short 
        distances (due to acceleration) and long distances.
        
        Calibration: Needs testing, but using base of 2500ms per meter
        Using power relationship: time = k * distance^exponent
        Exponent = 0.7 balances short/long distance needs
        """
        exponent = 0.5  # Adjust between 0.5-1.0 based on testing
        k = 2500 / math.pow(1, exponent)  # Base calibration constant
        return int(k * math.pow(distance, exponent))
    
    def _calculate_turn_time(self, degrees):
        """Calculate turn time in milliseconds based on degrees
        
        The robot's turning acceleration is non-linear - it starts slower and 
        speeds up over time. This creates a relationship closer to sqrt(x) rather
        than linear.
        
        Calibration: 4300ms at angular speed 100 = 90 degrees
        Using sqrt relationship: time = k * sqrt(degrees)
        Where k = 4300 / sqrt(90) ≈ 453.15
        """
        k = 4300 / math.sqrt(90)  # ≈ 453.15
        return int(k * math.sqrt(degrees))