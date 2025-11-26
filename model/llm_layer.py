"""
LLM layer for intent parsing and vision-based spatial reasoning
Uses Groq API for text/intent parsing and GPT-5 nano for vision
"""
from groq import Groq
from openai import OpenAI
import json


class LLMLayer:
    def __init__(self, groq_api_key, openai_api_key=None):
        # Groq for text/intent parsing
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_model = "llama-3.1-8b-instant"
        self.conversation_history = []
        
        # GPT-5 nano for vision/spatial reasoning
        self.openai_client = None
        self.vision_model = "gpt-5-nano-2025-08-07"  # Correct model name with date
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            print("Vision enabled with GPT-5 nano")
        else:
            print("Warning: No OpenAI API key provided - vision will be disabled")
    
    def parse_intent(self, user_speech):
        """Use Groq to parse user's intent from their speech with positive friction"""
        print(f"Parsing intent for: '{user_speech}'")
        
        try:
            # Add the new user message to history
            self.conversation_history.append({
                "role": "user",
                "content": f"User said: \"{user_speech}\""
            })
            
            # Build the full message list with system prompt + conversation history
            messages = [
                {
                    "role": "system",
                    "content": """You are Misty, an embodied robot assistant who engages in natural, collaborative dialogue. 
Your role is to parse user commands AND decide when to add positive conversational friction to ensure safe, clear task execution.

# Core Parsing Rules:
1. If the user corrects themselves, only return the final corrected command
2. Extract the action type: forward, backward, left, right, stop, describe_vision, spatial_navigate, speak, clarify, or unknown
3. Extract distance in meters (default to 1 if not specified, 0 for non-movement commands)
4. Extract any text to speak
5. Determine if you should add conversational friction before executing
6. USE CONVERSATION HISTORY to understand context and references like "it", "there", "yes", etc.
7. If the command references spatial objects (e.g., "go to the plant", "move towards the chair"), use action "spatial_navigate" with the target object

# Positive Friction Categories:

## Probing
The robot poses a question regarding an external aspect (environment, actions, or context) where the answer could have been inferred via observation or action, but asking creates clarity and safety. This redirects conversation to the user.

Use Probing when:
- Ambiguous spatial references (e.g., "over there", "that side") WITH NO PRIOR CONTEXT
- Unclear object references (e.g., "it", "that thing") WITH NO PRIOR CONTEXT
- Multiple interpretations exist and you need clarification
- Safety-critical information is missing

Examples:
- User: "move over there" → "Which direction would you like me to move?"
- User: "go to it" → "Which object are you referring to?"
- User: "open the drawer" → "Which drawer should I approach - left or right?"

## Assumption Reveal
The robot reveals subjective assumptions or beliefs about the environment, actions, or plans for transparency. This uncovers previously hidden information and opens dialogue.

Use Assumption Reveal when:
- You're making an assumption about which object the user means
- Multiple valid interpretations exist and you're choosing one
- You want to verify your understanding before acting

Examples:
- "I assume you mean the plant on the left side of the room"
- "I think you're referring to the blue chair near the window"
- "I'll move towards the potted plant I see directly ahead"

## Overspecification
The robot states objective facts that are already externally observable, bringing overly specific information to attention. This refines the conversation by stating self-evident information.

Use Overspecification when:
- Confirming exact parameters before execution
- Elaborating on actions for safety
- Providing unnecessary but helpful detail

Examples:
- "Moving forward exactly 2.5 meters towards the plant"
- "I can see the chair is 3 meters away and I'll approach it carefully"
- "Turning 30 degrees to the right before moving forward"

## Reflective Pause
The robot pauses or shows doubt to slow the conversation, depicting internal reflection or environmental changes.

Use Reflective Pause when:
- You need time to process complex commands
- Environmental conditions are uncertain
- Plans need to change mid-execution

Examples:
- "Let me take a moment to analyze the scene"
- "Hmm, I need to check my surroundings first"
- "Wait, let me reconsider the best path"

## Reinforcement
The robot restates a previous utterance for emphasis, rewinding the conversation.

Use Reinforcement when:
- Confirming critical safety information
- Emphasizing important constraints
- Reiterating user requirements

Examples:
- "Just to confirm: you want me to move 2 meters forward, then turn left?"
- "So I'll navigate to the plant near the window, correct?"

# Output Format:
Return ONLY a JSON object with these fields:
- "friction_type": "none", "probing", "assumption_reveal", "overspecification", "reflective_pause", or "reinforcement"
- "action": the action to take (or "clarify" if probing, or "unknown" if unclear)
- "distance": distance in meters (0 for non-movement or vision)
- "text": the friction message to speak (clarifying question, assumption statement, overspecification, etc.)
- "target_object": (for spatial_navigate) the object to navigate towards

# Mapping Friction to Actions:
- **Probing** → action: "clarify", text: your clarifying question
- **Assumption Reveal** → action: intended action, text: your assumption statement
- **Overspecification** → action: intended action, text: detailed explanation
- **Reflective Pause** → action: "speak", text: pause/reflection statement
- **Reinforcement** → action: intended action, text: confirmation/restatement

# Spatial Navigation Commands:
When user says things like:
- "go to the plant" -> {"friction_type": "none", "action": "spatial_navigate", "target_object": "plant", "distance": 0}
- "move towards that chair" -> {"friction_type": "none", "action": "spatial_navigate", "target_object": "chair", "distance": 0}
- "navigate to the door" -> {"friction_type": "none", "action": "spatial_navigate", "target_object": "door", "distance": 0}

# Action Sequences:
User can request multiple actions in sequence using keywords like:
- "and then", "after that", "then", "next"
Example: "move forward 2 meters and then turn left"

When multiple actions are detected, return:
{"friction_type": "none", "actions": [{"action": "forward", "distance": 2}, {"action": "left", "distance": 1}]}

# Examples:

**Simple command (no friction):**
User: "move forward 2 meters"
{"friction_type": "none", "action": "forward", "distance": 2, "text": "", "target_object": null}

**Spatial navigation (no friction):**
User: "go to the plant"
{"friction_type": "none", "action": "spatial_navigate", "target_object": "plant", "distance": 0, "text": ""}

**Probing - ambiguous reference:**
User: "move over there"
{"friction_type": "probing", "action": "clarify", "distance": 0, "text": "Which direction would you like me to move?", "target_object": null}

**Assumption Reveal - clarifying assumption:**
User: "go to the chair"
Vision shows multiple chairs
{"friction_type": "assumption_reveal", "action": "spatial_navigate", "target_object": "chair", "distance": 0, "text": "I see multiple chairs. I'll move towards the blue chair on the left."}

**Overspecification - detailed safety info:**
User: "move forward"
{"friction_type": "overspecification", "action": "forward", "distance": 1, "text": "Moving forward exactly 1 meter at a safe speed"}

**Reflective Pause - needs analysis:**
User: "go to that complex area"
{"friction_type": "reflective_pause", "action": "speak", "distance": 0, "text": "Let me analyze the scene first to find the safest path", "target_object": null}

**Reinforcement - confirming safety-critical:**
User: "move backward 3 meters"
{"friction_type": "reinforcement", "action": "backward", "distance": 3, "text": "Just to confirm: moving backward 3 meters"}

**Vision request:**
User: "what do you see"
{"friction_type": "none", "action": "describe_vision", "distance": 0, "text": "", "target_object": null}

**Multi-action sequence:**
User: "move forward 2 meters and then turn left"
{"friction_type": "none", "actions": [{"action": "forward", "distance": 2}, {"action": "left", "distance": 1}]}

**Follow-up with context:**
Previous: Asked user to clarify which direction
User: "to the left"
{"friction_type": "none", "action": "left", "distance": 1, "text": ""}"""
                }
            ]
            
            # Add all conversation history
            messages.extend(self.conversation_history)
            
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            llm_output = response.choices[0].message.content.strip()
            print(f"LLM raw output: {llm_output}")
            
            # Add the assistant's response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": llm_output
            })
            
            # Extract JSON from response
            json_start = llm_output.find('{')
            json_end = llm_output.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = llm_output[json_start:json_end]
                intent = json.loads(json_str)
                
                # Check if it's a multi-action command
                if 'actions' in intent:
                    print(f"Parsed multi-action sequence: {intent['actions']}")
                    return intent
                
                # Single action
                action = intent.get('action', 'unknown')
                distance = intent.get('distance', 1.0)
                text = intent.get('text', '')
                friction_type = intent.get('friction_type', 'none')
                target_object = intent.get('target_object', None)
                
                print(f"Parsed intent - Action: {action}, Distance: {distance}, Target: {target_object}, Friction: {friction_type}")
                return {
                    'action': action,
                    'distance': distance,
                    'text': text,
                    'friction_type': friction_type,
                    'target_object': target_object
                }
            else:
                print("Could not find JSON in response")
                return {'action': 'unknown', 'distance': 0, 'text': '', 'friction_type': 'none', 'target_object': None}
                
        except Exception as e:
            print(f"Error calling Groq LLM: {e}")
            import traceback
            traceback.print_exc()
            return {'action': 'unknown', 'distance': 0, 'text': '', 'friction_type': 'none', 'target_object': None}
    
    def analyze_spatial_scene(self, image_data_base64, target_object):
        """
        Use GPT-5 nano to analyze the scene and determine spatial navigation info
        
        Args:
            image_data_base64: Base64 encoded image data
            target_object: The object to navigate towards (e.g., "plant", "chair")
            
        Returns:
            dict with:
                - found: bool, whether the object is visible
                - direction: str, "straight", "left", "right", or "not_visible"
                - distance: float, estimated distance in meters
                - confidence: str, "high", "medium", or "low"
                - reasoning: str, explanation of the analysis
                - suggested_action: str, what the robot should do
        """
        if not self.openai_client:
            print("Error: No OpenAI API key provided - cannot analyze spatial scene")
            return None
        
        try:
            print(f"Analyzing spatial scene with GPT-5 nano for target: {target_object}")
            
            prompt = f"""You are a vision system for a mobile robot. Analyze this image to help the robot navigate towards a {target_object}.

Your task:
1. Determine if you can see the {target_object} in the image
2. If visible, determine its spatial position relative to the robot's camera:
   - Is it directly ahead ("straight")?
   - Is it to the left ("left") and by approximately how many degrees (estimate)?
   - Is it to the right ("right") and by approximately how many degrees (estimate)?
3. Estimate the distance to the {target_object} in meters (consider typical object sizes and perspective)
4. Assess your confidence level: "high", "medium", or "low"
5. Provide reasoning for your analysis
6. Suggest what action the robot should take

Return ONLY a JSON object with this exact structure:
{{
    "found": true or false,
    "direction": "straight" or "left" or "right" or "not_visible",
    "turn_degrees": estimated degrees to turn (0 if straight, negative for left, positive for right),
    "distance": estimated distance in meters,
    "confidence": "high" or "medium" or "low",
    "reasoning": "brief explanation of what you see and why",
    "suggested_action": "what the robot should do next"
}}

Examples:
- If you see a plant directly in front, 2 meters away:
{{"found": true, "direction": "straight", "turn_degrees": 0, "distance": 2.0, "confidence": "high", "reasoning": "I can see a potted plant directly in the center of the frame at approximately 2 meters", "suggested_action": "Move straight forward 2 meters"}}

- If you see a chair to the right, requiring ~30 degree turn:
{{"found": true, "direction": "right", "turn_degrees": 30, "distance": 1.5, "confidence": "medium", "reasoning": "I can see a chair positioned to the right side of the frame, roughly 30 degrees off center", "suggested_action": "Turn right 30 degrees, then move forward 1.5 meters"}}

- If you don't see the target object:
{{"found": false, "direction": "not_visible", "turn_degrees": 0, "distance": 0, "confidence": "high", "reasoning": "I cannot see a {target_object} anywhere in the current field of view", "suggested_action": "Ask user for clarification on the location"}}

Be precise and practical in your estimates. Return ONLY the JSON, nothing else."""

            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=1500  # Increased to allow for reasoning tokens + output
            )
            
            vision_output = response.choices[0].message.content.strip()
            print(f"GPT-5 nano raw output: {vision_output}")
            
            # Extract JSON from response
            json_start = vision_output.find('{')
            json_end = vision_output.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = vision_output[json_start:json_end]
                analysis = json.loads(json_str)
                
                print(f"Spatial analysis - Found: {analysis['found']}, Direction: {analysis['direction']}, Distance: {analysis['distance']}m")
                return analysis
            else:
                print("Could not find JSON in GPT-5 nano response")
                return None
                
        except Exception as e:
            print(f"Error in GPT-5 nano spatial analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def describe_image(self, image_data_base64):
        """Get general description of an image using GPT-5 nano"""
        if not self.openai_client:
            print("Error: No OpenAI API key provided - cannot describe images")
            return "Vision is not available. Please provide an OpenAI API key."
        
        try:
            print("Analyzing image with GPT-5 nano...")
            
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe what you see in this image in 2-3 sentences. Be specific about objects, people, and the setting."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=800  # Increased to allow for reasoning tokens + output
            )
            
            description = response.choices[0].message.content.strip()
            print(f"Vision description: {description}")
            return description
            
        except Exception as e:
            print(f"Error in GPT-5 nano image description: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def clear_conversation_history(self):
        """Clear conversation history (useful for starting fresh)"""
        self.conversation_history = []
        print("Conversation history cleared")