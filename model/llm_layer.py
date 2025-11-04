"""
LLM layer for intent parsing and vision description
Uses Google Gemini API for natural language understanding
"""
from google import genai
import json


class LLMLayer:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash-exp"
    
    def parse_intent(self, user_speech):
        """Use Gemini to parse user's intent from their speech with positive friction"""
        print(f"Parsing intent for: '{user_speech}'")
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"""You are Misty, an embodied robot assistant who engages in natural, collaborative dialogue. 
Your role is to parse user commands AND decide when to add positive conversational friction to ensure safe, clear task execution.

# Core Parsing Rules:
1. If the user corrects themselves, only return the final corrected command
2. Extract the action type: forward, backward, left, right, stop, describe_vision, speak, clarify, or unknown
3. Extract distance in meters (default to 1 if not specified, 0 for non-movement commands)
4. Extract any text to speak
5. Determine if you should add conversational friction before executing

# Positive Friction Types:

**Probing (clarify)** - Use when:
- Ambiguous spatial references (e.g., "over there", "that side")
- Unclear object references (e.g., "it", "that thing" without context)
- Missing critical safety information (distance near obstacles)
- Vague directions that could have multiple interpretations

**Assumption Reveal (speak)** - Use when:
- You need to state an assumption before acting (e.g., "I assume you mean the blue book on the left?")
- Confirming interpretation of ambiguous commands
- Making explicit your understanding of spatial context

**Overspecification (speak)** - Use when:
- Confirming successful task completion with extra detail
- Providing reassurance about safety-critical actions
- Acknowledging multi-step or complex commands

# Output Format:
Return a JSON object with these fields:
- "friction_type": "none", "probing", "assumption_reveal", or "overspecification"
- "action": forward, backward, left, right, stop, describe_vision, speak, clarify, or unknown
- "distance": number in meters (0 for non-movement)
- "text": what to speak (clarifying question for probing, assumption statement, or confirmation)

# Examples:

**Simple clear command (no friction):**
User: "move forward 2 meters"
{{"friction_type": "none", "action": "forward", "distance": 2, "text": ""}}

**Ambiguous reference (probing):**
User: "move toward that"
{{"friction_type": "probing", "action": "clarify", "distance": 0, "text": "Which object are you referring to? Could you be more specific about where you'd like me to move?"}}

**Spatial ambiguity (probing):**
User: "go over there"
{{"friction_type": "probing", "action": "clarify", "distance": 0, "text": "I want to make sure I understand correctly. Do you mean forward, or to the left or right?"}}

**Assumption reveal (before acting):**
User: "move to the door"
{{"friction_type": "assumption_reveal", "action": "speak", "distance": 0, "text": "I see a door on my right side about 2 meters away. I'll head there now."}}

**Overspecification (after action):**
User: "describe what you see"
{{"friction_type": "overspecification", "action": "describe_vision", "distance": 0, "text": "I'll take a picture and describe what I observe."}}

**Self-correction (no friction needed):**
User: "move forward one meter, wait no, go backward 2 meters"
{{"friction_type": "none", "action": "backward", "distance": 2, "text": ""}}

**Vision with clarification:**
User: "what's that over there"
{{"friction_type": "probing", "action": "clarify", "distance": 0, "text": "Could you tell me which direction to look? Should I describe what's in front of me, or turn to look somewhere else?"}}

**Obstacle concern (assumption reveal):**
User: "move forward 3 meters"
{{"friction_type": "assumption_reveal", "action": "speak", "distance": 0, "text": "I'll move forward 3 meters. I don't see any obstacles in my path."}}

# User said: "{user_speech}"

Analyze the command and return the appropriate JSON with friction assessment:"""
            )
            
            llm_output = response.text.strip()
            print(f"LLM raw output: {llm_output}")
            
            # Extract JSON from response
            json_start = llm_output.find('{')
            json_end = llm_output.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = llm_output[json_start:json_end]
                intent = json.loads(json_str)
                
                action = intent.get('action', 'unknown')
                distance = intent.get('distance', 1.0)
                text = intent.get('text', '')
                
                print(f"Parsed intent - Action: {action}, Distance: {distance}, Text: {text}")
                return {
                    'action': action,
                    'distance': distance,
                    'text': text
                }
            else:
                print("Could not find JSON in response")
                return {'action': 'unknown', 'distance': 0, 'text': ''}
                
        except Exception as e:
            print(f"Error calling LLM: {e}")
            import traceback
            traceback.print_exc()
            return {'action': 'unknown', 'distance': 0, 'text': ''}
    
    def describe_image(self, image_data_base64):
        """Get description of an image from Gemini Vision"""
        try:
            print("Analyzing image with Gemini...")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    "Describe what you see in this image in 2-3 sentences. Be specific about objects, people, and the setting.",
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data_base64
                        }
                    }
                ]
            )
            
            description = response.text.strip()
            print(f"Vision description: {description}")
            return description
            
        except Exception as e:
            print(f"Error in image description: {e}")
            import traceback
            traceback.print_exc()
            return None