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
        """Use Gemini to parse user's intent from their speech"""
        print(f"Parsing intent for: '{user_speech}'")
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"""You are a command parser for a robot named Misty. 
Analyze what the user says and extract their FINAL intent.

Rules:
1. If the user corrects themselves, only return the final corrected command
2. Extract the action type: forward, backward, left, right, stop, describe_vision, speak, or unknown
3. Extract the distance in meters (default to 1 if not specified, 0 for non-movement commands)
4. Extract any text to speak (for speak action)
5. Return ONLY a JSON object with 'action', 'distance', and 'text' fields
6. If unclear or not a valid command, return {{"action": "unknown", "distance": 0, "text": ""}}

Movement Examples:
- "move forward 2 meters" -> {{"action": "forward", "distance": 2, "text": ""}}
- "go left" -> {{"action": "left", "distance": 1, "text": ""}}
- "move forward one meter, wait no, go backward 2 meters" -> {{"action": "backward", "distance": 2, "text": ""}}
- "stop" -> {{"action": "stop", "distance": 0, "text": ""}}

Vision Examples:
- "what do you see" -> {{"action": "describe_vision", "distance": 0, "text": ""}}
- "describe what's in front of you" -> {{"action": "describe_vision", "distance": 0, "text": ""}}
- "look and tell me what you see" -> {{"action": "describe_vision", "distance": 0, "text": ""}}

Speak Examples:
- "say hello" -> {{"action": "speak", "distance": 0, "text": "hello"}}
- "tell me a joke" -> {{"action": "speak", "distance": 0, "text": "Why don't scientists trust atoms? Because they make up everything!"}}
- "introduce yourself" -> {{"action": "speak", "distance": 0, "text": "Hi, I'm Misty, a robot assistant here to help you!"}}

User said: "{user_speech}"

Return only the JSON object, nothing else:"""
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