"""
LLM layer using GPT-5 nano as unified VLM for both intent parsing and vision
Combines text and vision analysis in a single multimodal call
"""
from openai import OpenAI
import json


class LLMLayer:
    def __init__(self, openai_api_key):
        # GPT-5 nano for unified text + vision processing
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.vision_model = "gpt-5-nano-2025-08-07"
        self.conversation_history = []
        print("Unified VLM enabled with GPT-5 nano")
    
    def parse_intent_with_vision(self, user_speech, image_data_base64=None):
        """
        Unified function that uses GPT-5 nano to parse user intent with optional vision
        Combines text parsing, spatial reasoning, and positive friction in one call
        
        Args:
            user_speech: The user's spoken/typed command
            image_data_base64: Optional base64 encoded image for spatial commands
            
        Returns:
            dict with:
                - action: the action to take
                - distance: distance parameter
                - text: friction message or speech text
                - friction_type: type of friction applied
                - target_object: for spatial_navigate actions
                - turn_degrees: for spatial navigation
                - clarification_needed: specific clarification question if needed
        """
        print(f"Parsing intent for: '{user_speech}'")
        
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": f"User said: \"{user_speech}\""
            })
            
            # Build the unified system prompt (more concise)
            system_prompt = """You are Misty, a robot with vision that parses commands and decides when to ask clarifying questions.

# Actions: forward, backward, left, right, turn_left, turn_right, stop, describe_vision, spatial_navigate, speak, clarify, unknown

# Friction Types (when to slow down and ask):
- **probing**: Ask questions for clarity (ambiguous references, multiple objects visible, missing safety info)
- **assumption_reveal**: State your assumptions transparently
- **overspecification**: Confirm exact parameters
- **reflective_pause**: Show you're thinking
- **reinforcement**: Restate for emphasis

# Spatial Commands with Vision:
If user gives a command that asks you to move towards an object X, for example "go to X" or "go over to X", AND you see an image:
1. Count instances of X in the image
2. If MULTIPLE found → friction_type: "probing", action: "clarify", ask which one with distinguishing details
3. If ONE found → action: "spatial_navigate", estimate turn_degrees based on the location of X relative to the center y-axis of the image (negative=left, positive=right, 0=straight) and distance
4. If NOT found → friction_type: "probing", action: "clarify", ask for location

# Multi-action: "X and then Y" → {"actions": [{"action": "X"}, {"action": "Y"}]}

# Output JSON:
{
  "friction_type": "none/probing/assumption_reveal/overspecification/reflective_pause/reinforcement",
  "action": "action_name or clarify",
  "distance": 0,
  "text": "what to say",
  "target_object": "object name or null",
  "turn_degrees": 0,
  "clarification_needed": "question or null",
  "confidence": "high/medium/low"
}

Examples:
- "move forward 2m" → {"friction_type":"none","action":"forward","distance":2,"text":"","target_object":null,"turn_degrees":0,"clarification_needed":null}
- "go to plant" + image shows 1 plant ahead → {"friction_type":"none","action":"spatial_navigate","target_object":"plant","distance":2.0,"turn_degrees":0,"text":"","clarification_needed":null,"confidence":"high"}
- "go to trash bin" + image shows 2 bins → {"friction_type":"probing","action":"clarify","target_object":"trash bin","distance":0,"turn_degrees":0,"text":"I see two trash bins - the blue one on the left near the door, or the black one on the right by the desk?","clarification_needed":"I see two trash bins - the blue one on the left near the door, or the black one on the right by the desk?","confidence":"high"}
- "what do you see" → {"friction_type":"none","action":"describe_vision","distance":0,"text":"","target_object":null,"turn_degrees":0,"clarification_needed":null}

Return ONLY valid JSON, nothing else."""

            # Build message content
            message_content = []
            
            # Add text
            message_content.append({
                "type": "text",
                "text": system_prompt
            })
            
            # Build user message with optional image
            user_content = []
            
            # Add the command text
            user_content.append({
                "type": "text",
                "text": f"User command: \"{user_speech}\""
            })
            
            # Add image if provided (for spatial commands)
            if image_data_base64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data_base64}"
                    }
                })
                print("Including vision analysis in intent parsing")
            
            # Build full messages list
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            
            # Add conversation history
            messages.extend(self.conversation_history)
            
            # Add current user message with optional image
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Call GPT-5 nano
            print(f"Calling GPT-5 nano with {len(messages)} messages")
            
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                max_completion_tokens=2000
                # Note: GPT-5 nano doesn't support temperature parameter, uses default of 1
            )
            
            print(f"Response received. Finish reason: {response.choices[0].finish_reason}")
            
            llm_output = response.choices[0].message.content
            
            # Debug: Check if output is None or empty
            if llm_output is None:
                print("ERROR: GPT-5 nano returned None")
                print(f"Full response object: {response}")
                llm_output = ""
            else:
                llm_output = llm_output.strip()
            
            print(f"GPT-5 nano raw output: '{llm_output}'")
            print(f"Output length: {len(llm_output)} characters")
            
            # Add assistant's response to history
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
                
                # Single action - ensure all fields exist
                result = {
                    'action': intent.get('action', 'unknown'),
                    'distance': intent.get('distance', 0),
                    'text': intent.get('text', ''),
                    'friction_type': intent.get('friction_type', 'none'),
                    'target_object': intent.get('target_object', None),
                    'turn_degrees': intent.get('turn_degrees', 0),
                    'clarification_needed': intent.get('clarification_needed', None),
                    'confidence': intent.get('confidence', 'medium')
                }
                
                action = result['action']
                friction = result['friction_type']
                target = result['target_object']
                
                print(f"Parsed intent - Action: {action}, Target: {target}, Friction: {friction}")
                
                return result
            else:
                print("Could not find JSON in response")
                return {
                    'action': 'unknown',
                    'distance': 0,
                    'text': '',
                    'friction_type': 'none',
                    'target_object': None,
                    'turn_degrees': 0,
                    'clarification_needed': None,
                    'confidence': 'low'
                }
                
        except Exception as e:
            print(f"Error calling GPT-5 nano: {e}")
            import traceback
            traceback.print_exc()
            return {
                'action': 'unknown',
                'distance': 0,
                'text': '',
                'friction_type': 'none',
                'target_object': None,
                'turn_degrees': 0,
                'clarification_needed': None,
                'confidence': 'low'
            }
    
    def describe_image(self, image_data_base64):
        """Get general description of an image using GPT-5 nano"""
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
                max_completion_tokens=800
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