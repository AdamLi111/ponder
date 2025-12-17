"""
LLM layer using GPT-5 nano WITHOUT positive friction for control testing
Makes best-guess assumptions rather than asking clarifying questions
Includes find_object support
"""
from openai import OpenAI
import json


class LLMLayerNoFriction:
    def __init__(self, openai_api_key):
        # GPT-5 nano for unified text + vision processing
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.vision_model = "gpt-5-nano-2025-08-07"
        self.conversation_history = []
        print("Unified VLM enabled with GPT-5 nano (NO FRICTION MODE)")
    
    def parse_intent_with_vision(self, user_speech, image_data_base64=None):
        """
        Unified function that uses GPT-5 nano to parse user intent with optional vision
        NO POSITIVE FRICTION - makes best guesses and executes immediately
        
        Args:
            user_speech: The user's spoken/typed command
            image_data_base64: Optional base64 encoded image for spatial commands
            
        Returns:
            dict with:
                - action: the action to take
                - distance: distance parameter
                - text: speech text (minimal)
                - target_object: for spatial_navigate actions
                - turn_degrees: for spatial navigation
                - confidence: confidence level
        """
        print(f"Parsing intent for: '{user_speech}'")
        
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": f"User said: \"{user_speech}\""
            })
            
            # Build the system prompt WITHOUT friction mechanisms
            system_prompt = """You are Misty, a robot with vision that parses commands into executable actions.

# Actions: forward, backward, left, right, turn_left, turn_right, stop, describe_vision, find_object, spatial_navigate, speak, unknown

# Find Object Action:
When user says "find [object]" or "look for [object]" or "where is my [object]":
- Return action: "find_object" with target_object set to the object name
- This triggers a 360-degree scan with 4 images

# Spatial Commands with Vision:
If user says "go to X" AND you see an image:
1. Identify X in the image
2. If multiple instances of X are visible, choose the one based on your reasoning.
3. If X is not straight in front of you, estimate turn_degrees (negative=left, positive=right, 0=straight) necessary to face straight at X, and then estimate the distance to the selected object
4. Return action: "spatial_navigate" with the target object

# Multi-action: "X and then Y" → {"actions": [{"action": "X"}, {"action": "Y"}]}

# Output JSON:
{
  "action": "action_name",
  "distance": 0,
  "text": "brief acknowledgment",
  "target_object": "object name or null",
  "turn_degrees": 0,
  "confidence": "high/medium/low"
}

Examples:
- "find my bag" → {"action":"find_object","target_object":"bag","distance":0,"text":"Looking for your bag","turn_degrees":0,"confidence":"high"}
- "where is the laptop" → {"action":"find_object","target_object":"laptop","distance":0,"text":"Searching for the laptop","turn_degrees":0,"confidence":"high"}
- "move forward 2m" → {"action":"forward","distance":2,"text":"Moving forward","target_object":null,"turn_degrees":0,"confidence":"high"}
- "go to plant" + image shows 1 plant ahead → {"action":"spatial_navigate","target_object":"plant","distance":2.0,"turn_degrees":0,"text":"Going to the plant","confidence":"high"}
- "go to cup" + image shows 2 cups → {"action":"spatial_navigate","target_object":"cup","distance":1.5,"turn_degrees":-10,"text":"Going to the cup","confidence":"medium"}
- "what do you see" → {"action":"describe_vision","distance":0,"text":"","target_object":null,"turn_degrees":0,"confidence":"high"}

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
                print("Including vision analysis in intent parsing (no friction mode)")
            
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
            print(f"Calling GPT-5 nano with {len(messages)} messages (NO FRICTION)")
            
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                max_completion_tokens=2000
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
                
                # Single action - ensure all fields exist (no friction fields)
                result = {
                    'action': intent.get('action', 'unknown'),
                    'distance': intent.get('distance', 0),
                    'text': intent.get('text', ''),
                    'target_object': intent.get('target_object', None),
                    'turn_degrees': intent.get('turn_degrees', 0),
                    'confidence': intent.get('confidence', 'medium'),
                    # Set friction fields to defaults for compatibility
                    'friction_type': 'none',
                    'clarification_needed': None
                }
                
                action = result['action']
                target = result['target_object']
                
                print(f"Parsed intent (NO FRICTION) - Action: {action}, Target: {target}")
                
                return result
            else:
                print("Could not find JSON in response")
                return {
                    'action': 'unknown',
                    'distance': 0,
                    'text': '',
                    'target_object': None,
                    'turn_degrees': 0,
                    'confidence': 'low',
                    'friction_type': 'none',
                    'clarification_needed': None
                }
                
        except Exception as e:
            print(f"Error calling GPT-5 nano: {e}")
            import traceback
            traceback.print_exc()
            return {
                'action': 'unknown',
                'distance': 0,
                'text': '',
                'target_object': None,
                'turn_degrees': 0,
                'confidence': 'low',
                'friction_type': 'none',
                'clarification_needed': None
            }
    
    def find_object_in_images(self, target_object, images):
        """
        Analyze multiple images from 360-degree scan to find an object
        Same implementation as friction version since this is analysis, not friction
        
        Args:
            target_object: Name of the object to find
            images: List of dicts with 'direction' and 'data' (base64) keys
        
        Returns:
            dict with found, response, count, locations
        """
        print(f"Analyzing {len(images)} images to find: {target_object}")
        
        try:
            prompt = f"""You are analyzing 4 images from a robot's 360-degree scan to find a {target_object}.

The images are from these directions:
1. Front view (starting position)
2. Left view (90 degrees left from start)
3. Back view (180 degrees from start)
4. Right view (270 degrees left / 90 degrees right from start)

Your task:
1. Look for "{target_object}" in ALL 4 images
2. Count how many instances you find
3. For each instance, note which direction and describe its location

Respond in JSON format:
{{
  "found": true/false,
  "count": number_of_instances,
  "instances": [
    {{
      "direction": "front/left/back/right",
      "description": "brief location description"
    }}
  ]
}}

Return ONLY valid JSON, nothing else."""

            user_content = [{"type": "text", "text": prompt}]
            
            for img in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img['data']}"
                    }
                })
            
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": user_content}],
                max_completion_tokens=2000
            )
            
            llm_output = response.choices[0].message.content.strip()
            
            json_start = llm_output.find('{')
            json_end = llm_output.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = llm_output[json_start:json_end]
                analysis = json.loads(json_str)
                
                found = analysis.get('found', False)
                count = analysis.get('count', 0)
                instances = analysis.get('instances', [])
                
                if not found or count == 0:
                    response_text = f"I scanned the entire room but couldn't find your {target_object}."
                elif count == 1:
                    instance = instances[0]
                    direction = instance.get('direction', 'unknown')
                    description = instance.get('description', 'nearby')
                    response_text = f"I found your {target_object}. It's {description}, to my {direction}."
                else:
                    response_text = f"I found {count} {target_object}s. "
                    for i, instance in enumerate(instances, 1):
                        direction = instance.get('direction', 'unknown')
                        description = instance.get('description', 'there')
                        response_text += f"Number {i} is {description} to my {direction}. "
                    response_text += f"Which one is yours?"
                
                return {
                    'found': found,
                    'response': response_text,
                    'count': count,
                    'locations': instances
                }
            else:
                return {
                    'found': False,
                    'response': f"Sorry, I had trouble analyzing the images.",
                    'count': 0,
                    'locations': []
                }
                
        except Exception as e:
            print(f"Error in find_object_in_images: {e}")
            import traceback
            traceback.print_exc()
            return {
                'found': False,
                'response': f"Sorry, I encountered an error while searching.",
                'count': 0,
                'locations': []
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