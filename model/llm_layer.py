"""
LLM layer using GPT-5 nano as unified VLM for both intent parsing and vision
Includes 360-degree object finding capability
"""
from openai import OpenAI
import json


class LLMLayer:
    def __init__(self, openai_api_key, logger=None):
        # GPT-5 nano for unified text + vision processing
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.vision_model = "gpt-5-nano-2025-08-07"
        self.conversation_history = []
        self.logger = logger
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

# Actions: forward, backward, left, right, turn_left, turn_right, stop, describe_vision, find_object, spatial_navigate, speak, clarify, unknown

# Find Object Action:
When user says "find [object]" or "look for [object]" or "where is my [object]":
- Return action: "find_object" with target_object set to the object name
- This triggers a 360-degree scan with 4 images
- No image needed in this parsing step

# Friction Types (when to slow down and ask):
- **probing**: Ask questions ONLY for truly ambiguous situations (multiple identical target objects, unclear user intent, CRITICAL SAFETY CONCERNS)
- **assumption_reveal**: State your assumptions transparently
- **overspecification**: Confirm exact parameters
- **reflective_pause**: Show you're thinking
- **reinforcement**: Restate for emphasis

# CRITICAL SAFETY RULES (HIGHEST PRIORITY):
When you see an image WITH any movement command, evaluate in this order:

**PRIORITY 1 - EDGE/DROP HAZARDS (MOST CRITICAL):**
1. **FIRST**, check if moving the requested distance will cause Misty to fall off an edge/desk/stairs
2. Look for visible edges, drops, or the end of the current surface
3. **If the command would cause falling** → ALWAYS use probing friction to confirm user intent, regardless of obstacles
4. Edge safety takes absolute precedence over all other concerns

**PRIORITY 2 - OBSTACLE AVOIDANCE (SECONDARY):**
5. **After confirming no edge hazard**, check for obstacles blocking the direct path
6. If obstacle exists AND user says "go around" → autonomously pick safest side and generate multi-action sequence
7. If obstacle exists but user hasn't said to go around → use probing friction to ask
8. Only ask which side if BOTH sides equally blocked

# Autonomous Obstacle Avoidance:
When user says "go around [obstacle]" or similar:
1. **Carefully analyze the image** to determine which side (left OR right) has more clearance and is safer
2. **Choose the side with more open space** - this could be left OR right depending on the scene
3. Generate a multi-action sequence using the "actions" array format:
   - Turn to face the clear side
   - Move forward to bypass obstacle
   - Turn back toward target
   - Move forward to reach target
4. **DO NOT default to always turning left** - pick the side that makes sense based on visual analysis
5. DO NOT ask which side unless both sides are equally blocked

# Spatial Navigation ("go to X"):
1. Count instances of X in the image
2. Check for edge hazards first, then obstacles between Misty and X
3. If MULTIPLE X found → ask which one
4. If ONE obstacle blocks path → use probing to ask if user wants to go around or stop there
5. If ONE X found AND clear path AND no edge hazard → navigate with turn_degrees and distance

# Output JSON Format:

For single actions:
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

For multi-action sequences:
If more clearance on the left:
{
  "friction_type": "none/assumption_reveal",
  "text": "brief description of plan",
  "confidence": "high/medium/low",
  "actions": [
    {"action": "turn_left", "turn_degrees": 30, "distance": 0},
    {"action": "forward", "distance": 2.0},
    {"action": "turn_right", "turn_degrees": 60, "distance": 0},
    {"action": "forward", "distance": 1.0}
  ]
}

if more clearance on the right:
{
  "friction_type": "none/assumption_reveal",
  "text": "brief description of plan",
  "confidence": "high/medium/low",
  "actions": [
    {"action": "turn_right", "turn_degrees": 30, "distance": 0},
    {"action": "forward", "distance": 2.0},
    {"action": "turn_left", "turn_degrees": 60, "distance": 0},
    {"action": "forward", "distance": 1.0}
  ]
}

# EXAMPLES (in priority order):

## CRITICAL: Edge/Drop Detection (HIGHEST PRIORITY):
- "move forward 3m" + image shows desk edge 1m ahead → {"friction_type":"probing","action":"clarify","text":"Moving forward 3 meters will make me fall off the desk edge. Are you sure you want me to proceed?","clarification_needed":"Moving forward 3 meters will make me fall off the desk edge. Are you sure?","confidence":"high"}
- "go backward 2m" + image shows staircase behind → {"friction_type":"probing","action":"clarify","text":"Moving backward will make me fall down the stairs. Should I stop here instead?","clarification_needed":"Moving backward will make me fall down the stairs. Should I stop?","confidence":"high"}
- "move forward 5m" + desk edge visible at 4m, box at 2m → {"friction_type":"probing","action":"clarify","text":"I see a desk edge ahead that I would fall off. There's also a box in the way. Should I stop before the edge?","clarification_needed":"Desk edge ahead - would cause falling. Should I stop?","confidence":"high"}

## Single Movement Commands (no hazards):
- "move forward 2m" + clear path, no edges → {"friction_type":"none","action":"forward","distance":2,"text":"","target_object":null,"turn_degrees":0,"confidence":"high"}
- "turn right 70 degrees" → {"friction_type":"none","action":"turn_right","distance":0,"text":"","target_object":null,"turn_degrees":70,"confidence":"high"}
- "turn left 90 degrees" → {"friction_type":"none","action":"turn_left","distance":0,"text":"","target_object":null,"turn_degrees":90,"confidence":"high"}

## Obstacle Detection (no edge hazards present):
- "move forward 3m" + box at 2m, no edge hazards → {"friction_type":"probing","action":"clarify","text":"I see a box blocking my path. Should I go around it or stop?","clarification_needed":"I see a box blocking my path. Should I go around it or stop?","confidence":"medium"}
- "go to chair" + chair visible but trash can blocking, no edges → {"friction_type":"probing","action":"clarify","text":"I see the chair behind a trash can. Should I go around it or stop there?","clarification_needed":"I see the chair behind a trash can. Should I go around it or stop?","confidence":"medium"}

## Autonomous Navigation (Multi-action):
- "go around it" + image shows LEFT side clear, right blocked, no edge hazards → {"friction_type":"assumption_reveal","text":"I'll navigate around the left side where there's more clearance","confidence":"high","actions":[{"action":"turn_left","turn_degrees":30},{"action":"forward","distance":1.5},{"action":"turn_right","turn_degrees":30},{"action":"forward","distance":1.0}]}
- "go around it" + image shows RIGHT side clear, left blocked, no edge hazards → {"friction_type":"assumption_reveal","text":"I'll navigate around the right side where there's more clearance","confidence":"high","actions":[{"action":"turn_right","turn_degrees":25},{"action":"forward","distance":1.3},{"action":"turn_left","turn_degrees":50},{"action":"forward","distance":0.9}]}
- "go around the box" + both sides clear, RIGHT has more space, no edges → {"friction_type":"none","text":"Navigating around the right side","confidence":"high","actions":[{"action":"turn_right","turn_degrees":30},{"action":"forward","distance":1.4},{"action":"turn_left","turn_degrees":55},{"action":"forward","distance":0.8}]}
- "go around the obstacle" + both sides clear, LEFT has more space, no edges → {"friction_type":"none","text":"Navigating around the left side","confidence":"high","actions":[{"action":"turn_left","turn_degrees":28},{"action":"forward","distance":1.3},{"action":"turn_right","turn_degrees":52},{"action":"forward","distance":0.9}]}
- "go around it to the left" + user explicitly specifies left, no edge hazards → {"friction_type":"none","text":"Going around the left side","confidence":"high","actions":[{"action":"turn_left","turn_degrees":25},{"action":"forward","distance":1.2},{"action":"turn_right","turn_degrees":50},{"action":"forward","distance":0.8}]}
- "stop at it" + no edge hazards → {"friction_type":"none","text":"Ok, I'll stop at the obstacle's location","confidence":"high","action":"forward","distance":1.8]}

## Clear Path Navigation:
- "go to plant" + one plant, clear path, no edge hazards → {"friction_type":"none","action":"spatial_navigate","target_object":"plant","distance":2.0,"turn_degrees":5,"text":"","clarification_needed":null,"confidence":"high"}

## Find Object Commands:
- "find my bag" → {"friction_type":"none","action":"find_object","target_object":"bag","distance":0,"text":"","turn_degrees":0,"confidence":"high"}
- "look for the keys" → {"friction_type":"none","action":"find_object","target_object":"keys","distance":0,"text":"","turn_degrees":0,"confidence":"high"}
- "where is my laptop" → {"friction_type":"none","action":"find_object","target_object":"laptop","distance":0,"text":"","turn_degrees":0,"confidence":"high"}

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
                max_completion_tokens=8192
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
            
            # Log VLM output
            if self.logger:
                self.logger.log_vlm_output(llm_output)
            
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
    
    def find_object_in_images(self, target_object, images):
        """
        Analyze multiple images from 360-degree scan to find an object
        
        Args:
            target_object: Name of the object to find
            images: List of dicts with 'direction' and 'data' (base64) keys
                   e.g., [{'direction': 'front', 'data': 'base64...'}, ...]
        
        Returns:
            dict with:
                - found: boolean indicating if object was found
                - response: text response to speak to user
                - count: number of instances found
                - locations: list of location descriptions
        """
        print(f"Analyzing {len(images)} images to find: {target_object}")
        
        try:
            # Build the prompt for finding the object
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
      "description": "brief location description (e.g., 'on the floor near the door', 'on the desk', 'hanging on the wall')"
    }}
  ]
}}

If NOT found: {{"found": false, "count": 0, "instances": []}}
If ONE found: Include direction and location
If MULTIPLE found: List all with directions and brief appearance differences

Return ONLY valid JSON, nothing else."""

            # Build messages with all 4 images
            user_content = [{"type": "text", "text": prompt}]
            
            for img in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img['data']}"
                    }
                })
            
            print(f"Sending {len(images)} images to GPT-5 nano for analysis...")
            
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                max_completion_tokens=2000
            )
            
            llm_output = response.choices[0].message.content.strip()
            print(f"VLM analysis output: {llm_output}")
            
            # Parse JSON response
            json_start = llm_output.find('{')
            json_end = llm_output.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = llm_output[json_start:json_end]
                analysis = json.loads(json_str)
                
                found = analysis.get('found', False)
                count = analysis.get('count', 0)
                instances = analysis.get('instances', [])
                
                # Generate appropriate response based on findings
                if not found or count == 0:
                    response_text = f"I scanned the entire room but couldn't find your {target_object}. It's not visible from my current location."
                    
                elif count == 1:
                    instance = instances[0]
                    direction = instance.get('direction', 'unknown')
                    description = instance.get('description', 'nearby')
                    response_text = f"I found your {target_object}. It's {description}, which is to my {direction}."
                    
                else:  # Multiple instances
                    response_text = f"I found {count} {target_object}s in the room. "
                    for i, instance in enumerate(instances, 1):
                        direction = instance.get('direction', 'unknown')
                        description = instance.get('description', 'there')
                        response_text += f"The {i}{self._ordinal_suffix(i)} one is {description} to my {direction}. "
                    response_text += f"Which {target_object} is yours?"
                
                return {
                    'found': found,
                    'response': response_text,
                    'count': count,
                    'locations': instances
                }
            
            else:
                print("Could not parse JSON from VLM response")
                return {
                    'found': False,
                    'response': f"Sorry, I had trouble analyzing the images to find your {target_object}.",
                    'count': 0,
                    'locations': []
                }
                
        except Exception as e:
            print(f"Error in find_object_in_images: {e}")
            import traceback
            traceback.print_exc()
            return {
                'found': False,
                'response': f"Sorry, I encountered an error while searching for your {target_object}.",
                'count': 0,
                'locations': []
            }
    
    def _ordinal_suffix(self, n):
        """Return ordinal suffix for a number (st, nd, rd, th)"""
        if 10 <= n % 100 <= 20:
            return 'th'
        return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    
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