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
- "go to chair" + chair visible but trash can blocking, no edges → {"friction_type":"probing","action":"clarify","text":"I see the chair behind a trash can. Should I go around it or stop?","clarification_needed":"I see the chair behind a trash can. Should I go around it or stop?","confidence":"medium"}

## Autonomous Navigation (Multi-action):
- "go around it" + image shows LEFT side clear, right blocked, no edge hazards → {"friction_type":"assumption_reveal","text":"I'll navigate around the left side where there's more clearance","confidence":"high","actions":[{"action":"turn_left","turn_degrees":30},{"action":"forward","distance":1.5},{"action":"turn_right","turn_degrees":30},{"action":"forward","distance":1.0}]}
- "go around it" + image shows RIGHT side clear, left blocked, no edge hazards → {"friction_type":"assumption_reveal","text":"I'll navigate around the right side where there's more clearance","confidence":"high","actions":[{"action":"turn_right","turn_degrees":25},{"action":"forward","distance":1.3},{"action":"turn_left","turn_degrees":50},{"action":"forward","distance":0.9}]}
- "go around the box" + both sides clear, RIGHT has more space, no edges → {"friction_type":"none","text":"Navigating around the right side","confidence":"high","actions":[{"action":"turn_right","turn_degrees":30},{"action":"forward","distance":1.4},{"action":"turn_left","turn_degrees":55},{"action":"forward","distance":0.8}]}
- "go around the obstacle" + both sides clear, LEFT has more space, no edges → {"friction_type":"none","text":"Navigating around the left side","confidence":"high","actions":[{"action":"turn_left","turn_degrees":28},{"action":"forward","distance":1.3},{"action":"turn_right","turn_degrees":52},{"action":"forward","distance":0.9}]}
- "go around it to the left" + user explicitly specifies left, no edge hazards → {"friction_type":"none","text":"Going around the left side","confidence":"high","actions":[{"action":"turn_left","turn_degrees":25},{"action":"forward","distance":1.2},{"action":"turn_right","turn_degrees":50},{"action":"forward","distance":0.8}]}

## Clear Path Navigation:
- "go to plant" + one plant, clear path, no edge hazards → {"friction_type":"none","action":"spatial_navigate","target_object":"plant","distance":2.0,"turn_degrees":5,"text":"","clarification_needed":null,"confidence":"high"}

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