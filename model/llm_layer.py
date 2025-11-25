"""
LLM layer for intent parsing and vision description
Uses Groq API for text/intent parsing and Gemini for vision
"""
from groq import Groq
from google import genai
import json


class LLMLayer:
    def __init__(self, groq_api_key, gemini_api_key=None):
        # Groq for text/intent parsing
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_model = "llama-3.1-8b-instant"
        self.conversation_history = []
        
        # Gemini for vision (optional)
        self.gemini_client = None
        self.gemini_model = "gemini-2.0-flash-exp"
        if gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key)
            print("Vision enabled with Gemini Flash")
        else:
            print("Warning: No Gemini API key provided - vision will be disabled")
    
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
2. Extract the action type: forward, backward, left, right, stop, describe_vision, speak, clarify, or unknown
3. Extract distance in meters (default to 1 if not specified, 0 for non-movement commands)
4. Extract any text to speak
5. Determine if you should add conversational friction before executing
6. USE CONVERSATION HISTORY to understand context and references like "it", "there", "yes", etc.
7. **CRITICAL: Detect action sequences** - if the user says commands like "move forward 1 meter and then go right 1 meter" or "go left then forward", return MULTIPLE actions

# Action Sequence Detection:
- Look for words like "and then", "then", "after that", "next", "followed by"
- Return an array of actions for sequences
- Each action in the sequence should have its own friction_type, action, distance, and text

# Positive Friction Types:

**Probing (clarify)** - Use when:
- Ambiguous spatial references (e.g., "over there", "that side") WITH NO PRIOR CONTEXT
- Unclear object references (e.g., "it", "that thing") WITH NO PRIOR CONTEXT
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

**For SINGLE actions:**
Return a JSON object with these fields:
{"friction_type": "none", "action": "forward", "distance": 2, "text": ""}

**For ACTION SEQUENCES:**
Return a JSON object with an "actions" array:
{"actions": [
  {"friction_type": "none", "action": "forward", "distance": 1, "text": ""},
  {"friction_type": "none", "action": "right", "distance": 1, "text": ""}
]}

# Examples:

**Simple single command (no friction):**
User: "move forward 2 meters"
{"friction_type": "none", "action": "forward", "distance": 2, "text": ""}

**Action sequence:**
User: "move forward 1 meter and then go right 1 meter"
{"actions": [
  {"friction_type": "none", "action": "forward", "distance": 1, "text": ""},
  {"friction_type": "none", "action": "right", "distance": 1, "text": ""}
]}

**Complex sequence:**
User: "go left 2 meters then forward 1 meter then turn right"
{"actions": [
  {"friction_type": "none", "action": "left", "distance": 2, "text": ""},
  {"friction_type": "none", "action": "forward", "distance": 1, "text": ""},
  {"friction_type": "none", "action": "turn_right", "distance": 90, "text": ""}
]}

**Sequence with confirmation friction (long sequence):**
User: "move forward 3 meters then left 2 meters then backward 1 meter then right 2 meters"
{"friction_type": "overspecify", "action": "speak", "distance": 0, "text": "I'll execute 4 movements: forward 3m, left 2m, backward 1m, then right 2m. Say confirm to proceed."}

**Ambiguous reference (probing):**
User: "move toward that"
{"friction_type": "probing", "action": "clarify", "distance": 0, "text": "Which object are you referring to? Could you be more specific about where you'd like me to move?"}

**Follow-up with context:**
Previous: Asked user to clarify which chair
User: "yes the one in front of you"
{"friction_type": "assumption_reveal", "action": "forward", "distance": 1, "text": "Moving toward the chair directly in front of me"}

**Confirmation response:**
Previous: Asked if user meant object X
User: "yes"
{"friction_type": "none", "action": "forward", "distance": 1, "text": ""}"""
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
                parsed = json.loads(json_str)
                
                # Check if it's a sequence or single action
                if "actions" in parsed:
                    # It's a sequence
                    actions = parsed["actions"]
                    print(f"Parsed {len(actions)} actions in sequence")
                    return actions
                else:
                    # Single action - wrap in list for consistent handling
                    action = parsed.get('action', 'unknown')
                    distance = parsed.get('distance', 1.0)
                    friction_type = parsed.get('friction_type', 'none')
                    text = parsed.get('text', '')
                    
                    print(f"Parsed intent - Action: {action}, Distance: {distance}, Text: {text}")
                    return [{
                        'action': action,
                        'distance': distance,
                        'friction_type': friction_type,
                        'text': text
                    }]
            else:
                print("Could not find JSON in response")
                return [{'action': 'unknown', 'distance': 0, 'friction_type': 'none', 'text': ''}]
                
        except Exception as e:
            print(f"Error calling Groq LLM: {e}")
            import traceback
            traceback.print_exc()
            return [{'action': 'unknown', 'distance': 0, 'friction_type': 'none', 'text': ''}]
    
    def clear_conversation_history(self):
        """Clear conversation history (useful for starting fresh)"""
        self.conversation_history = []
        print("Conversation history cleared")
    
    def describe_image(self, image_data_base64):
        """Get description of an image using Gemini Vision"""
        if not self.gemini_client:
            print("Error: No Gemini API key provided - cannot describe images")
            return "Vision is not available. Please provide a Gemini API key."
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=[
                    "Describe what you see in this image in detail. Be specific about objects, people, and the setting.",
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
            print(f"Error in describe_image: {e}")
            import traceback
            traceback.print_exc()
            return "Sorry, I couldn't analyze the image."