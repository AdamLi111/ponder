"""
Zero-shot VLM with Multi-turn Dialogue (Baseline 2)
Maintains conversation history but no friction
User can provide follow-ups/corrections but robot doesn't ask questions
"""
from openai import OpenAI
import json


class LLMLayerZeroShotMultiTurn:
    """
    Zero-shot VLM with multi-turn dialogue baseline
    - Maintains conversation history (user can clarify)
    - No positive friction (never proactively asks questions)
    - Can understand context from previous turns
    """
    
    def __init__(self, openai_api_key, logger=None):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.vision_model = "gpt-5-nano-2025-08-07"
        self.conversation_history = []  # Maintains history
        self.logger = logger
        print("Zero-shot VLM with Multi-turn enabled (with history, no friction)")
    
    def parse_intent_with_vision(self, user_speech, image_data_base64=None):
        """
        Multi-turn parsing: Uses conversation history but no friction
        
        Args:
            user_speech: Combined vision context + user command
            image_data_base64: Not used (vision from text)
            
        Returns:
            dict: Action to execute (never returns 'clarify')
        """
        print(f"Multi-turn zero-shot parsing: '{user_speech[:100]}...'")
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": f"User said: \"{user_speech}\""
        })
        
        # Build system prompt (no friction, but understands context)
        system_prompt = """You are Misty, a robot that executes navigation commands.

# IMPORTANT: Multi-turn mode (NO FRICTION)
- Maintain conversation context from previous turns
- Understand follow-up commands ("the red one", "go left instead")
- But NEVER ask clarifying questions proactively
- Execute immediately based on available information

# Actions: forward, backward, left, right, turn_left, turn_right, stop, describe_vision, find_object, spatial_navigate, speak, unknown

# Response Format:
Return JSON with:
- action: the action to take
- distance: for movement actions
- turn_degrees: for turn actions
- target_object: for spatial_navigate/find_object
- text: optional statement
- confidence: "high", "medium", or "low"

# Rules:
1. NEVER use action "clarify" - no proactive questions
2. Use conversation history to resolve references
3. If user provides clarification, update your understanding
4. If ambiguous, make best guess with available context
5. For safety concerns, choose conservative actions
6. Always return an executable action"""

        try:
            # Build messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history (limited to last 10 turns)
            recent_history = self.conversation_history[-10:]
            for msg in recent_history:
                messages.append(msg)
            
            # Call API
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=4096
            )
            
            output = response.choices[0].message.content.strip()
            print(f"Multi-turn zero-shot output: '{output}'")
            
            intent = json.loads(output)
            
            # Store assistant response in history
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Action: {intent.get('action')}"
            })
            
            # Ensure it never returns clarify action
            if intent.get('action') == 'clarify':
                print("WARNING: Multi-turn zero-shot tried to clarify - forcing execution")
                # Convert to executable action
                if intent.get('target_object'):
                    intent['action'] = 'spatial_navigate'
                    intent['distance'] = 2.0
                else:
                    intent['action'] = 'forward'
                    intent['distance'] = 1.0
            
            # Log if enabled
            if self.logger:
                self.logger.log_llm_call(user_speech, intent)
            
            return intent
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {
                "action": "unknown",
                "text": "Failed to parse command",
                "confidence": "low"
            }
        except Exception as e:
            print(f"Error in multi-turn zero-shot: {e}")
            return {
                "action": "unknown",
                "text": str(e),
                "confidence": "low"
            }
    
    def reset_conversation(self):
        """Reset conversation history (for new task)"""
        self.conversation_history = []