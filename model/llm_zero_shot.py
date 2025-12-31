"""
Zero-shot VLM Layer (Baseline 1)
Pure zero-shot: No conversation history, no friction
Each command processed independently
"""
from openai import OpenAI
import json


class LLMLayerZeroShot:
    """
    Zero-shot VLM baseline
    - No conversation history (stateless)
    - No positive friction (never asks clarifying questions)
    - Processes each command independently
    """
    
    def __init__(self, openai_api_key, logger=None):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.vision_model = "gpt-5-nano-2025-08-07"
        # NO conversation history - pure zero-shot
        self.logger = logger
        print("Zero-shot VLM enabled (no history, no friction)")
    
    def parse_intent_with_vision(self, user_speech, image_data_base64=None):
        """
        Zero-shot parsing: No conversation history, direct execution only
        
        Args:
            user_speech: Combined vision context + user command
            image_data_base64: Not used in zero-shot (vision from text)
            
        Returns:
            dict: Action to execute (never returns 'clarify')
        """
        print(f"Zero-shot parsing (no history): '{user_speech[:100]}...'")
        
        # Build zero-shot system prompt (no friction, no history)
        system_prompt = """You are Misty, a robot that executes navigation commands.

# IMPORTANT: You are in ZERO-SHOT mode
- NO conversation history (process each command independently)
- NO clarifying questions (never use action: "clarify")
- Make best guess and execute immediately

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
1. NEVER use action "clarify" - you must execute immediately
2. If ambiguous, make the most reasonable assumption
3. If multiple objects match, go to the nearest/most obvious one
4. If unsafe, choose a safe alternative distance
5. Always return an executable action"""

        try:
            # Single-turn call - no conversation history
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_speech}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=4096
            )
            
            output = response.choices[0].message.content.strip()
            print(f"Zero-shot raw output: '{output}'")
            
            intent = json.loads(output)
            
            # Ensure it never returns clarify action
            if intent.get('action') == 'clarify':
                print("WARNING: Zero-shot model tried to clarify - forcing execution")
                # Convert to spatial_navigate or forward
                if intent.get('target_object'):
                    intent['action'] = 'spatial_navigate'
                    intent['distance'] = 2.0  # Default distance
                else:
                    intent['action'] = 'forward'
                    intent['distance'] = 1.0
            
            # Log if enabled
            if self.logger:
                self.logger.log_llm_call(user_speech, intent)
            
            return intent
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error in zero-shot: {e}")
            return {
                "action": "unknown",
                "text": "Failed to parse command",
                "confidence": "low"
            }
        except Exception as e:
            print(f"Error in zero-shot VLM: {e}")
            return {
                "action": "unknown", 
                "text": str(e),
                "confidence": "low"
            }