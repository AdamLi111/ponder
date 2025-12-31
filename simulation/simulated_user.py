"""
Simulated User Module
LLM-based agent that simulates user behavior
Has access to complete world state (omniscient view)
Starts with VAGUE commands, disambiguates when robot asks clarifications
NOW: Tracks goal progress and issues follow-up commands for multi-step tasks
Returns None when goal is fully achieved (no further commands)
"""
from openai import OpenAI


class SimulatedUser:
    """
    LLM-based agent that simulates user behavior
    Has access to COMPLETE world state (omniscient view)
    Starts with VAGUE commands, disambiguates when robot asks clarifications
    Tracks multi-step goal progress and issues follow-ups
    """
    
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = "gpt-4o-mini"
        self.conversation_history = []
        self.actions_observed = []
        self.world_model = None
        self.goal_steps_completed = []
    
    def reset(self, world_model):
        """Reset for a new task with world model"""
        self.world_model = world_model
        self.conversation_history = []
        self.actions_observed = []
        self.goal_steps_completed = []
        
        world_state = world_model.get_full_state_description()
        
        self.system_prompt = f"""You are a human user giving commands to a robot named Misty.

{world_state}

YOUR GOAL: {world_model.task_goal}

IMPORTANT: You can SEE the entire world state above (you're omniscient like a player in a text game), but Misty can only see what's in front of it through its forward-facing camera.

MULTI-STEP GOAL AWARENESS:
- Your goal may have MULTIPLE parts (e.g., "navigate to X AND check Y" or "go to X and find Y")
- After each robot action, check if ALL parts of your goal are complete
- If only SOME parts are done, issue the NEXT command for remaining parts
- Common patterns:
  * "Go to the left/middle/right/front/back X" → "Go to X"
  * "Navigate to X and report/check Y" → First "go to X", then "check Y" or "is Y...?"
  * "Go to X and find Y" → First "go to X", then "find the Y" or "look for Y"
  * "Navigate to X and count Y" → First "go to X", then "how many Y are there?"

Output ONLY the user's speech, nothing else."""
        
    def generate_initial_command(self):
        """Generate VAGUE initial command for FIRST part of goal"""
        
        prompt = f"""Based on your goal, give a SIMPLE, VAGUE first command to the robot.

Your goal: {self.world_model.task_goal}

Rules:
- Mention ONLY the basic action and target object
- Do NOT include ANY description about the target object(eg., color, direction, location, etc.)
- If goal has multiple parts (e.g., "go to X and check Y"), only command the FIRST part now
- Keep it under 8 words

Examples:
- Goal "Go to the plant on the left/right/middle/front/back" → "Go over to the plant"
- Goal "Find my blue marker" → "Find my marker"
- Goal "Move forward 3 meters" → "Move forward"
- Goal "Go to middle box" → "Go to the box"
- Goal "Go to the office and find my bag" → "Go to the office"

Your vague initial command:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=50
        )
        
        command = response.choices[0].message.content.strip()
        command = command.replace('"', '').replace("'", "")
        
        self.conversation_history.append({
            "role": "user", 
            "content": command
        })
        return command
    
    def observe_robot_action(self, action_description):
        """User observes what the robot did"""
        self.actions_observed.append(action_description)
    
    def check_goal_progress(self):
        """
        Analyze what parts of the goal have been completed
        Returns: (all_complete: bool, remaining_steps: list)
        FIXED: Better detection for directional navigation (behind/front)
        FIXED: General areas (kitchen, office) don't require specific navigation
        """
        goal = self.world_model.task_goal.lower()
        actions = [a.lower() for a in self.actions_observed]
        actions_str = " ".join(actions)
        
        remaining = []
        
        # General areas that don't require specific navigation
        general_areas = ["kitchen", "office", "living room", "bedroom", "bathroom", 
                        "hallway", "garage", "room", "area"]
        
        # Check for navigation component
        nav_keywords = ["navigate to", "go to", "move to", "reach"]
        has_nav_goal = any(k in goal for k in nav_keywords)
        
        # Check if navigation target is a general area (already there)
        nav_is_general_area = False
        if has_nav_goal:
            for area in general_areas:
                if f"to {area}" in goal or f"to the {area}" in goal:
                    nav_is_general_area = True
                    break
        
        # For directional goals (behind/front), check if robot actually moved in that direction
        has_behind_goal = "behind" in goal
        has_front_goal = "in front" in goal or "ahead" in goal
        
        nav_done = False
        if has_nav_goal:
            if nav_is_general_area:
                # General area - consider navigation done (we're already there)
                nav_done = True
            else:
                # Check basic navigation actions
                basic_nav_done = any(k in actions_str for k in ["navigated", "moved forward", "moved backward", "reached", "turned"])
                
                if has_behind_goal:
                    nav_done = "moved backward" in actions_str or "navigated" in actions_str
                elif has_front_goal:
                    nav_done = "moved forward" in actions_str or "navigated" in actions_str
                else:
                    nav_done = basic_nav_done
        
        if has_nav_goal and not nav_done:
            remaining.append("navigation")
        
        # Check for perceptual/report component
        report_keywords = ["report", "check", "describe", "count", "find", "look"]
        has_report_goal = any(k in goal for k in report_keywords)
        
        # Check if report/observation is done
        report_done = False
        
        # Method 1: Explicit perceptual action keywords
        explicit_done_keywords = ["described", "counted", "found", "checked", "reported", "360° scan"]
        if any(k in actions_str for k in explicit_done_keywords):
            report_done = True
        
        # Method 2: Robot spoke an answer (e.g., "Misty said: 'There are 4 chairs'")
        if not report_done:
            # Number words that count as answering a "count" question
            number_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", 
                           "eight", "nine", "ten", "eleven", "twelve", "no", "none", "several", "many"]
            
            for action in actions:
                if "misty said:" in action:
                    # For count tasks: look for digits OR number words
                    if "count" in goal:
                        has_digit = any(c.isdigit() for c in action)
                        has_number_word = any(nw in action.lower() for nw in number_words)
                        if has_digit or has_number_word:
                            report_done = True
                            break
                    # For check tasks: look for yes/no or status words
                    if "check" in goal and any(w in action for w in ["yes", "no", "is", "plugged", "power"]):
                        report_done = True
                        break
                    # For report tasks: any substantive speech counts
                    if "report" in goal and len(action) > 20:
                        report_done = True
                        break
                    # For describe tasks: any substantive speech counts
                    if "describe" in goal and len(action) > 20:
                        report_done = True
                        break
        
        if has_report_goal and not report_done:
            remaining.append("observation")
        
        all_complete = len(remaining) == 0
        return all_complete, remaining
    
    def respond_to_robot(self, robot_message, robot_action_description=None, task_complete=False):
        """
        Generate response to robot's question or after observing robot's action
        
        Returns:
            str: User's next command/response
            None: If goal is fully achieved (signals task completion, no further commands)
        """
        
        # Check goal progress
        all_complete, remaining_steps = self.check_goal_progress()
        
        # If explicitly told task is complete OR all goal parts done → return None
        # This signals to simulator that task is done, no more commands needed
        if task_complete or all_complete:
            return None  # Signal completion - don't say anything
        
        self.conversation_history.append({
            "role": "assistant", 
            "content": robot_message
        })
        
        world_state = self.world_model.get_full_state_description()
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
        messages.append({
            "role": "user",
            "content": f"=== UPDATED WORLD STATE ===\n{world_state}"
        })
        
        messages.append({
            "role": "user", 
            "content": "=== CONVERSATION SO FAR ==="
        })
        
        for msg in self.conversation_history:
            if msg["role"] == "user":
                messages.append({
                    "role": "assistant", 
                    "content": f"You said: \"{msg['content']}\""
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Robot: \"{msg['content']}\""
                })
        
        if robot_action_description:
            messages.append({
                "role": "user",
                "content": f"\nYou observed: {robot_action_description}"
            })
        
        if self.actions_observed:
            messages.append({
                "role": "user",
                "content": f"\nAll robot actions so far: {', '.join(self.actions_observed)}"
            })
        
        # Add goal progress hint for remaining steps
        if remaining_steps:
            step_hints = {
                "observation": "The robot has moved but hasn't done the observation/check/find part yet. Ask it to do that now.",
                "navigation": "The robot hasn't reached the destination yet."
            }
            hints = [step_hints.get(s, "") for s in remaining_steps if s in step_hints]
            if hints:
                messages.append({
                    "role": "user",
                    "content": f"\n⚠️ GOAL PROGRESS: {' '.join(hints)}"
                })
        
        messages.append({
            "role": "user",
            "content": "\nHow do you respond? (Be brief, 3-10 words. If robot completed navigation but goal requires observation/check/find, issue that command now.)"
        })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            max_tokens=80
        )
        
        user_response = response.choices[0].message.content.strip()
        
        # Clean up
        user_response = user_response.replace("User:", "").replace('"', '').strip()
        user_response = user_response.replace("You said:", "").strip()
        user_response = user_response.replace("I respond:", "").strip()
        
        if user_response.lower().startswith("you said:"):
            user_response = user_response[9:].strip()
        
        self.conversation_history.append({
            "role": "user", 
            "content": user_response
        })
        
        return user_response