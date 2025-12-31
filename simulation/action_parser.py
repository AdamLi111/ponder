"""
Action Parser Module
Parses robot VLM JSON outputs into human-readable action descriptions
"""


class ActionParser:
    """
    Parses robot VLM JSON outputs into human-readable action descriptions
    """
    
    @staticmethod
    def parse_action(vlm_response):
        """
        Convert VLM JSON response into description of what robot did/will do
        
        Returns:
            str: Human-readable description of robot's action
        """
        action = vlm_response.get('action')
        
        if action == 'clarify':
            # Robot asked a question, extract it
            question = vlm_response.get('text', 'asked for clarification')
            return f"Misty asked: '{question}'"
        
        if action == 'find_object':
            target = vlm_response.get('target_object', 'object')
            return f"Misty will perform 360° scan to find {target}"
        
        if action == 'describe_vision':
            return "Misty described what it sees"
        
        if action == 'speak':
            text = vlm_response.get('text', '')
            return f"Misty said: '{text}'"
        
        # Check for multi-action sequence
        if 'actions' in vlm_response:
            actions_list = vlm_response['actions']
            descriptions = []
            for act in actions_list:
                act_type = act.get('action')
                if act_type == 'forward':
                    dist = act.get('distance', 0)
                    descriptions.append(f"moved forward {dist}m")
                elif act_type == 'backward':
                    dist = act.get('distance', 0)
                    descriptions.append(f"moved backward {dist}m")
                elif act_type == 'turn_left':
                    degrees = act.get('turn_degrees', 0)
                    descriptions.append(f"turned left {degrees}°")
                elif act_type == 'turn_right':
                    degrees = act.get('turn_degrees', 0)
                    descriptions.append(f"turned right {degrees}°")
            
            if descriptions:
                statement = vlm_response.get('text', '')
                result = "Misty executed: " + ", then ".join(descriptions)
                if statement:
                    result = f"Misty said '{statement}', then: " + ", then ".join(descriptions)
                return result
        
        # Single action
        if action == 'forward':
            dist = vlm_response.get('distance', 0)
            return f"Misty moved forward {dist} meters"
        
        if action == 'backward':
            dist = vlm_response.get('distance', 0)
            return f"Misty moved backward {dist} meters"
        
        if action == 'turn_left':
            degrees = vlm_response.get('turn_degrees', 0)
            return f"Misty turned left {degrees} degrees"
        
        if action == 'turn_right':
            degrees = vlm_response.get('turn_degrees', 0)
            return f"Misty turned right {degrees} degrees"
        
        if action == 'spatial_navigate':
            target = vlm_response.get('target_object', 'object')
            dist = vlm_response.get('distance', 0)
            degrees = vlm_response.get('turn_degrees', 0)
            return f"Misty navigated to {target} (adjusted {degrees}°, moved {dist}m)"
        
        if action == 'stop':
            return "Misty stopped"
        
        return f"Misty executed action: {action}"