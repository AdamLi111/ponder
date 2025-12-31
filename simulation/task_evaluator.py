"""
Task Success Evaluator
Evaluates whether robot achieved the task goal based on final world state
Uses exact coordinate-based position checking with structured scenes
FIXED: Better perceptual action detection and smarter target extraction
FIXED: Collision detection causes immediate failure
UPDATED: Collision with obstacle is logged as explicit failure reason
"""
import re
import math


class TaskEvaluator:
    """
    Evaluates task success by comparing final robot state against task goal
    """
    
    @staticmethod
    def evaluate_task_success(world_model, task_goal, interaction_log):
        """
        Determine if task was successfully completed
        UPDATED: Check for collision first - collision causes immediate failure
        """
        # Check for collision first - this overrides all other conditions
        if interaction_log.get("collision") is not None:
            collision_info = interaction_log["collision"]
            return {
                "success": False,
                "goal_conditions_met": 0,
                "total_goal_conditions": 1,  # At minimum, "don't collide" is a condition
                "success_rate": 0.0,
                "failure_reason": f"collision with obstacle: {collision_info.get('obstacle_name', 'unknown')}",
                "collision": True,
                "collision_details": collision_info,
                "metrics": {
                    "final_position": world_model.robot_position,
                    "final_orientation": world_model.robot_orientation,
                    "total_actions": len(world_model.action_history)
                }
            }
        
        goal_conditions = TaskEvaluator._parse_goal_conditions(task_goal)
        
        conditions_met = 0
        total_conditions = len(goal_conditions)
        failure_reasons = []
        
        for condition in goal_conditions:
            is_met, reason = TaskEvaluator._check_condition(
                condition, 
                world_model,
                interaction_log  # Pass interaction log for content checking
            )
            if is_met:
                conditions_met += 1
            else:
                failure_reasons.append(reason)
        
        success = (conditions_met == total_conditions)
        success_rate = conditions_met / total_conditions if total_conditions > 0 else 0.0
        
        return {
            "success": success,
            "goal_conditions_met": conditions_met,
            "total_goal_conditions": total_conditions,
            "success_rate": success_rate,
            "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
            "collision": False,
            "metrics": {
                "final_position": world_model.robot_position,
                "final_orientation": world_model.robot_orientation,
                "total_actions": len(world_model.action_history)
            }
        }
    
    @staticmethod
    def _parse_goal_conditions(task_goal):
        """Extract specific conditions from task goal"""
        conditions = []
        goal_lower = task_goal.lower()
        
        # Perceptual tasks: describe, count, check status, report
        if "describe" in goal_lower or "report" in goal_lower or "check" in goal_lower or "count" in goal_lower:
            conditions.append({
                "type": "perceptual_task",
                "goal_text": goal_lower  # Pass full goal for context
            })
            # May also have navigation component
            if "navigate to" in goal_lower or "go to" in goal_lower:
                target = TaskEvaluator._extract_target_from_goal(goal_lower)
                if target:
                    conditions.append({
                        "type": "navigate_to_object",
                        "target": target,
                        "distance_threshold": 2.0
                    })
            return conditions
        
        # Find object tasks
        if "find" in goal_lower:
            target_match = re.search(r'find.*?(phone|keys|cup|bottle|bag|book|box)', goal_lower)
            if target_match:
                conditions.append({
                    "type": "find_object",
                    "target": target_match.group(1)
                })
            return conditions
        
        # Navigate to object
        if "navigate to" in goal_lower or "go to" in goal_lower:
            target = TaskEvaluator._extract_target_from_goal(goal_lower)
            if target:
                distance_threshold = 0.8
                if "very close" in goal_lower or "within" in goal_lower:
                    dist_match = re.search(r'within\s+(\d+\.?\d*)\s*m', goal_lower)
                    if dist_match:
                        distance_threshold = float(dist_match.group(1))
                    else:
                        distance_threshold = 0.5
                
                conditions.append({
                    "type": "navigate_to_object",
                    "target": target,
                    "distance_threshold": distance_threshold
                })
        
        # Move forward/backward specific distance
        elif "move forward" in goal_lower or "move backward" in goal_lower:
            direction = "forward" if "forward" in goal_lower else "backward"
            dist_match = re.search(r'(\d+\.?\d*)\s*meter', goal_lower)
            if dist_match:
                target_distance = float(dist_match.group(1))
                conditions.append({
                    "type": "move_distance",
                    "direction": direction,
                    "target_distance": target_distance,
                    "tolerance": 0.3
                })
            else:
                conditions.append({
                    "type": "move_distance",
                    "direction": direction,
                    "target_distance": 0.5,
                    "tolerance": 0.5
                })
        
        # Turn around
        elif "turn around" in goal_lower or "180" in goal_lower:
            conditions.append({
                "type": "turn_to_orientation",
                "target_orientation": 180,
                "tolerance": 30
            })
        
        # Default fallback
        if not conditions:
            target = TaskEvaluator._extract_target_from_goal(goal_lower)
            if target:
                conditions.append({
                    "type": "navigate_to_object",
                    "target": target,
                    "distance_threshold": 0.5
                })
        
        return conditions
    
    @staticmethod
    def _extract_target_from_goal(goal_lower):
        """
        Extract target object from goal description
        FIXED: Check specific named targets FIRST before positional patterns
        FIXED: Handle directional qualifiers like "behind", "in front of"
        """
        # PRIORITY 1: Check for specific named targets first (color + object, etc.)
        # These take precedence over positional extraction
        specific_targets = [
            # Color-qualified objects
            "red cup", "blue cup", "green cup", "white cup",
            "red book", "blue book", "green book", 
            "red mug", "white mug",
            # Position-qualified objects
            "left bottle", "center bottle", "middle bottle", "right bottle",
            "left chair", "right chair",
            "left door", "right door",
            "left trash bin", "center trash bin", "right trash bin",
            # Size-qualified objects
            "medium box", "small box", "large box",
            # Directional objects
            "back plant", "front plant", "back door", "front door",
        ]
        
        for target in specific_targets:
            if target in goal_lower:
                return target
        
        # PRIORITY 2: Directional patterns (behind/in front of)
        # "the plant behind you" -> "back plant"
        # "the chair in front of you" -> "front chair"
        directional_patterns = [
            # "the X behind you/me" -> "back X"
            (r'the\s+(\w+)\s+behind\s+(?:you|me|the robot)', lambda m: f"back {m.group(1)}"),
            # "X behind you" (without "the")
            (r'(\w+)\s+behind\s+(?:you|me|the robot)', lambda m: f"back {m.group(1)}"),
            # "the X in front of you/me" -> "front X"
            (r'the\s+(\w+)\s+in\s+front\s+(?:of\s+)?(?:you|me|the robot)', lambda m: f"front {m.group(1)}"),
            # "to the X behind" -> "back X"
            (r'to\s+the\s+(\w+)\s+behind', lambda m: f"back {m.group(1)}"),
            # "behind" alone with object mentioned elsewhere
            (r'(\w+).*behind', lambda m: f"back {m.group(1)}"),
        ]
        
        filler_words = ["one", "it", "that", "this", "thing", "go", "navigate", "move"]
        
        for pattern, extractor in directional_patterns:
            match = re.search(pattern, goal_lower)
            if match:
                extracted = extractor(match)
                obj_word = extracted.split()[-1] if extracted.split() else ""
                if obj_word not in filler_words:
                    return extracted
        
        # PRIORITY 3: Positional patterns for phrases like "door on the right side"
        # First, try to extract with special handling for adjective + noun patterns
        # e.g., "the middle water bottle" should extract "middle bottle", not "middle water"
        
        # Pattern for "the [position] [adjective] [noun]" - e.g., "the middle water bottle"
        adj_noun_pattern = r'the\s+(left|right|center|middle)\s+(\w+)\s+(\w+)'
        adj_noun_match = re.search(adj_noun_pattern, goal_lower)
        if adj_noun_match:
            position = adj_noun_match.group(1)
            word2 = adj_noun_match.group(2)  # Could be adjective or noun
            word3 = adj_noun_match.group(3)  # Likely the noun
            
            # Common adjectives that appear before object nouns
            common_adjectives = ["water", "plastic", "glass", "metal", "wooden", "small", "large", 
                                "big", "old", "new", "red", "blue", "green", "white", "black"]
            
            # If word2 is an adjective, use word3 as the object
            if word2 in common_adjectives:
                extracted = f"{position} {word3}"
            else:
                extracted = f"{position} {word2}"
            
            extracted = extracted.replace("middle", "center")
            if extracted.split()[-1] not in filler_words:
                return extracted
        
        positional_patterns = [
            # "the door on the right side" -> "right door"
            (r'the\s+(\w+)\s+on\s+the\s+(left|right|center|middle)\s*(?:side)?', lambda m: f"{m.group(2)} {m.group(1)}"),
            # "the left/right door" -> "left door" / "right door"  
            (r'the\s+(left|right|center|middle)\s+(\w+)', lambda m: f"{m.group(1)} {m.group(2)}"),
            # "door to the left/right" -> "left door" / "right door"
            (r'(\w+)\s+to\s+the\s+(left|right)', lambda m: f"{m.group(2)} {m.group(1)}"),
        ]
        
        for pattern, extractor in positional_patterns:
            match = re.search(pattern, goal_lower)
            if match:
                extracted = extractor(match)
                # Normalize "middle" to "center"
                extracted = extracted.replace("middle", "center")
                
                # Skip if extracted object is a filler word (e.g., "left one")
                obj_word = extracted.split()[-1] if extracted.split() else ""
                if obj_word in filler_words:
                    continue  # Try next pattern or fall through to generic
                
                return extracted
        
        # PRIORITY 4: Generic objects
        generic_objects = [
            "cup", "bottle", "plant", "book", "laptop", "keys", 
            "phone", "door", "chair", "desk", "table", "fridge",
            "trash bin", "box", "bag", "kitchen", "office", "mug"
        ]
        
        for obj in generic_objects:
            if obj in goal_lower:
                return obj
        
        return None
    
    @staticmethod
    def _check_condition(condition, world_model, interaction_log=None):
        """Check if a specific condition is met"""
        cond_type = condition["type"]
        
        if cond_type == "navigate_to_object":
            return TaskEvaluator._check_navigation_to_object(condition, world_model)
        
        elif cond_type == "move_distance":
            return TaskEvaluator._check_move_distance(condition, world_model)
        
        elif cond_type == "turn_to_orientation":
            return TaskEvaluator._check_orientation(condition, world_model)
        
        elif cond_type == "find_object":
            return TaskEvaluator._check_find_object(condition, world_model)
        
        elif cond_type == "perceptual_task":
            return TaskEvaluator._check_perceptual_task(condition, world_model, interaction_log)
        
        # Legacy support
        elif cond_type == "describe_vision":
            return TaskEvaluator._check_perceptual_task(condition, world_model, interaction_log)
        
        return False, "Unknown condition type"
    
    @staticmethod
    def _check_navigation_to_object(condition, world_model):
        """
        Check if robot successfully navigated to the target object
        Uses exact coordinates for accurate position checking
        FIXED: Handles directional qualifiers (back/front) by checking properties and position
        """
        target = condition["target"]
        threshold = condition["distance_threshold"]
        
        target_obj = None
        target_lower = target.lower()
        
        # Check for directional qualifiers
        directional_qualifiers = {
            "back": ["back", "behind"],
            "front": ["front", "ahead", "forward"]
        }
        
        target_direction = None
        target_object_type = None
        
        for direction, keywords in directional_qualifiers.items():
            for kw in keywords:
                if kw in target_lower:
                    target_direction = direction
                    # Extract object type (e.g., "back plant" -> "plant")
                    target_object_type = target_lower.replace(kw, "").strip()
                    break
            if target_direction:
                break
        
        for obj in world_model.objects:
            obj_name_lower = obj["name"].lower()
            obj_props = obj.get("properties", {})
            obj_pos = obj["position"]
            
            # Case 1: Directional target (e.g., "back plant")
            if target_direction and target_object_type:
                # Check if object type matches
                if target_object_type not in obj_name_lower and obj_name_lower not in target_object_type:
                    continue
                
                # Check if direction matches via properties
                obj_id = str(obj_props.get("id", "")).lower()
                if obj_id == target_direction:
                    target_obj = obj
                    break
                
                # Check if direction matches via position (back = negative y, front = positive y)
                # Relative to initial robot position (0, 0)
                if target_direction == "back" and obj_pos[1] < 0:
                    target_obj = obj
                    break
                elif target_direction == "front" and obj_pos[1] > 0:
                    # Only match front if there isn't a "back" object we should prefer
                    if target_obj is None:
                        target_obj = obj
                        # Don't break - keep looking for better match
            
            # Case 2: Exact match
            elif obj_name_lower == target_lower:
                target_obj = obj
                break
            
            # Case 3: Check if target specifies position (left/right/center)
            elif any(q in target_lower for q in ["left", "center", "middle", "right"]):
                position_qualifiers = ["left", "center", "middle", "right"]
                for qual in position_qualifiers:
                    if qual in target_lower:
                        obj_has_qual = (qual in obj_name_lower or 
                                       obj_props.get("side") == qual or
                                       obj_props.get("id") == qual)
                        obj_type_matches = any(t in obj_name_lower for t in target_lower.split() if t != qual)
                        
                        if obj_has_qual and (obj_type_matches or target_lower.replace(qual, "").strip() in obj_name_lower):
                            target_obj = obj
                            break
            
            # Case 4: Partial match (no qualifier)
            elif target_lower in obj_name_lower:
                if target_obj is None:
                    target_obj = obj
        
        if not target_obj:
            return False, f"Target object '{target}' not found in world"
        
        robot_pos = world_model.robot_position
        target_pos = target_obj["position"]
        
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        distance_to_target = math.sqrt(dx**2 + dy**2)
        
        if distance_to_target <= threshold:
            return True, None
        else:
            return False, f"Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), target '{target}' at ({target_pos[0]:.2f}, {target_pos[1]:.2f}), distance {distance_to_target:.2f}m > threshold {threshold}m"
    
    @staticmethod
    def _check_move_distance(condition, world_model):
        """Check if robot moved with the specified action type"""
        direction = condition.get("direction", "forward")
        
        direction_keywords = {
            "forward": ["moved forward", "forward"],
            "backward": ["moved backward", "backward", "back up"]
        }
        
        keywords = direction_keywords.get(direction, [direction])
        
        for action_desc in world_model.action_history:
            action_lower = action_desc.lower()
            if any(keyword in action_lower for keyword in keywords):
                return True, None
        
        return False, f"No {direction} movement action found"
    
    @staticmethod
    def _check_orientation(condition, world_model):
        """Check if robot turned to correct orientation"""
        target_orientation = condition["target_orientation"]
        tolerance = condition["tolerance"]
        
        diff = abs(world_model.robot_orientation - target_orientation)
        if diff > 180:
            diff = 360 - diff
        
        if diff <= tolerance:
            return True, None
        else:
            return False, f"Orientation {world_model.robot_orientation}° instead of {target_orientation}° (tolerance: {tolerance}°)"
    
    @staticmethod
    def _check_find_object(condition, world_model):
        """Check if robot performed find action"""
        target = condition["target"]
        
        for action in world_model.action_history:
            if "360° scan" in action or "find_object" in action.lower():
                if target in action.lower():
                    return True, None
        
        return False, f"Did not perform 360° scan to find {target}"
    
    @staticmethod
    def _check_perceptual_task(condition, world_model, interaction_log=None):
        """
        Check if robot performed perceptual/observation task
        FIXED: Recognizes "speak" actions that answer check/report questions
        FIXED: Recognizes number words (not just digits) for count tasks
        """
        goal_text = condition.get("goal_text", "")
        
        # Keywords that indicate robot completed perceptual task
        perceptual_action_keywords = [
            "described", "look around", "looking", "checked", "reported",
            "counted", "scanning", "360° scan"
        ]
        
        # Check action history for explicit perceptual actions
        for action in world_model.action_history:
            action_lower = action.lower()
            if any(keyword in action_lower for keyword in perceptual_action_keywords):
                return True, None
        
        # NEW: Check if robot SPOKE an answer to the perceptual question
        # If goal asks to "check if X" or "report Y" and robot said something relevant
        if interaction_log and "turns" in interaction_log:
            # Number words for count detection
            number_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", 
                           "eight", "nine", "ten", "eleven", "twelve", "no", "none", "several", "many"]
            
            # Extract what we're checking for from the goal
            check_subjects = []
            
            # "check if laptop is plugged" -> look for "plugged" in robot speech
            if "plugged" in goal_text:
                check_subjects.extend(["plugged", "power", "connected"])
            if "count" in goal_text:
                check_subjects.extend(["count", "there are", "there is", "total"])
                check_subjects.extend(number_words)  # Add number words
            if "report" in goal_text or "check" in goal_text:
                # Generic report - any substantive answer counts
                check_subjects.extend(["yes", "no", "is", "are"])
            
            # Check robot's spoken responses
            for turn in interaction_log["turns"]:
                action_desc = turn.get("robot_action_description", "")
                
                # If robot said something (not just moved)
                if "Misty said:" in action_desc:
                    # Extract what was said
                    speech_match = re.search(r"Misty said: '(.+?)'", action_desc)
                    if speech_match:
                        speech = speech_match.group(1).lower()
                        
                        # For count tasks, also check for digits
                        if "count" in goal_text:
                            has_digit = any(c.isdigit() for c in speech)
                            has_number_word = any(nw in speech for nw in number_words)
                            if has_digit or has_number_word:
                                return True, None
                        
                        # Check if speech contains relevant answer
                        if any(subj in speech for subj in check_subjects):
                            return True, None
                        
                        # Also accept any definitive answer for check tasks
                        if "check" in goal_text and len(speech) > 5:
                            # Robot gave some answer
                            return True, None
        
        return False, "Did not perform describe/observation action or provide perceptual answer"
    
    @staticmethod
    def _extract_navigation_target_for_perceptual(goal_lower):
        """
        Extract the LOCATION to navigate to for perceptual tasks.
        Prioritizes locations over objects being observed.
        e.g., "Navigate to kitchen and count chairs" -> "kitchen" (not "chair")
        Returns None if the location is a general area (not a specific object)
        """
        # Location keywords that are general areas (not specific objects)
        # These mean "be in this area" rather than "go to this object"
        general_areas = ["kitchen", "office", "living room", "bedroom", "bathroom", 
                        "hallway", "garage", "room", "area"]
        
        # Specific objects that can be navigated to
        navigable_objects = ["desk", "table", "counter", "shelf", "door", "fridge"]
        
        # First, try to find what comes right after "navigate to" or "go to"
        nav_pattern = r'(?:navigate to|go to)\s+(?:the\s+)?(\w+)'
        nav_match = re.search(nav_pattern, goal_lower)
        if nav_match:
            nav_target = nav_match.group(1)
            
            # If it's a general area, return None (no specific navigation needed)
            if nav_target in general_areas:
                return None
            
            # If it's a navigable object, return it
            if nav_target in navigable_objects:
                return nav_target
        
        # Check for navigable objects in the first part of the goal (before "and")
        first_part = goal_lower.split(" and ")[0] if " and " in goal_lower else goal_lower
        for obj in navigable_objects:
            if obj in first_part:
                return obj
        
        # If we only found a general area, no specific navigation needed
        for area in general_areas:
            if area in first_part:
                return None
        
        # Fallback to regular extraction
        return TaskEvaluator._extract_target_from_goal(goal_lower)