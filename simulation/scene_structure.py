"""
Structured Scene Format
Defines explicit scene structure with coordinates for accurate state tracking
HuggingFace-ready format
"""


class SceneStructure:
    """
    Structured scene representation with explicit coordinates
    """
    
    def __init__(self, scene_dict):
        """
        Initialize from structured scene dictionary
        
        Args:
            scene_dict: {
                "scene_text": str,
                "robot_initial": {"position": [x, y], "orientation": degrees},
                "objects": [{"name": str, "position": [x, y], "properties": {}}],
                "hazards": [{"type": str, "position": [x, y]}]
            }
        """
        self.scene_text = scene_dict["scene_text"]
        self.robot_initial = scene_dict["robot_initial"]
        self.objects = scene_dict["objects"]
        self.hazards = scene_dict.get("hazards", [])
    
    def get_object_by_name(self, name):
        """Get object by name"""
        for obj in self.objects:
            if obj["name"].lower() == name.lower():
                return obj
        return None
    
    def get_objects_in_fov(self, robot_pos, robot_orientation, fov_angle=120, max_distance=10.0):
        """
        Get objects visible in robot's field of view
        
        Args:
            robot_pos: [x, y]
            robot_orientation: degrees (0=north, 90=east, 180=south, 270=west)
            fov_angle: Field of view in degrees (default 120°)
            max_distance: Maximum visibility distance
            
        Returns:
            list: Objects visible to robot
        """
        import math
        
        visible_objects = []
        
        for obj in self.objects:
            obj_pos = obj["position"]
            
            # Calculate relative position
            dx = obj_pos[0] - robot_pos[0]
            dy = obj_pos[1] - robot_pos[1]
            
            # Calculate distance
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > max_distance:
                continue
            
            # Calculate angle to object
            angle_to_obj = math.atan2(dx, dy) * 180 / math.pi
            
            # Normalize to 0-360
            angle_to_obj = angle_to_obj % 360
            robot_orientation_norm = robot_orientation % 360
            
            # Calculate angle difference
            angle_diff = abs(angle_to_obj - robot_orientation_norm)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Check if within FOV
            if angle_diff <= fov_angle / 2:
                visible_objects.append({
                    **obj,
                    "distance": distance,
                    "angle": angle_diff
                })
        
        return visible_objects
    
    def to_huggingface_format(self):
        """Convert to HuggingFace dataset format"""
        return {
            "scene_description": self.scene_text,
            "initial_state": {
                "robot_position": self.robot_initial["position"],
                "robot_orientation": self.robot_initial["orientation"],
                "objects": self.objects,
                "hazards": self.hazards
            }
        }