"""
Simulated Vision Module
Extracts vision context from world model based on robot's POV
"""


class SimulatedVision:
    """
    Simulates vision by extracting from world model based on robot's POV
    """
    
    @staticmethod
    def generate_from_world_model(world_model):
        """
        Generate vision context from world model based on robot's current POV
        This is what Misty's forward-facing camera would see
        """
        return world_model.get_robot_pov_description()