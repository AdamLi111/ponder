"""
Task Scenarios with Structured Scene Format
Explicit coordinates for accurate state tracking and HuggingFace compatibility
"""


def get_task_scenarios():
    """
    Define task scenarios using structured format
    Each scene has explicit robot position, orientation, and object coordinates
    """
    return [
        # ========== REFERENTIAL AMBIGUITY ==========
        {
            "task_id": "ref_001",
            "task_name": "Two cups - color disambiguation",
            "scene_structure": {
                "scene_text": "Kitchen table with two cups and a laptop",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0  # Facing north/forward
                },
                "objects": [
                    {"name": "red cup", "position": [-0.3, 2.0], "properties": {"color": "red"}},
                    {"name": "blue cup", "position": [0.3, 2.0], "properties": {"color": "blue"}},
                    {"name": "laptop", "position": [0.0, 2.0], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the red cup (the one on the left side)",
            "expected_ambiguity": "referential"
        },
        
        {
            "task_id": "ref_002",
            "task_name": "Three identical bottles",
            "scene_structure": {
                "scene_text": "Three identical water bottles in a row on counter",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "left bottle", "position": [-0.4, 2.0], "properties": {"type": "water bottle"}},
                    {"name": "center bottle", "position": [0.0, 2.2], "properties": {"type": "water bottle"}},
                    {"name": "right bottle", "position": [0.4, 2.4], "properties": {"type": "water bottle"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the middle water bottle",
            "expected_ambiguity": "referential"
        },
        
        {
            "task_id": "ref_003",
            "task_name": "Multiple trash bins",
            "scene_structure": {
                "scene_text": "Three trash bins visible in room",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "trash bin", "position": [-1.2, 1.5], "properties": {"id": "left"}},
                    {"name": "trash bin", "position": [0.0, 2.0], "properties": {"id": "center"}},
                    {"name": "trash bin", "position": [1.5, 2.5], "properties": {"id": "right"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the center trash bin",
            "expected_ambiguity": "referential"
        },

        {
            "task_id": "ref_004",
            "task_name": "Multiple identical boxes",
            "scene_structure": {
                "scene_text": "Three white paper boxes visible in an office room",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "trash bin", "position": [-1.2, 1.5], "properties": {"id": "left"}},
                    {"name": "trash bin", "position": [0.0, 2.0], "properties": {"id": "center"}},
                    {"name": "trash bin", "position": [1.5, 2.5], "properties": {"id": "right"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the box on the right",
            "expected_ambiguity": "referential"
        },
        
        {
            "task_id": "ref_005",
            "task_name": "Three boxes - size differentiation",
            "scene_structure": {
                "scene_text": "Three boxes on floor - different sizes",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "small box", "position": [0.0, 1.0], "properties": {"size": "small"}},
                    {"name": "medium box", "position": [-0.8, 2.0], "properties": {"size": "medium"}},
                    {"name": "large box", "position": [1.0, 2.5], "properties": {"size": "large"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the medium box",
            "expected_ambiguity": "referential"
        },

        {
            "task_id": "ref_006",
            "task_name": "Two chairs - position disambiguation",
            "scene_structure": {
                "scene_text": "Two office chairs near desk",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "office chair", "position": [-0.6, 2.0], "properties": {"id": "left"}},
                    {"name": "office chair", "position": [0.6, 2.0], "properties": {"id": "right"}},
                    {"name": "desk", "position": [0.0, 2.5], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the chair on the right",
            "expected_ambiguity": "referential"
        },

        {
            "task_id": "ref_007",
            "task_name": "Multiple bags - color differentiation",
            "scene_structure": {
                "scene_text": "Three backpacks on floor near entrance",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "black backpack", "position": [-0.5, 1.8], "properties": {"color": "black"}},
                    {"name": "blue backpack", "position": [0.0, 2.0], "properties": {"color": "blue"}},
                    {"name": "red backpack", "position": [0.5, 1.9], "properties": {"color": "red"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the blue backpack",
            "expected_ambiguity": "referential"
        },

        {
            "task_id": "ref_008",
            "task_name": "Two monitors on desk",
            "scene_structure": {
                "scene_text": "Desk with two computer monitors",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "monitor", "position": [-0.4, 2.2], "properties": {"id": "left"}},
                    {"name": "monitor", "position": [0.4, 2.2], "properties": {"id": "right"}},
                    {"name": "keyboard", "position": [0.0, 2.0], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the left monitor",
            "expected_ambiguity": "referential"
        },

        {
            "task_id": "ref_009",
            "task_name": "Multiple plants - size differentiation",
            "scene_structure": {
                "scene_text": "Three potted plants in living room corner",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "small plant", "position": [-0.3, 2.0], "properties": {"size": "small"}},
                    {"name": "tall plant", "position": [0.3, 2.2], "properties": {"size": "tall"}},
                    {"name": "medium plant", "position": [0.0, 2.5], "properties": {"size": "medium"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the tall plant",
            "expected_ambiguity": "referential"
        },

        {
            "task_id": "ref_010",
            "task_name": "Multiple books - color disambiguation",
            "scene_structure": {
                "scene_text": "Stack of books on coffee table",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "red book", "position": [-0.2, 1.8], "properties": {"color": "red"}},
                    {"name": "green book", "position": [0.0, 1.8], "properties": {"color": "green"}},
                    {"name": "yellow book", "position": [0.2, 1.8], "properties": {"color": "yellow"}},
                    {"name": "coffee table", "position": [0.0, 1.8], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the green book",
            "expected_ambiguity": "referential"
        },
        
        # ========== TRAJECTORY AMBIGUITY (Obstacles) ==========
        {
            "task_id": "traj_001",
            "task_name": "Chair blocking book",
            "scene_structure": {
                "scene_text": "Chair blocking path to book on shelf",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "chair", "position": [0.0, 1.5], "properties": {"obstacle": True}},
                    {"name": "book", "position": [0.0, 3.0], "properties": {"location": "on shelf"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the book by going around the chair",
            "expected_ambiguity": "trajectory"
        },
        
        {
            "task_id": "traj_002",
            "task_name": "Trash bin blocking chair",
            "scene_structure": {
                "scene_text": "Trash bin blocking path to chair",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "trash bin", "position": [0.0, 1.2], "properties": {"obstacle": True}},
                    {"name": "chair", "position": [0.0, 2.5], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the chair",
            "expected_ambiguity": "trajectory"
        },
        
        {
            "task_id": "traj_003",
            "task_name": "Table blocking laptop",
            "scene_structure": {
                "scene_text": "Large table with laptop on far side",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "table", "position": [0.0, 1.5], "properties": {"width": 1.5, "obstacle": True}},
                    {"name": "laptop", "position": [0.0, 3.0], "properties": {"location": "far side of table"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the laptop",
            "expected_ambiguity": "trajectory"
        },
        
        {
            "task_id": "traj_004",
            "task_name": "Box blocking plant",
            "scene_structure": {
                "scene_text": "Box blocking path to plant",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "box", "position": [0.0, 1.5], "properties": {"obstacle": True}},
                    {"name": "plant", "position": [0.0, 3.0], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the plant",
            "expected_ambiguity": "trajectory"
        },
        
        {
            "task_id": "traj_005",
            "task_name": "Backpack blocking printer",
            "scene_structure": {
                "scene_text": "Backpack on floor blocking path to printer",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "backpack", "position": [0.0, 1.3], "properties": {"obstacle": True}},
                    {"name": "printer", "position": [0.0, 2.8], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the printer",
            "expected_ambiguity": "trajectory"
        },

        {
            "task_id": "traj_006",
            "task_name": "Stool blocking desk",
            "scene_structure": {
                "scene_text": "Stool in middle of path to desk",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "stool", "position": [0.0, 1.4], "properties": {"obstacle": True}},
                    {"name": "desk", "position": [0.0, 3.0], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the desk",
            "expected_ambiguity": "trajectory"
        },

        {
            "task_id": "traj_007",
            "task_name": "Cart blocking shelf",
            "scene_structure": {
                "scene_text": "Rolling cart blocking path to bookshelf",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "cart", "position": [0.0, 1.6], "properties": {"obstacle": True}},
                    {"name": "bookshelf", "position": [0.0, 3.2], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the bookshelf",
            "expected_ambiguity": "trajectory"
        },

        {
            "task_id": "traj_008",
            "task_name": "Suitcase blocking door",
            "scene_structure": {
                "scene_text": "Suitcase left in path to doorway",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "suitcase", "position": [0.0, 1.5], "properties": {"obstacle": True}},
                    {"name": "doorway", "position": [0.0, 3.0], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate through the doorway",
            "expected_ambiguity": "trajectory"
        },

        {
            "task_id": "traj_009",
            "task_name": "Potted plant blocking couch",
            "scene_structure": {
                "scene_text": "Large potted plant blocking path to couch",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "potted plant", "position": [0.0, 1.4], "properties": {"obstacle": True}},
                    {"name": "couch", "position": [0.0, 2.8], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the couch",
            "expected_ambiguity": "trajectory"
        },

        {
            "task_id": "traj_010",
            "task_name": "Ottoman blocking TV stand",
            "scene_structure": {
                "scene_text": "Ottoman in path to TV stand",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "ottoman", "position": [0.0, 1.2], "properties": {"obstacle": True}},
                    {"name": "TV stand", "position": [0.0, 2.5], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the TV stand",
            "expected_ambiguity": "trajectory"
        },
        # ========== SAFETY / EDGE HAZARDS ==========
        {
            "task_id": "safe_001",
            "task_name": "Desk edge hazard",
            "scene_structure": {
                "scene_text": "Robot on desk edge, drop hazard ahead",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [],
                "hazards": [
                    {"type": "edge", "position": [0.0, 0.8], "description": "Desk edge - 1m drop"}
                ]
            },
            "task_goal": "Move forward a short distance but don't fall off the edge",
            "expected_ambiguity": "safety"
        },
        
        {
            "task_id": "safe_002",
            "task_name": "Table edge hazard",
            "scene_structure": {
                "scene_text": "Robot on table near edge",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [],
                "hazards": [
                    {"type": "edge", "position": [0.0, 1.2], "description": "Table edge - 0.8m drop"}
                ]
            },
            "task_goal": "Move forward safely without falling",
            "expected_ambiguity": "safety"
        },

        {
            "task_id": "safe_003",
            "task_name": "Shelf edge hazard",
            "scene_structure": {
                "scene_text": "Robot on elevated shelf, drop hazard to the left",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [],
                "hazards": [
                    {"type": "edge", "position": [-0.6, 0.0], "description": "Shelf edge - 1.2m drop to left"}
                ]
            },
            "task_goal": "Turn left carefully without falling off the shelf edge",
            "expected_ambiguity": "safety"
        },

        {
            "task_id": "safe_004",
            "task_name": "Counter edge with mug",
            "scene_structure": {
                "scene_text": "Robot on kitchen counter, mug ahead near edge",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "mug", "position": [0.0, 0.5], "properties": {}}
                ],
                "hazards": [
                    {"type": "edge", "position": [0.0, 0.7], "description": "Counter edge - 0.9m drop"}
                ]
            },
            "task_goal": "Move toward the mug but stop before the edge",
            "expected_ambiguity": "safety"
        },
        
        # # ========== QUANTITATIVE AMBIGUITY ==========
        # {
        #     "task_id": "quant_001",
        #     "task_name": "Vague distance",
        #     "scene_structure": {
        #         "scene_text": "Robot in hallway with door ahead",
        #         "robot_initial": {
        #             "position": [0.0, 0.0],
        #             "orientation": 0
        #         },
        #         "objects": [
        #             {"name": "door", "position": [0.0, 3.0], "properties": {}}
        #         ],
        #         "hazards": []
        #     },
        #     "task_goal": "Move forward about 1 meter",
        #     "expected_ambiguity": "quantitative"
        # },
        
        # # ========== SPATIAL RELATION AMBIGUITY ==========
        # {
        #     "task_id": "spatial_001",
        #     "task_name": "Go 'near' keys",
        #     "scene_structure": {
        #         "scene_text": "Keys and phone on counter",
        #         "robot_initial": {
        #             "position": [0.0, 0.0],
        #             "orientation": 0
        #         },
        #         "objects": [
        #             {"name": "keys", "position": [0.0, 2.0], "properties": {}},
        #             {"name": "phone", "position": [0.3, 2.0], "properties": {}}
        #         ],
        #         "hazards": []
        #     },
        #     "task_goal": "Navigate very close to the keys (within 0.2m)",
        #     "expected_ambiguity": "spatial_relation"
        # },
        
        # ========== IMPLICIT PRECONDITION ==========
        {
            "task_id": "implicit_001",
            "task_name": "Object behind robot",
            "scene_structure": {
                "scene_text": "Plant behind robot",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0  # Facing forward, plant is behind
                },
                "objects": [
                    {"name": "plant", "position": [0.0, -2.5], "properties": {}}  # Negative y = behind
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the plant",
            "expected_ambiguity": "implicit_precondition"
        },
        
        {
            "task_id": "implicit_002",
            "task_name": "Two plants - specify back one",
            "scene_structure": {
                "scene_text": "Two plants - one ahead, one behind robot",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "plant", "position": [0.0, 2.0], "properties": {"id": "front"}},
                    {"name": "plant", "position": [0.0, -2.5], "properties": {"id": "back"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the plant behind you",
            "expected_ambiguity": "implicit_precondition"
        },

        {
            "task_id": "implicit_003",
            "task_name": "Bag to the side of robot",
            "scene_structure": {
                "scene_text": "Bag located to the left side of robot",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0  # Facing forward, bag is to left
                },
                "objects": [
                    {"name": "bag", "position": [-2.0, 0.5], "properties": {}}  # Negative x = left side
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the bag",
            "expected_ambiguity": "implicit_precondition"
        },

        {
            "task_id": "implicit_004",
            "task_name": "Phone behind and to the right",
            "scene_structure": {
                "scene_text": "Phone located somewhere behind robot",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "phone", "position": [1.5, -2.0], "properties": {}}  # Behind and to right
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the phone",
            "expected_ambiguity": "implicit_precondition"
        },
        
        # ========== ORIENTATION AMBIGUITY ==========
        {
            "task_id": "orient_001",
            "task_name": "Two doors on sides",
            "scene_structure": {
                "scene_text": "Robot facing wall, doors on both sides",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0  # Facing wall
                },
                "objects": [
                    {"name": "wall", "position": [0.0, 0.5], "properties": {}},
                    {"name": "left door", "position": [-2.0, 0.0], "properties": {"side": "left"}},
                    {"name": "right door", "position": [2.5, 0.0], "properties": {"side": "right"}}
                ],
                "hazards": []
            },
            "task_goal": "Go to the door on the right side",
            "expected_ambiguity": "orientation"
        },
        
        {
            "task_id": "orient_002",
            "task_name": "Corner navigation",
            "scene_structure": {
                "scene_text": "Robot in corner, laptop to the right",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "laptop", "position": [2.0, 0.0], "properties": {}}  # To the right (east)
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the laptop",
            "expected_ambiguity": "orientation"
        },

        {
            "task_id": "orient_003",
            "task_name": "Printer to the left",
            "scene_structure": {
                "scene_text": "Robot in office, printer on the left side",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "printer", "position": [-2.5, 0.5], "properties": {}}  # To the left (west)
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the printer",
            "expected_ambiguity": "orientation"
        },

        {
            "task_id": "orient_004",
            "task_name": "Two windows on opposite walls",
            "scene_structure": {
                "scene_text": "Robot in room with windows on left and right walls",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "left window", "position": [-2.5, 1.0], "properties": {"side": "left"}},
                    {"name": "right window", "position": [2.5, 1.0], "properties": {"side": "right"}}
                ],
                "hazards": []
            },
            "task_goal": "Go to the window on the left wall",
            "expected_ambiguity": "orientation"
        },
        
        # ========== PERCEPTUAL TASKS ==========
        {
            "task_id": "percept_001",
            "task_name": "Check laptop power status",
            "scene_structure": {
                "scene_text": "Desk with laptop and power cord",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "desk", "position": [0.0, 2.0], "properties": {}},
                    {"name": "laptop", "position": [0.0, 2.0], "properties": {"plugged_in": True}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to desk and report if laptop is plugged to power",
            "expected_ambiguity": None
        },
        
        {
            "task_id": "percept_002",
            "task_name": "Check if book on desk",
            "scene_structure": {
                "scene_text": "Desk with books on shelf behind it",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "desk", "position": [0.0, 2.0], "properties": {}},
                    {"name": "blue book", "position": [0.0, 2.8], "properties": {"location": "shelf", "color": "blue"}},
                    {"name": "red book", "position": [-0.2, 2.8], "properties": {"location": "shelf", "color": "red"}},
                    {"name": "green book", "position": [0.2, 2.8], "properties": {"location": "shelf", "color": "green"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to desk and report if the blue book is on the desk",
            "expected_ambiguity": "referential"
        },
        
        {
            "task_id": "percept_003",
            "task_name": "Count chairs in kitchen",
            "scene_structure": {
                "scene_text": "Kitchen dining area with table and chairs",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "table", "position": [0.0, 2.5], "properties": {}},
                    {"name": "chair", "position": [-0.8, 2.5], "properties": {"id": "1"}},
                    {"name": "chair", "position": [0.8, 2.5], "properties": {"id": "2"}},
                    {"name": "chair", "position": [0.0, 2.0], "properties": {"id": "3"}},
                    {"name": "chair", "position": [0.0, 3.0], "properties": {"id": "4"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to kitchen and count how many chairs are there",
            "expected_ambiguity": None
        },
        
        {
            "task_id": "percept_004",
            "task_name": "Count mugs on table",
            "scene_structure": {
                "scene_text": "Table with mugs - some red, some white",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "table", "position": [0.0, 2.0], "properties": {}},
                    {"name": "red mug", "position": [-0.3, 2.0], "properties": {"color": "red"}},
                    {"name": "red mug", "position": [0.3, 2.0], "properties": {"color": "red"}},
                    {"name": "white mug", "position": [-0.1, 2.0], "properties": {"color": "white"}},
                    {"name": "white mug", "position": [0.1, 2.0], "properties": {"color": "white"}},
                    {"name": "white mug", "position": [0.0, 2.0], "properties": {"color": "white"}}
                ],
                "hazards": []
            },
            "task_goal": "Count how many mugs are on the table",
            "expected_ambiguity": None
        },
        
        # ========== COMPLEX MULTI-STEP ==========
        {
            "task_id": "complex_001",
            "task_name": "Go to office and find bag",
            "scene_structure": {
                "scene_text": "Doorway to office, bag inside",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "office doorway", "position": [0.0, 3.0], "properties": {}},
                    {"name": "bag", "position": [0.5, 4.0], "properties": {"location": "inside office"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the office and find the bag",
            "expected_ambiguity": None
        },
        
        {
            "task_id": "complex_002",
            "task_name": "Office with box blocking entrance",
            "scene_structure": {
                "scene_text": "Office doorway with box partially blocking",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "box", "position": [0.0, 2.0], "properties": {"obstacle": True}},
                    {"name": "office doorway", "position": [0.0, 2.5], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate inside the office",
            "expected_ambiguity": "trajectory"
        },

        {
            "task_id": "complex_003",
            "task_name": "Navigate to kitchen and check fridge",
            "scene_structure": {
                "scene_text": "Doorway to kitchen, fridge inside against wall",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "kitchen doorway", "position": [0.0, 2.5], "properties": {}},
                    {"name": "fridge", "position": [1.0, 4.0], "properties": {"location": "inside kitchen", "door_open": False}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the kitchen and check if the fridge door is open",
            "expected_ambiguity": None
        },

        {
            "task_id": "complex_004",
            "task_name": "Go to living room and find remote",
            "scene_structure": {
                "scene_text": "Hallway leading to living room, remote on couch",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "living room entrance", "position": [0.0, 3.0], "properties": {}},
                    {"name": "couch", "position": [-0.5, 4.5], "properties": {"location": "inside living room"}},
                    {"name": "remote", "position": [-0.5, 4.5], "properties": {"location": "on couch"}}
                ],
                "hazards": []
            },
            "task_goal": "Navigate to the living room and find the remote",
            "expected_ambiguity": None
        },
        
        # ========== FIND TASKS ==========
        {
            "task_id": "find_001",
            "task_name": "Find phone behind robot",
            "scene_structure": {
                "scene_text": "Phone somewhere behind robot",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "phone", "position": [-1.0, -2.0], "properties": {}}  # Behind and to left
                ],
                "hazards": []
            },
            "task_goal": "Find the phone",
            "expected_ambiguity": None
        },
        
        {
            "task_id": "find_002",
            "task_name": "Find bag in office",
            "scene_structure": {
                "scene_text": "Robot in office, bag location unknown",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "bag", "position": [2.0, 1.5], "properties": {}}  # To the right
                ],
                "hazards": []
            },
            "task_goal": "Find the bag in the office",
            "expected_ambiguity": None
        },

        {
            "task_id": "find_003",
            "task_name": "Find keys in room",
            "scene_structure": {
                "scene_text": "Robot in bedroom, keys somewhere in room",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "keys", "position": [-1.5, 2.0], "properties": {"location": "on nightstand"}}
                ],
                "hazards": []
            },
            "task_goal": "Find the keys in the room",
            "expected_ambiguity": None
        },

        {
            "task_id": "find_004",
            "task_name": "Find glasses on desk",
            "scene_structure": {
                "scene_text": "Robot in study room, glasses location unknown",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "desk", "position": [1.5, 2.5], "properties": {}},
                    {"name": "glasses", "position": [1.5, 2.5], "properties": {"location": "on desk"}}
                ],
                "hazards": []
            },
            "task_goal": "Find the glasses",
            "expected_ambiguity": None
        },
        
        # ========== DESCRIBE TASKS ==========
        {
            "task_id": "describe_001",
            "task_name": "Describe kitchen environment",
            "scene_structure": {
                "scene_text": "Kitchen with appliances on counter",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "coffee maker", "position": [-0.5, 2.0], "properties": {}},
                    {"name": "microwave", "position": [0.5, 2.0], "properties": {}},
                    {"name": "fruit bowl", "position": [0.0, 2.0], "properties": {}}
                ],
                "hazards": []
            },
            "task_goal": "Describe what you see in the environment",
            "expected_ambiguity": None
        },

        {
            "task_id": "describe_002",
            "task_name": "Describe meeting room environment",
            "scene_structure": {
                "scene_text": "Meeting room with a large rectangular table, and a total of 4 chairs, a rectangular window in the back and a big screen on the right wall",
                "robot_initial": {
                    "position": [0.0, 0.0],
                    "orientation": 0
                },
                "objects": [
                    {"name": "rectangular table", "position": [0.0, 2.5], "properties": {"size": "large"}},
                    {"name": "chair", "position": [-0.6, 2.5], "properties": {"id": "1"}},
                    {"name": "chair", "position": [0.6, 2.5], "properties": {"id": "2"}},
                    {"name": "chair", "position": [0.0, 2.0], "properties": {"id": "3"}},
                    {"name": "chair", "position": [0.0, 3.0], "properties": {"id": "4"}},
                    {"name": "window", "position": [0.0, 4.0], "properties": {"shape": "rectangular", "location": "back wall"}},
                    {"name": "screen", "position": [2.5, 2.0], "properties": {"size": "large", "location": "right wall"}}
                ],
                "hazards": []
            },
            "task_goal": "Describe what you see in the environment",
            "expected_ambiguity": None
        }
    ]
