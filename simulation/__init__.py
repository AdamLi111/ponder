"""
Simulation package for EmbodiedPF synthetic experiments

This package contains all modules for running synthetic user simulations
with world-model based state tracking and dynamic LLM user agents.

Now uses structured scene format with explicit coordinates for:
- Accurate position tracking
- FOV-based visibility calculation  
- Precise success evaluation
- HuggingFace dataset compatibility

Modules:
    - world_model: Coordinate-based world state tracking
    - scene_structure: Structured scene format definition
    - simulated_vision: Vision extraction from world model (FOV-based)
    - action_parser: Parse VLM JSON responses to human-readable text
    - simulated_user: LLM-based user agent with omniscient world view
    - task_scenarios: Task definitions with structured scenes
    - simulator: Main orchestration logic
    - success_evaluator: Coordinate-based task success evaluation
    - model_config: Model configuration and factory
    - run_simulation: Entry point to run experiments
"""

__version__ = "2.0.0"
__author__ = "EmbodiedPF Team"

# Import main classes for convenience
from .world_model import WorldModel
from .scene_structure import SceneStructure
from .simulated_vision import SimulatedVision
from .action_parser import ActionParser
from .simulated_user import SimulatedUser
from .simulator import SyntheticUserSimulator
from .task_scenarios import get_task_scenarios
from .task_evaluator import TaskEvaluator
from .model_config import ModelConfig

__all__ = [
    'WorldModel',
    'SceneStructure',
    'SimulatedVision',
    'ActionParser',
    'SimulatedUser',
    'SyntheticUserSimulator',
    'get_task_scenarios',
    'TaskEvaluator',
    'ModelConfig',
]