"""
Run Synthetic Simulation Experiments
Main entry point for running synthetic experiments with EmbodiedPF system
Supports multiple model variants for baseline comparison
UPDATED: Supports filtering by ambiguity type
"""
import sys
import os
import argparse

# Add parent directory to path so we can import from model/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from model_config import ModelConfig
from simulator import SyntheticUserSimulator
from task_scenarios import get_task_scenarios


def get_available_ambiguity_types():
    """Get list of all ambiguity types from task scenarios"""
    scenarios = get_task_scenarios()
    types = set()
    for task in scenarios:
        amb_type = task.get("expected_ambiguity")
        if amb_type:
            types.add(amb_type)
        else:
            types.add("none")  # For tasks without ambiguity (perceptual, find, etc.)
    return sorted(types)


def main():
    """
    Run synthetic experiments with configurable model selection
    
    Available Models:
    - FRICTION_ENABLED: GPT-5 nano with Positive Friction (your main system)
    - NO_FRICTION: GPT-5 nano without Friction (control baseline)
    - ZERO_SHOT_VLM: Zero-shot VLM (minimal prompt, no history)
    - ZERO_SHOT_MULTITURN: Zero-shot VLM + Multi-turn (minimal prompt, with history)
    
    Display Modes:
    - INFO: Clean conversation view (User → Misty actions only)
    - DEBUG: Full details (vision context, JSON responses, stack traces)
    
    Ambiguity Types:
    - referential: Multiple similar objects
    - trajectory: Obstacles blocking path
    - safety: Edge/hazard scenarios
    - implicit_precondition: Object behind/to side of robot
    - orientation: Directional disambiguation
    - none: Perceptual, find, describe, complex tasks
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run EmbodiedPF synthetic experiments")
    parser.add_argument(
        "--ambiguity", "-a",
        type=str,
        default=None,
        help=f"Filter by ambiguity type. Available: {', '.join(get_available_ambiguity_types())}"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        choices=["friction", "no_friction", "zero_shot", "zero_shot_multiturn"],
        help="Model type to use"
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=None,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default=None,
        choices=["INFO", "DEBUG"],
        help="Log level"
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List available ambiguity types and exit"
    )
    
    args = parser.parse_args()
    
    # List types and exit if requested
    if args.list_types:
        print("Available ambiguity types:")
        for t in get_available_ambiguity_types():
            # Count tasks of this type
            scenarios = get_task_scenarios()
            count = sum(1 for task in scenarios 
                       if (task.get("expected_ambiguity") or "none") == t)
            print(f"  - {t}: {count} tasks")
        return
    
    # ============= CONFIGURATION =============
    # Model selection (CLI overrides default)
    MODEL_MAP = {
        "friction": ModelConfig.FRICTION_ENABLED,
        "no_friction": ModelConfig.NO_FRICTION,
        "zero_shot": ModelConfig.ZERO_SHOT_VLM,
        "zero_shot_multiturn": ModelConfig.ZERO_SHOT_MULTITURN,
    }
    
    if args.model:
        MODEL_TYPE = MODEL_MAP[args.model]
    else:
        MODEL_TYPE = ModelConfig.FRICTION_ENABLED  # Default
    
    LOG_LEVEL = args.log_level or "INFO"
    NUM_EPISODES = args.episodes or 100
    AMBIGUITY_TYPE = args.ambiguity  # None means all types
    # =========================================
    
    print("="*70)
    print("Synthetic User Simulator for EmbodiedPF")
    print("="*70)
    print(f"Model: {ModelConfig.get_model_description(MODEL_TYPE)}")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"Episodes: {NUM_EPISODES}")
    if AMBIGUITY_TYPE:
        print(f"Ambiguity Filter: {AMBIGUITY_TYPE}")
    else:
        print(f"Ambiguity Filter: ALL TYPES")
    print("="*70)
    print("\nUser Behavior: Starts VAGUE, disambiguates when asked")
    print("World Model: Text-game style state tracking with POV-based vision\n")
    
    # Initialize the selected model
    print(f"Initializing model: {MODEL_TYPE}...")
    robot_llm = ModelConfig.get_model(MODEL_TYPE)
    
    # Get API key from ModelConfig
    from model_config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return
    
    # Create simulator with selected model and ambiguity filter
    print("Initializing user simulator...")
    simulator = SyntheticUserSimulator(
        robot_llm_layer=robot_llm,
        openai_api_key=OPENAI_API_KEY,
        model_name=ModelConfig.get_model_description(MODEL_TYPE),
        log_level=LOG_LEVEL,
        ambiguity_filter=AMBIGUITY_TYPE  # NEW: Pass ambiguity filter
    )
    
    # Check if we have tasks after filtering
    if len(simulator.task_scenarios) == 0:
        print(f"❌ Error: No tasks found for ambiguity type '{AMBIGUITY_TYPE}'")
        print(f"Available types: {', '.join(get_available_ambiguity_types())}")
        return
    
    print(f"Tasks available: {len(simulator.task_scenarios)}")
    
    # Run experiments
    print("Starting experiments...\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Include model name and ambiguity type in output filename
    model_suffix = MODEL_TYPE.replace("_", "-")
    if AMBIGUITY_TYPE:
        output_file = f"synthetic_results_{model_suffix}_{AMBIGUITY_TYPE}_{timestamp}.json"
    else:
        output_file = f"synthetic_results_{model_suffix}_{timestamp}.json"
    
    results = simulator.run_experiments(
        num_episodes=NUM_EPISODES,
        output_file=output_file
    )
    
    print("\n✅ Simulation complete!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()