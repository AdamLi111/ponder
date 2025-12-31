"""
Batch Runner for Model Comparison
Runs experiments with all model variants for comprehensive baseline comparison
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from model_config import ModelConfig, OPENAI_API_KEY
from simulator import SyntheticUserSimulator
import time


def run_model_comparison(models_to_test=None, num_episodes=10, log_level="INFO"):
    """
    Run experiments with multiple models for comparison
    
    Args:
        models_to_test: List of model types to test (None = all models)
        num_episodes: Number of episodes per model
        log_level: "INFO" or "DEBUG"
    """
    if models_to_test is None:
        models_to_test = ModelConfig.get_all_models()
    
    print("="*70)
    print("BATCH MODEL COMPARISON EXPERIMENTS")
    print("="*70)
    print(f"Models to test: {len(models_to_test)}")
    for model_type in models_to_test:
        print(f"  - {ModelConfig.get_model_description(model_type)}")
    print(f"Episodes per model: {num_episodes}")
    print(f"Log level: {log_level}")
    print("="*70)
    print()
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return
    
    all_results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for i, model_type in enumerate(models_to_test, 1):
        print("\n" + "="*70)
        print(f"RUNNING MODEL {i}/{len(models_to_test)}: {model_type}")
        print("="*70)
        
        # Initialize model
        print(f"Initializing {ModelConfig.get_model_description(model_type)}...")
        robot_llm = ModelConfig.get_model(model_type)
        
        # Create simulator
        simulator = SyntheticUserSimulator(
            robot_llm_layer=robot_llm,
            openai_api_key=OPENAI_API_KEY,
            model_name=ModelConfig.get_model_description(model_type),
            log_level=log_level
        )
        
        # Run experiments
        model_suffix = model_type.replace("_", "-")
        output_file = f"results_{model_suffix}_{timestamp}.json"
        
        results = simulator.run_experiments(
            num_episodes=num_episodes,
            output_file=output_file
        )
        
        all_results[model_type] = {
            "output_file": output_file,
            "results": results
        }
        
        # Brief pause between models
        if i < len(models_to_test):
            print(f"\nPausing before next model...")
            time.sleep(2)
    
    # Print comparison summary
    print("\n\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for model_type in models_to_test:
        result_data = all_results[model_type]["results"]
        info = result_data["experiment_info"]
        
        # Calculate metrics
        completed = sum(1 for ep in result_data['episodes'] 
                       if ep['interaction'].get('completed', False))
        
        successful = sum(
            1 for ep in result_data['episodes']
            if ep['interaction'].get('success_evaluation', {}).get('success', False)
        )
        
        total_turns = sum(ep['interaction'].get('total_turns', 0) 
                         for ep in result_data['episodes'])
        clarifications = sum(
            1 for ep in result_data['episodes']
            for turn in ep['interaction'].get('turns', [])
            if 'Misty asked:' in turn.get('robot_action_description', '')
        )
        
        avg_goal_success = sum(
            ep['interaction'].get('success_evaluation', {}).get('success_rate', 0)
            for ep in result_data['episodes']
        ) / num_episodes if num_episodes > 0 else 0
        
        print(f"\n{ModelConfig.get_model_description(model_type)}:")
        print(f"  File: {all_results[model_type]['output_file']}")
        print(f"  Completed: {completed}/{num_episodes} ({completed/num_episodes*100:.1f}%)")
        print(f"  Successful: {successful}/{num_episodes} ({successful/num_episodes*100:.1f}%)")
        print(f"  Avg goal success: {avg_goal_success*100:.1f}%")
        print(f"  Total turns: {total_turns}")
        print(f"  Clarifications: {clarifications} ({clarifications/total_turns*100:.1f}%)")
        print(f"  Avg turns/episode: {total_turns/num_episodes:.1f}")
    
    print("\n" + "="*70)
    print("✅ All models tested!")
    print("="*70)
    
    return all_results


def main():
    """Main entry point for batch comparison"""
    
    # ============= CONFIGURATION =============
    # Select which models to test
    MODELS_TO_TEST = [
        ModelConfig.FRICTION_ENABLED,
        ModelConfig.NO_FRICTION,
        ModelConfig.ZERO_SHOT_VLM,
        ModelConfig.ZERO_SHOT_MULTITURN
    ]
    
    NUM_EPISODES = 50       # Episodes per model
    LOG_LEVEL = "INFO"      # "INFO" or "DEBUG"
    # =========================================
    
    results = run_model_comparison(
        models_to_test=MODELS_TO_TEST,
        num_episodes=NUM_EPISODES,
        log_level=LOG_LEVEL
    )
    
    print(f"\n🎉 Batch comparison complete!")
    print(f"Total models tested: {len(MODELS_TO_TEST)}")
    print(f"Total episodes: {NUM_EPISODES * len(MODELS_TO_TEST)}")


if __name__ == "__main__":
    main()