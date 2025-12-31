"""
Synthetic User Simulator
Orchestrates synthetic experiments with simulated user and robot system
Supports multiple model variants for baseline comparison
NOW: Properly continues multi-step tasks instead of early termination
UPDATED: Collision with obstacle causes immediate task failure
"""
import json
import random
import time
from datetime import datetime

from world_model import WorldModel
from simulated_vision import SimulatedVision
from action_parser import ActionParser
from simulated_user import SimulatedUser
from task_scenarios import get_task_scenarios
from simulation.task_evaluator import TaskEvaluator


class SyntheticUserSimulator:
    """
    Orchestrates synthetic experiments with simulated user and robot system
    Supports multiple model configurations for comparison
    UPDATED: Supports filtering by ambiguity type
    """
    
    def __init__(self, robot_llm_layer, openai_api_key, model_name="unknown", log_level="INFO", ambiguity_filter=None):
        self.robot_llm = robot_llm_layer
        self.simulated_user = SimulatedUser(openai_api_key)
        self.log_level = log_level.upper()
        self.model_name = model_name
        self.ambiguity_filter = ambiguity_filter
        
        # Load and filter task scenarios
        all_scenarios = get_task_scenarios()
        if ambiguity_filter:
            # Filter by ambiguity type
            # Handle "none" as a special case for tasks without expected_ambiguity
            if ambiguity_filter.lower() == "none":
                self.task_scenarios = [
                    task for task in all_scenarios 
                    if task.get("expected_ambiguity") is None
                ]
            else:
                self.task_scenarios = [
                    task for task in all_scenarios 
                    if task.get("expected_ambiguity") == ambiguity_filter.lower()
                ]
        else:
            self.task_scenarios = all_scenarios
        
    def simulate_interaction(self, task, max_turns=6):
        """
        Simulate a complete task interaction between user and robot
        Uses world model for accurate state tracking
        NOW: Continues until goal is fully achieved or max_turns reached
        UPDATED: Collision causes immediate task failure
        """
        # Initialize world model
        world_model = WorldModel(task['scene_structure'], task['task_goal'])
        
        # Header
        if self.log_level == "DEBUG":
            print(f"\n{'='*70}")
            print(f"Task: {task['task_name']}")
            print(f"\n{world_model.get_full_state_description()}")
            print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print(f"Task: {task['task_name']}")
            print(f"{'='*70}")
        
        # Reset simulated user with world model
        self.simulated_user.reset(world_model)
        
        interaction_log = {
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "scene_description": task["scene_structure"],
            "task_goal": task["task_goal"],
            "timestamp": datetime.now().isoformat(),
            "turns": [],
            "collision": None  # Track collision info
        }
        
        # Generate initial VAGUE command from simulated user
        try:
            user_command = self.simulated_user.generate_initial_command()
            
            turn_num = 1
            task_complete = False
            collision_occurred = False
            
            while turn_num <= max_turns and not task_complete and not collision_occurred:
                # Generate vision context from world model (Misty's POV)
                vision_description = SimulatedVision.generate_from_world_model(world_model)
                
                if self.log_level == "DEBUG":
                    print(f"\n--- Misty's POV (Turn {turn_num}) ---")
                    print(vision_description)
                    print(f"--- End POV ---")
                
                # Display
                if self.log_level == "INFO":
                    print(f"\n[Turn {turn_num}]")
                    print(f"User: {user_command}")
                elif self.log_level == "DEBUG":
                    print(f"\n--- Turn {turn_num} ---")
                    print(f"User: {user_command}")
                
                # Prepare robot input
                robot_input = f"{vision_description}\nObjective Scene Description: {task['scene_structure']}\n\nUser command: {user_command}"
                
                # Get robot VLM response
                try:
                    vlm_response = self.robot_llm.parse_intent_with_vision(
                        robot_input,
                        image_data_base64=None
                    )
                    
                    if self.log_level == "DEBUG":
                        print(f"Robot VLM Response: {json.dumps(vlm_response, indent=2)}")
                    
                    # Parse what robot did/said
                    robot_action_description = ActionParser.parse_action(vlm_response)
                    print(f"Misty: {robot_action_description}")
                    
                    # Update world model based on robot's action and check for collision
                    collision_result = world_model.update_from_action(robot_action_description)
                    
                    # Handle collision - immediate task failure
                    if collision_result:
                        collision_occurred = True
                        interaction_log["collision"] = collision_result
                        
                        if self.log_level == "INFO":
                            print(f"💥 COLLISION: {collision_result['collision_message']}")
                        elif self.log_level == "DEBUG":
                            print(f"💥 COLLISION DETECTED!")
                            print(f"   Obstacle: {collision_result['obstacle_name']}")
                            print(f"   Obstacle Position: {collision_result['obstacle_position']}")
                            print(f"   Robot stopped at: ({collision_result['collision_point'][0]:.2f}, {collision_result['collision_point'][1]:.2f})")
                        
                        # Log this turn before breaking
                        if self.log_level == "INFO":
                            turn_log = {
                                "turn": turn_num,
                                "user_input": user_command,
                                "vision_context": vision_description,
                                "robot_action_description": robot_action_description,
                                "collision": collision_result
                            }
                        else:
                            turn_log = {
                                "turn": turn_num,
                                "user_input": user_command,
                                "vision_context": vision_description,
                                "vlm_response": vlm_response,
                                "robot_action_description": robot_action_description,
                                "world_state_after": world_model.get_full_state_description(),
                                "collision": collision_result
                            }
                        interaction_log["turns"].append(turn_log)
                        
                        # Exit loop - collision causes immediate failure
                        break
                    
                    if self.log_level == "DEBUG":
                        print(f"[Position after action: ({world_model.robot_position[0]:.2f}, {world_model.robot_position[1]:.2f}), Orientation: {world_model.robot_orientation}°]")
                    
                    # Log this turn
                    if self.log_level == "INFO":
                        turn_log = {
                            "turn": turn_num,
                            "user_input": user_command,
                            "vision_context": vision_description,
                            "robot_action_description": robot_action_description
                        }
                    else:
                        turn_log = {
                            "turn": turn_num,
                            "user_input": user_command,
                            "vision_context": vision_description,
                            "vlm_response": vlm_response,
                            "robot_action_description": robot_action_description,
                            "world_state_after": world_model.get_full_state_description()
                        }
                    interaction_log["turns"].append(turn_log)
                    
                    # User observes the action
                    self.simulated_user.observe_robot_action(robot_action_description)
                    
                    # Check if robot needs clarification
                    if vlm_response.get('action') == 'clarify':
                        robot_message = vlm_response.get('text', 'Can you clarify?')
                        
                        # Get user's response to clarification
                        user_command = self.simulated_user.respond_to_robot(
                            robot_message,
                            robot_action_description
                        )
                        
                        # Safety check: if None returned, mark complete
                        if user_command is None:
                            if self.log_level == "DEBUG":
                                print(f"→ User agent signaled completion during clarification")
                            task_complete = True
                        else:
                            turn_num += 1
                        
                    else:
                        # Robot executed an action - check if FULL goal achieved
                        temp_success_eval = TaskEvaluator.evaluate_task_success(
                            world_model, 
                            task['task_goal'], 
                            interaction_log
                        )
                        
                        if temp_success_eval['success']:
                            # ALL goal conditions met!
                            if self.log_level == "INFO":
                                print(f"→ Task completed\n")
                            elif self.log_level == "DEBUG":
                                print(f"→ Task completed (all goal conditions achieved)")
                                print(f"\n{world_model.get_full_state_description()}")
                            task_complete = True
                            
                        elif turn_num >= max_turns:
                            # Hit max turns without full success
                            if self.log_level == "DEBUG":
                                print(f"→ Max turns reached, ending task")
                            task_complete = True
                            
                        else:
                            # Goal NOT fully achieved - let user issue follow-up
                            # This is the key change from early termination!
                            
                            if self.log_level == "DEBUG":
                                print(f"→ Action executed, {temp_success_eval['goal_conditions_met']}/{temp_success_eval['total_goal_conditions']} conditions met. Getting follow-up command...")
                            
                            # Generate robot's spoken output (if any)
                            robot_message = vlm_response.get('text', '')
                            if not robot_message:
                                # Create a status message for non-speaking actions
                                robot_message = f"Done. {robot_action_description}"
                            
                            # Get user's follow-up command
                            user_command = self.simulated_user.respond_to_robot(
                                robot_message,
                                robot_action_description,
                                task_complete=False
                            )
                            
                            # Check if user returned None (goal fully achieved)
                            if user_command is None:
                                # User agent determined all goals are met
                                if self.log_level == "DEBUG":
                                    print(f"→ User agent signaled goal completion (no further commands)")
                                task_complete = True
                            else:
                                # User has more commands - continue
                                turn_num += 1
                
                except Exception as e:
                    print(f"❌ Error in robot VLM: {e}")
                    if self.log_level == "DEBUG":
                        import traceback
                        traceback.print_exc()
                    interaction_log["error"] = str(e)
                    break
                    
        except Exception as e:
            print(f"❌ Error generating user command: {e}")
            if self.log_level == "DEBUG":
                import traceback
                traceback.print_exc()
            interaction_log["error"] = str(e)
        
        interaction_log["completed"] = task_complete or collision_occurred
        interaction_log["total_turns"] = len(interaction_log["turns"])
        
        # Final success evaluation (will check for collision)
        success_eval = TaskEvaluator.evaluate_task_success(
            world_model, 
            task['task_goal'], 
            interaction_log
        )
        interaction_log["success_evaluation"] = success_eval
        
        # Display success evaluation
        if self.log_level == "INFO":
            if success_eval['success']:
                print(f"✅ Task SUCCESS")
            else:
                print(f"❌ Task FAILED: {success_eval['failure_reason']}")
        elif self.log_level == "DEBUG":
            print(f"\n--- Success Evaluation ---")
            print(f"Success: {success_eval['success']}")
            print(f"Goal Conditions Met: {success_eval['goal_conditions_met']}/{success_eval['total_goal_conditions']}")
            print(f"Success Rate: {success_eval['success_rate']*100:.1f}%")
            if success_eval['failure_reason']:
                print(f"Failure Reason: {success_eval['failure_reason']}")
            print(f"Final Position: {success_eval['metrics']['final_position']}")
            print(f"Final Orientation: {success_eval['metrics']['final_orientation']}°")
        
        return interaction_log
    
    def run_experiments(self, num_episodes=10, output_file="synthetic_results.json"):
        """Run synthetic experiments for multiple episodes"""
        if self.log_level == "DEBUG":
            print(f"\n{'#'*70}")
            print(f"Starting Synthetic Experiments")
            print(f"Model: {self.model_name}")
            print(f"Episodes: {num_episodes}")
            print(f"Log Level: {self.log_level}")
            print(f"Output: {output_file}")
            print(f"{'#'*70}\n")
        else:
            print(f"\n{'#'*70}")
            print(f"Model: {self.model_name}")
            print(f"Running {num_episodes} Episodes")
            print(f"{'#'*70}\n")
        
        all_results = {
            "experiment_info": {
                "model_name": self.model_name,
                "start_time": datetime.now().isoformat(),
                "num_episodes": num_episodes,
                "simulation_mode": "Vague-to-Specific Dynamic User Agent (Multi-Step)",
                "ambiguity_filter": self.ambiguity_filter or "all",
                "total_tasks": len(self.task_scenarios)
            },
            "episodes": []
        }
        
        for episode_num in range(num_episodes):
            if self.log_level == "DEBUG":
                print(f"\n\n{'#'*70}")
                print(f"EPISODE {episode_num + 1}/{num_episodes}")
                print(f"{'#'*70}")
            else:
                print(f"\n{'='*70}")
                print(f"EPISODE {episode_num + 1}/{num_episodes}")
            
            task = random.choice(self.task_scenarios)
            interaction_log = self.simulate_interaction(task)
            
            episode_result = {
                "episode_num": episode_num + 1,
                "interaction": interaction_log
            }
            
            all_results["episodes"].append(episode_result)
            time.sleep(0.5)
        
        # Save results
        all_results["experiment_info"]["end_time"] = datetime.now().isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Calculate statistics
        completed_tasks = sum(1 for ep in all_results['episodes'] 
                             if ep['interaction'].get('completed', False))
        
        successful_tasks = sum(
            1 for ep in all_results['episodes']
            if ep['interaction'].get('success_evaluation', {}).get('success', False)
        )
        
        # Count collisions
        collision_count = sum(
            1 for ep in all_results['episodes']
            if ep['interaction'].get('collision') is not None
        )
        
        total_turns = sum(ep['interaction'].get('total_turns', 0) 
                         for ep in all_results['episodes'])
        
        clarifications = sum(
            1 for ep in all_results['episodes']
            for turn in ep['interaction'].get('turns', [])
            if 'Misty asked:' in turn.get('robot_action_description', '')
        )
        
        avg_goal_success = sum(
            ep['interaction'].get('success_evaluation', {}).get('success_rate', 0)
            for ep in all_results['episodes']
        ) / num_episodes if num_episodes > 0 else 0
        
        print(f"\n\n{'#'*70}")
        print(f"Experiment Complete!")
        print(f"{'#'*70}")
        print(f"Results saved to: {output_file}")
        print(f"Episodes completed: {completed_tasks}/{num_episodes} ({completed_tasks/num_episodes*100:.1f}%)")
        print(f"Tasks successful: {successful_tasks}/{num_episodes} ({successful_tasks/num_episodes*100:.1f}%)")
        print(f"Collisions: {collision_count}/{num_episodes} ({collision_count/num_episodes*100:.1f}%)")
        print(f"Avg goal condition success: {avg_goal_success*100:.1f}%")
        print(f"Total turns: {total_turns}")
        print(f"Clarifications: {clarifications} ({clarifications/total_turns*100:.1f}% of turns)" if total_turns > 0 else "Clarifications: 0")
        print(f"Avg turns/episode: {total_turns/num_episodes:.1f}")
        print(f"{'#'*70}\n")
        
        return all_results