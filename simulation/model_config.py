"""
Model Configuration
Defines different model variants for baseline comparison experiments
"""
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ModelConfig:
    """Configuration for different model variants"""
    
    # Available models
    FRICTION_ENABLED = "friction_enabled"
    NO_FRICTION = "no_friction"
    ZERO_SHOT_VLM = "zero_shot_vlm"
    ZERO_SHOT_MULTITURN = "zero_shot_multiturn"
    
    @staticmethod
    def get_model(model_type):
        """
        Get the appropriate LLM layer based on model type
        
        Args:
            model_type: One of the ModelConfig constants
            
        Returns:
            Initialized LLM layer
        """
        if model_type == ModelConfig.FRICTION_ENABLED:
            from model.llm_layer import LLMLayer
            return LLMLayer(openai_api_key=OPENAI_API_KEY)
        
        elif model_type == ModelConfig.NO_FRICTION:
            from model.llm_without_friction import LLMLayerNoFriction
            return LLMLayerNoFriction(openai_api_key=OPENAI_API_KEY)
        
        elif model_type == ModelConfig.ZERO_SHOT_VLM:
            from model.llm_zero_shot import LLMLayerZeroShot
            return LLMLayerZeroShot(openai_api_key=OPENAI_API_KEY)
        
        elif model_type == ModelConfig.ZERO_SHOT_MULTITURN:
            from model.llm_zero_shot_multiturn import LLMLayerZeroShotMultiTurn
            return LLMLayerZeroShotMultiTurn(openai_api_key=OPENAI_API_KEY)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_description(model_type):
        """Get human-readable description of model"""
        descriptions = {
            ModelConfig.FRICTION_ENABLED: "GPT-5 nano with Positive Friction",
            ModelConfig.NO_FRICTION: "GPT-5 nano without Friction (Control)",
            ModelConfig.ZERO_SHOT_VLM: "Zero-shot VLM (No History, No Friction)",
            ModelConfig.ZERO_SHOT_MULTITURN: "Zero-shot VLM + Multi-turn (History, No Friction)",
        }
        return descriptions.get(model_type, "Unknown Model")
    
    @staticmethod
    def get_all_models():
        """Get list of all available model types"""
        return [
            ModelConfig.FRICTION_ENABLED,
            ModelConfig.NO_FRICTION,
            ModelConfig.ZERO_SHOT_VLM,
            ModelConfig.ZERO_SHOT_MULTITURN,
        ]