"""
Main entry point for Misty robot EmbodiedPF system
Handles configuration and startup
Uses Groq (Llama) for text parsing and OpenAI (GPT-5 nano) for vision
"""
from model import MistyController
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
IP_ADDRESS = "172.20.10.2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For GPT-5 nano vision

# TEST MODE: Set to True to type commands instead of using voice
TEST_MODE = True  # Change to False for normal voice mode


def main():
    """Main entry point"""
    
    # Validate configuration
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in environment variables")
        return
    
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not found. Vision features will be disabled.")
    
    # Create controller
    controller = MistyController(
        ip_address=IP_ADDRESS,
        groq_api_key=GROQ_API_KEY,
        openai_api_key=OPENAI_API_KEY  # Changed from claude_api_key
    )
    
    # Run controller
    try:
        controller.run(test_mode=TEST_MODE)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()