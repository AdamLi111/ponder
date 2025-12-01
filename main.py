"""
Main entry point for Misty robot EmbodiedPF system
Handles configuration and startup
Uses GPT-5 nano as unified VLM for both text and vision
"""
from model import MistyController
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
IP_ADDRESS = "172.20.10.2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For GPT-5 nano (text + vision)

# TEST MODE: Set to True to type commands instead of using voice
TEST_MODE = True  # Change to False for normal voice mode

# LAPTOP MICROPHONE MODE: Set to True to use laptop mic instead of Misty's mic
# Only applies when TEST_MODE = False
USE_LAPTOP_MIC = True  # Change to False to use Misty's built-in microphone


def main():
    """Main entry point"""
    
    # Validate configuration
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Create controller
    controller = MistyController(
        ip_address=IP_ADDRESS,
        openai_api_key=OPENAI_API_KEY,
        use_laptop_mic=USE_LAPTOP_MIC if not TEST_MODE else False
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