"""
Conversation Logger module
Logs user input, VLM raw output, and Misty's speech in turn-based format
"""
import os
from datetime import datetime


class ConversationLogger:
    def __init__(self, log_dir="logs", session_name=None):
        """
        Initialize conversation logger
        
        Args:
            log_dir: Directory to save log files
            session_name: Optional custom session name, otherwise uses timestamp
        """
        self.log_dir = log_dir
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate log filename
        if session_name:
            filename = f"{session_name}.txt"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.txt"
        
        self.log_path = os.path.join(log_dir, filename)
        
        # Initialize turn tracking
        self.current_turn = 0
        self.turn_data = {}
        
        print(f"Conversation logging initialized: {self.log_path}")
        
        # Write header
        with open(self.log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Misty Conversation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def start_turn(self, user_input):
        """
        Start a new conversation turn with user input
        
        Args:
            user_input: The user's command/speech text
        """
        self.current_turn += 1
        self.turn_data = {
            'user': user_input,
            'vlm_output': None,
            'misty_responses': []
        }
        
        print(f"[Logger] Turn {self.current_turn} started - User: {user_input}")
    
    def log_vlm_output(self, vlm_raw_output):
        """
        Log the raw VLM output
        
        Args:
            vlm_raw_output: Raw string output from the VLM
        """
        if self.current_turn == 0:
            print("[Logger Warning] VLM output logged before turn started")
            return
        
        self.turn_data['vlm_output'] = vlm_raw_output
        print(f"[Logger] VLM output logged for Turn {self.current_turn}")
    
    def log_misty_speech(self, speech_text):
        """
        Log what Misty said
        
        Args:
            speech_text: Text that Misty spoke
        """
        if self.current_turn == 0:
            print("[Logger Warning] Misty speech logged before turn started")
            return
        
        self.turn_data['misty_responses'].append(speech_text)
        print(f"[Logger] Misty speech logged: {speech_text}")
    
    def end_turn(self):
        """
        End the current turn and write to log file
        """
        if self.current_turn == 0:
            print("[Logger Warning] Attempting to end turn that hasn't started")
            return
        
        # Write turn to file
        self._write_turn_to_file()
        
        print(f"[Logger] Turn {self.current_turn} completed and saved\n")
    
    def _write_turn_to_file(self):
        """Internal method to write current turn data to file"""
        with open(self.log_path, 'a') as f:
            f.write(f"Turn {self.current_turn}:\n")
            f.write("-" * 80 + "\n")
            
            # User input
            f.write(f"User: {self.turn_data['user']}\n\n")
            
            # VLM output
            if self.turn_data['vlm_output']:
                f.write(f"VLM_raw_output: {self.turn_data['vlm_output']}\n\n")
            else:
                f.write("VLM_raw_output: [None]\n\n")
            
            # Misty responses
            if self.turn_data['misty_responses']:
                f.write("Misty:\n")
                for i, response in enumerate(self.turn_data['misty_responses'], 1):
                    if len(self.turn_data['misty_responses']) > 1:
                        f.write(f"  {i}. {response}\n")
                    else:
                        f.write(f"  {response}\n")
            else:
                f.write("Misty: [No speech]\n")
            
            f.write("\n")
    
    def add_note(self, note):
        """
        Add a note to the log file (for errors, special events, etc.)
        
        Args:
            note: Note text to add
        """
        with open(self.log_path, 'a') as f:
            f.write(f"[NOTE - {datetime.now().strftime('%H:%M:%S')}] {note}\n\n")
        
        print(f"[Logger] Note added: {note}")
    
    def finalize(self):
        """
        Finalize the log file with summary statistics
        """
        with open(self.log_path, 'a') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Session ended - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total turns: {self.current_turn}\n")
            f.write("=" * 80 + "\n")
        
        print(f"[Logger] Session finalized. Total turns: {self.current_turn}")
        print(f"[Logger] Log saved to: {self.log_path}")