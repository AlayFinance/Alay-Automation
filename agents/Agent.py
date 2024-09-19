import os
import google.generativeai as genai
import yaml

class AIAgent:
    def __init__(self, config_file='config.yaml', model_name="gemini-1.5-flash", generation_config=None, role="You are an AI Agent for Oona VIP Car Insurance"):
        # Load configuration from YAML
        self.config = self.load_config(config_file)
        # Configure the Gemini API
        genai.configure(api_key=self.config["GEMINI_API_KEY"])
        
        # Define the model and its generation configuration
        self.generation_config = generation_config if generation_config else {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self.model_name = model_name
        self.system_instruction = role
        
        # Create the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            system_instruction=self.system_instruction,
        )
        self.chat_session = None
    
    @staticmethod
    def load_config(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def start_chat_session(self, history=None):
        """Start a new chat session with optional history."""
        history = history if history else []
        self.chat_session = self.model.start_chat(history=history)
    
    def send_message(self, message):
        """Send a message to the chat session and return the model's response."""
        if not self.chat_session:
            raise ValueError("Chat session not started. Use start_chat_session() first.")
        return self.chat_session.send_message(message).text
    
    def set_system_instruction(self, instruction):
        """Customize the system instruction for the AI model."""
        self.system_instruction = instruction
        # Reinitialize the model with the new instruction
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            system_instruction=self.system_instruction,
        )
    
    def set_generation_config(self, new_config):
        """Update the generation configuration of the model."""
        self.generation_config.update(new_config)
        # Reinitialize the model with the new generation config
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            system_instruction=self.system_instruction,
        )

# Example Usage:
agent = AIAgent()

# Customize the system instruction
agent.set_system_instruction("Generate content in a formal tone. Use markdown formatting.")

# Start a chat session and send a message
agent.start_chat_session(history=[{"role": "user", "parts": ["Generate a marketing copy for a car insurance company."]}])
response = agent.send_message("Create a FB post for Oona VIP Car Insurance")

print(response)
