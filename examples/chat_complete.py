import argparse
import json
from typing import List, Dict, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
import sys
from openai import OpenAI
import re
from dynasor.cot import openai_chat_completion_stream
from dynasor.cot import effort_level

class OpenAIChatClient:
    def __init__(self, api_key: str, model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", base_url: str = "http://localhost:8000/v1"):
        """
        Initialize the OpenAI Chat Client.
        
        Args:
            api_key: OpenAI API key.
            model: The model name to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B).
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.console = Console()
        self.conversation_history: List[Dict] = [{"role": "system", "content": "You are a helpful AI assistant."}]
        
    def format_history(self, messages: List[Dict]) -> str:
        """
        Convert chat conversation history into a prompt string.
        """
        formatted = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                formatted += "" + content + "\n"
            elif role == "user":
                formatted += "<｜User｜>" + content + "\n"
            else:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                formatted += "<｜Assistant｜>" + content + "\n"
        return formatted + "<｜Assistant｜>"
    
    def add_to_history(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def display_message(self, content: str, role: str = "assistant"):
        """Display a message with proper formatting."""
        if role == "assistant":
            self.console.print(Markdown(content))
        else:
            self.console.print(f"\n[bold blue]You:[/bold blue] {content}\n")
    
    def chat(self, dynasor_saving_effort: str = "mid"):
        """Main chat loop."""
        self.console.print("[bold green]Welcome to Dynasor Chat! Type 'exit' to end the conversation.[/bold green]\n")
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("> ")
                if user_input.lower() in ['exit', 'quit']:
                    self.console.print("[bold green]Goodbye![/bold green]")
                    break

                # Display and store the user message
                self.display_message(user_input, "user")
                self.add_to_history("user", user_input)

                # Convert conversation history to a prompt and get streaming response using the OpenAI client
                prompt = self.format_history(self.conversation_history)
                
                if dynasor_saving_effort != "":
                    full_response = ""
                    for chunk in openai_chat_completion_stream(self.client, self.model, prompt, dynasor_saving_effort=effort_level(dynasor_saving_effort)):
                        full_response += chunk
                        self.console.print(chunk, end='')
                else:
                    full_response = ""
                    for chunk in openai_chat_completion_stream(self.client, self.model, prompt):
                        full_response += chunk
                        self.console.print(chunk, end='')
                
                self.console.print()  # New line after response
                self.add_to_history("assistant", full_response)
                
            except KeyboardInterrupt:
                self.console.print("\n[bold red]Interrupted by user[/bold red]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                self.console.print("[yellow]Please check if your OpenAI API key is correct and the service is accessible.[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="OpenAI Chat Client")
    parser.add_argument("--api-key", type=str, default="token-abc123",
                        help="API key")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Model name (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                        help="Base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--dynasor-saving-effort", type=lambda s: s if s == "" or s in ["mild", "low", "mid", "high", "crazy"] else (_ for _ in ()).throw(argparse.ArgumentTypeError("Value must be either empty or one of: 'mild', 'low', 'mid', 'high', 'crazy'")), default="",
                        help="Specify the Dynasor saving effort level. If non-empty, must be one of: 'mild', 'low', 'mid', 'high', or 'crazy'. (Default: empty)")
    args = parser.parse_args()
    
    try:
        client = OpenAIChatClient(api_key=args.api_key, model=args.model, base_url=args.base_url)
        client.chat(dynasor_saving_effort=args.dynasor_saving_effort)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()