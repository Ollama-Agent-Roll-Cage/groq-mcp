"""
Example Groq MCP Server for VS Code

This example demonstrates how to create an MCP server for VS Code that uses
Groq LLMs instead of Claude for AI functionality.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Now you can import from oarc_groq_mcp
from oarc_groq_mcp import GroqVSCodeMCP, GroqModel

def main():
    # Create a Groq MCP server
    # The API key will be loaded from the GROQ_API_KEY environment variable
    server = GroqVSCodeMCP(
        name="GroqCopilotTools",
        description="VS Code Copilot tools powered by Groq LLMs",
        default_model=GroqModel.LLAMA_3_70B  # Using the 70B model for high-quality responses
    )
    
    # Add an input variable for the Groq API key (if not using environment variable)
    server.add_input(
        input_id="groq-api-key",
        description="Groq API Key",
        password=True  # Will be stored securely by VS Code
    )
    
    # Add a general-purpose chat tool
    server.add_chat_tool(
        name="ask_groq",
        description="Ask Groq a question and get a response",
        system_prompt="You are a helpful, knowledgeable assistant. Provide clear and accurate information.",
        model=GroqModel.LLAMA_3_70B
    )
    
    # Add a coding assistant tool
    server.add_code_tool(
        name="code_with_groq",
        description="Generate code using Groq LLM expertise",
        system_prompt=(
            "You are an expert programmer proficient in all major programming languages. "
            "Provide clean, efficient, and well-documented code. "
            "Include explanations of your approach."
        ),
        model=GroqModel.DEEPSEEK_DISTILL_LLAMA_70B  # Using a specialized code model
    )
    
    # Add a specialized research tool
    server.add_chat_tool(
        name="research_with_groq",
        description="Research a topic in depth with Groq",
        system_prompt=(
            "You are a research assistant with expertise across multiple domains. "
            "Provide thorough, well-structured analysis with references where applicable. "
            "Consider different perspectives and highlight key insights."
        ),
        model=GroqModel.LLAMA_3_70B
    )
    
    # Add a creative writing tool
    server.add_chat_tool(
        name="write_with_groq",
        description="Generate creative content with Groq",
        system_prompt=(
            "You are a creative writing assistant with expertise in various styles and formats. "
            "Generate engaging, original content based on the user's request."
        ),
        model=GroqModel.LLAMA_3_8B  # Using the 8B model for faster creative responses
    )
    
    # Set up the VS Code workspace
    mcp_json_path, server_script_path = server.setup_vscode_workspace()
    
    print(f"MCP configuration saved to: {mcp_json_path}")
    print(f"Server script saved to: {server_script_path}")
    
    # Generate a VS Code URL to install the server
    url = server.generate_vscode_url()
    print(f"VS Code URL: {url}")
    
    print("\nInstructions:")
    print("1. Set your Groq API key using: export GROQ_API_KEY=your_api_key")
    print("2. Open VS Code and run the command 'MCP: List Servers'")
    print("3. Select 'GroqCopilotTools' and click 'Start Server'")
    print("4. Open the Chat View and select 'Agent mode'")
    print("5. Use the 'Tools' button to select your Groq-powered tools")

if __name__ == "__main__":
    main()
