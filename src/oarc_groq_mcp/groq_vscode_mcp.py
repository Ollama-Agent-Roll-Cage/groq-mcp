"""
GroqVSCodeMCP - Combine Groq LLMs with VS Code MCP servers

This module integrates Groq's powerful LLMs with the VSCodeMCP module to create
MCP servers that use Groq instead of Claude for their AI functionality.

Author: Your Name
Version: 1.0.0
"""

import os
import sys
import json
import base64
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable

# Import VSCodeMCP
try:
    from oarc_groq_mcp.vscode_mcp import VSCodeMCP, ServerType
except ImportError:
    raise ImportError("VSCodeMCP is not installed. Please ensure it's in your Python path.")

# Import Groq
try:
    from groq import Groq
except ImportError:
    raise ImportError("The Groq Python package is not installed. Please install it using: pip install groq")

class GroqModel(Enum):
    """Available Groq models for MCP tools."""
    
    # Text models
    LLAMA_3_8B = "llama-3-8b"
    LLAMA_3_70B = "llama-3-70b"
    LLAMA_3_1_8B = "llama-3.1-8b"
    LLAMA_3_1_70B = "llama-3.1-70b"
    LLAMA_3_2_11B = "llama-3.2-11b"
    LLAMA_3_3_70B_VERSATILE = "llama-3.3-70b-versatile"
    
    # Vision models
    LLAMA_3_1_8B_VISION = "llama-3.1-8b-vision"
    LLAMA_3_1_70B_VISION = "llama-3.1-70b-vision"
    LLAMA_3_2_11B_VISION = "llama-3.2-11b-vision-preview"
    
    # Code models
    QWEN_CODER = "qwen-coder"
    DEEPSEEK_DISTILL_LLAMA_70B = "deepseek-r1-distill-llama-70b"


class GroqVSCodeMCP:
    """
    Integrate Groq LLMs with VSCodeMCP to create MCP servers that use Groq for AI functionality.
    
    This class combines the functionality of VSCodeMCP for creating MCP servers with
    Groq's powerful LLMs for text generation, code completion, and other AI capabilities.
    """
    
    def __init__(
        self,
        name: str,
        description: str = None,
        groq_api_key: str = None,
        default_model: GroqModel = GroqModel.LLAMA_3_8B,
        command: str = "python",
        env: Dict[str, str] = None,
        workspace_folder: str = None
    ):
        """
        Initialize a new GroqVSCodeMCP instance.
        
        Args:
            name: Name of the MCP server
            description: Description of the server
            groq_api_key: Groq API key (defaults to GROQ_API_KEY environment variable)
            default_model: Default Groq model to use
            command: Command to run the server script
            env: Environment variables for the server
            workspace_folder: Path to the workspace folder
        """
        self.name = name
        self.description = description or f"Groq MCP Server: {name}"
        self.default_model = default_model
        
        # Initialize Groq client
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key not provided and not found in GROQ_API_KEY environment variable.")
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Initialize VSCodeMCP
        self.env = env or {}
        self.env["GROQ_API_KEY"] = self.groq_api_key
        
        self.vscode_mcp = VSCodeMCP(
            name=name,
            description=description,
            server_type=ServerType.STDIO,
            command=command,
            env=self.env,
            workspace_folder=workspace_folder
        )
        
        # Track Groq-specific tools
        self.groq_tools = []
    
    def add_input(self, input_id: str, description: str, password: bool = False, default_value: str = None):
        """
        Add an input variable that VS Code will prompt the user for.
        
        Args:
            input_id: Unique identifier for the input
            description: Description of the input to show to the user
            password: Whether the input should be treated as a password
            default_value: Default value for the input
        """
        self.vscode_mcp.add_input(input_id, description, password, default_value)
        return self
    
    def add_chat_tool(
        self,
        name: str,
        description: str,
        system_prompt: str = "You are a helpful assistant.",
        model: GroqModel = None,
    ):
        """
        Add a chat tool that uses Groq for text generation.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            system_prompt: System prompt for the Groq model
            model: Groq model to use (defaults to the instance default_model)
        """
        tool = {
            "name": name,
            "description": description,
            "type": "chat",
            "system_prompt": system_prompt,
            "model": model or self.default_model
        }
        
        self.groq_tools.append(tool)
        self.vscode_mcp.add_tool(name, description)
        return self
    
    def add_code_tool(
        self,
        name: str,
        description: str,
        system_prompt: str = "You are an expert programmer.",
        model: GroqModel = GroqModel.DEEPSEEK_DISTILL_LLAMA_70B,
    ):
        """
        Add a code generation tool that uses Groq for code completion.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            system_prompt: System prompt for the Groq model
            model: Groq model to use (defaults to DEEPSEEK_DISTILL_LLAMA_70B)
        """
        tool = {
            "name": name,
            "description": description,
            "type": "code",
            "system_prompt": system_prompt,
            "model": model
        }
        
        self.groq_tools.append(tool)
        self.vscode_mcp.add_tool(name, description)
        return self
    
    def add_custom_tool(
        self,
        name: str,
        description: str,
        function: Callable = None,
    ):
        """
        Add a custom tool that doesn't use Groq.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            function: The Python function that implements the tool
        """
        tool = {
            "name": name,
            "description": description,
            "type": "custom",
            "function": function
        }
        
        self.groq_tools.append(tool)
        self.vscode_mcp.add_tool(name, description, function)
        return self
    
    def generate_server_script(self):
        """
        Generate a Python script for the MCP server with Groq integration.
        
        Returns:
            str: Python script content
        """
        script = """#!/usr/bin/env python3
# Groq MCP Server for VS Code: {name}
# Generated by GroqVSCodeMCP

import os
import json
from fastmcp import FastMCP
from groq import Groq
from typing import Dict, List, Any, Optional, Union

# Create MCP server
mcp = FastMCP("{name}")

# Initialize Groq client
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")
    
groq_client = Groq(api_key=groq_api_key)

""".format(name=self.name)

        # Add tools
        for tool in self.groq_tools:
            if tool["type"] == "chat":
                script += self._generate_chat_tool_code(tool)
            elif tool["type"] == "code":
                script += self._generate_code_tool_code(tool)
            elif tool["type"] == "custom":
                script += self._generate_custom_tool_code(tool)
        
        # Add server run code
        script += """
if __name__ == "__main__":
    mcp.run()
"""
        
        return script
    
    def _generate_chat_tool_code(self, tool):
        """Generate code for a chat tool."""
        return f"""
@mcp.tool()
def {tool["name"]}(prompt, **kwargs):
    \"\"\"
    {tool["description"]}
    
    Args:
        prompt: The user prompt to send to Groq
        **kwargs: Additional parameters
        
    Returns:
        Generated response from Groq
    \"\"\"
    # Prepare the request
    system_message = "{tool["system_prompt"]}"
    
    # Call Groq API
    completion = groq_client.chat.completions.create(
        model="{tool["model"].value}",
        messages=[
            {{"role": "system", "content": system_message}},
            {{"role": "user", "content": prompt}}
        ],
        temperature=0.7,
    )
    
    # Get the response text
    response_text = completion.choices[0].message.content
    
    return {{"content": response_text}}
"""
    
    def _generate_code_tool_code(self, tool):
        """Generate code for a code tool."""
        return f"""
@mcp.tool()
def {tool["name"]}(prompt, language=None, **kwargs):
    \"\"\"
    {tool["description"]}
    
    Args:
        prompt: Description of the code to generate
        language: Programming language (optional)
        **kwargs: Additional parameters
        
    Returns:
        Generated code from Groq
    \"\"\"
    # Prepare the request
    system_message = "{tool["system_prompt"]}"
    
    # Add language to the prompt if specified
    if language:
        user_prompt = f"Write {{language}} code to {{prompt}}"
    else:
        user_prompt = prompt
    
    # Call Groq API
    completion = groq_client.chat.completions.create(
        model="{tool["model"].value}",
        messages=[
            {{"role": "system", "content": system_message}},
            {{"role": "user", "content": user_prompt}}
        ],
        temperature=0.1,  # Lower temperature for more deterministic code generation
    )
    
    # Get the response text
    code = completion.choices[0].message.content
    
    return {{"content": code}}
"""
    
    def _generate_custom_tool_code(self, tool):
        """Generate code for a custom tool."""
        if tool.get("function") is None:
            return f"""
@mcp.tool()
def {tool["name"]}(**kwargs):
    \"\"\"
    {tool["description"]}
    \"\"\"
    # Placeholder implementation
    return {{"content": "This is a placeholder implementation for {tool["name"]}. Replace this with your actual implementation."}}
"""
        else:
            # This is just a placeholder - we can't actually serialize the function
            return f"""
@mcp.tool()
def {tool["name"]}(**kwargs):
    \"\"\"
    {tool["description"]}
    \"\"\"
    # Custom implementation goes here
    # Replace this with your actual implementation
    return {{"content": "Custom implementation for {tool["name"]}."}}
"""
    
    def save_server_script(self, path=None):
        """
        Save the server script to a file.
        
        Args:
            path: Path to save the file (defaults to server.py in current directory)
            
        Returns:
            str: Path to the saved file
        """
        script = self.generate_server_script()
        
        # Use VSCodeMCP's method but override the script content
        self.vscode_mcp._server_script = script
        return self.vscode_mcp.save_server_script(path)
    
    def setup_vscode_workspace(self, path=None):
        """
        Set up the VS Code workspace with MCP configuration and server script.
        
        Args:
            path: Path to the workspace folder (defaults to current directory)
            
        Returns:
            tuple: (mcp_json_path, server_script_path)
        """
        # Generate the server script first
        script = self.generate_server_script()
        self.vscode_mcp._server_script = script
        
        # Then set up the workspace using VSCodeMCP
        return self.vscode_mcp.setup_vscode_workspace(path)
    
    def save_mcp_json(self, path=None):
        """
        Save the mcp.json configuration to a file.
        
        Args:
            path: Path to save the file (defaults to .vscode/mcp.json in workspace)
            
        Returns:
            str: Path to the saved file
        """
        return self.vscode_mcp.save_mcp_json(path)
    
    def install_to_vscode(self, user_profile=False):
        """
        Install the MCP server to VS Code using the command line interface.
        
        Args:
            user_profile: Whether to install to user profile instead of workspace
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.vscode_mcp.install_to_vscode(user_profile)
    
    def generate_vscode_url(self):
        """
        Generate a VS Code URL to install the MCP server.
        
        Returns:
            str: VS Code URL
        """
        return self.vscode_mcp.generate_vscode_url()


# Example usage
if __name__ == "__main__":
    # Create a GroqVSCodeMCP instance
    groq_mcp = GroqVSCodeMCP(
        name="GroqMCPServer",
        description="MCP Server using Groq LLMs"
    )
    
    # Add a chat tool
    groq_mcp.add_chat_tool(
        name="chat_with_groq",
        description="Chat with a Groq LLM",
        system_prompt="You are a helpful, friendly AI assistant.",
        model=GroqModel.LLAMA_3_8B
    )
    
    # Add a code tool
    groq_mcp.add_code_tool(
        name="generate_code",
        description="Generate code using Groq LLM",
        system_prompt="You are an expert programmer. Write clean, efficient code."
    )
    
    # Set up the workspace
    mcp_json_path, server_script_path = groq_mcp.setup_vscode_workspace()
    
    print(f"MCP configuration saved to: {mcp_json_path}")
    print(f"Server script saved to: {server_script_path}")
