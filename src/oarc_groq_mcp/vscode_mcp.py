"""
VSCodeMCP - A simple Python module for creating and managing MCP servers for VS Code

This module makes it easy to define and add Model Context Protocol (MCP) servers 
to Visual Studio Code's Copilot agent mode. It combines functionality for defining
tools, resources, and prompts, and provides methods to install them as VS Code MCP servers.

Author: Your Name
Version: 1.0.0
"""

import os
import sys
import json
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import argparse

class ServerType(Enum):
    """Enum for MCP server connection types."""
    STDIO = "stdio"
    SSE = "sse"

class VSCodeMCP:
    """
    A simple class for creating and managing MCP servers for Visual Studio Code.
    
    This class provides methods to:
    - Define tools, resources, and prompts
    - Generate configuration for VS Code
    - Install servers using VS Code's command line interface
    - Create shareable configuration files for your workspace
    """
    
    def __init__(
        self, 
        name: str, 
        description: str = None,
        server_type: ServerType = ServerType.STDIO,
        command: str = "python",
        args: List[str] = None,
        env: Dict[str, str] = None,
        url: str = None,
        headers: Dict[str, str] = None,
        env_file: str = None,
        workspace_folder: str = None
    ):
        """
        Initialize a new VSCodeMCP instance.
        
        Args:
            name: Name of the MCP server
            description: Description of the server (optional)
            server_type: Type of server connection (stdio or sse)
            command: Command to start the server executable (for stdio)
            args: Arguments to pass to the command (for stdio)
            env: Environment variables for the server (for stdio)
            url: URL of the server (for sse)
            headers: HTTP headers for the server (for sse)
            env_file: Path to an .env file for environment variables
            workspace_folder: Path to the workspace folder
        """
        self.name = name
        self.description = description or f"MCP Server: {name}"
        self.server_type = server_type
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.url = url
        self.headers = headers or {}
        self.env_file = env_file
        self.workspace_folder = workspace_folder or os.getcwd()
        
        # Track custom inputs that require user interaction
        self.inputs = []
        
        # Track tools, resources, and prompts
        self.tools = []
        self.resources = []
        self.prompts = []
        
        # Internal server script content
        self._server_script = None
    
    def add_input(self, input_id: str, description: str, password: bool = False, default_value: str = None):
        """
        Add an input variable that VS Code will prompt the user for.
        
        Args:
            input_id: Unique identifier for the input
            description: Description of the input to show to the user
            password: Whether the input should be treated as a password
            default_value: Default value for the input
        """
        input_config = {
            "type": "promptString",
            "id": input_id,
            "description": description
        }
        
        if password:
            input_config["password"] = True
            
        if default_value:
            input_config["default"] = default_value
            
        self.inputs.append(input_config)
        return self
    
    def add_tool(self, name: str, description: str, function=None):
        """
        Add a tool to the MCP server.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            function: The Python function that implements the tool (optional)
        """
        tool = {
            "name": name,
            "description": description,
            "function": function
        }
        
        self.tools.append(tool)
        return self
    
    def add_resource(self, uri: str, description: str, function=None):
        """
        Add a resource to the MCP server.
        
        Args:
            uri: URI template for the resource
            description: Description of the resource
            function: The Python function that implements the resource (optional)
        """
        resource = {
            "uri": uri,
            "description": description,
            "function": function
        }
        
        self.resources.append(resource)
        return self
    
    def add_prompt(self, name: str, description: str, function=None):
        """
        Add a prompt to the MCP server.
        
        Args:
            name: Name of the prompt
            description: Description of the prompt
            function: The Python function that implements the prompt (optional)
        """
        prompt = {
            "name": name,
            "description": description,
            "function": function
        }
        
        self.prompts.append(prompt)
        return self
    
    def _generate_server_config(self):
        """
        Generate the server configuration for VS Code.
        
        Returns:
            Dict: Server configuration
        """
        if self.server_type == ServerType.STDIO:
            config = {
                "type": "stdio",
                "command": self.command,
                "args": self.args
            }
            
            if self.env:
                config["env"] = self.env
                
            if self.env_file:
                config["envFile"] = self.env_file
                
        elif self.server_type == ServerType.SSE:
            config = {
                "type": "sse",
                "url": self.url
            }
            
            if self.headers:
                config["headers"] = self.headers
        
        return config
    
    def generate_mcp_json(self):
        """
        Generate the mcp.json configuration for VS Code.
        
        Returns:
            Dict: Complete mcp.json configuration
        """
        config = {
            "servers": {
                self.name: self._generate_server_config()
            }
        }
        
        if self.inputs:
            config["inputs"] = self.inputs
            
        return config
    
    def _replace_placeholders(self, text):
        """
        Replace placeholders in text with actual values.
        
        Args:
            text: Text with placeholders
            
        Returns:
            str: Text with placeholders replaced
        """
        if isinstance(text, str):
            # Replace ${workspaceFolder} with actual workspace folder
            return text.replace("${workspaceFolder}", self.workspace_folder)
        return text
    
    def save_mcp_json(self, path=None):
        """
        Save the mcp.json configuration to a file.
        
        Args:
            path: Path to save the file (defaults to .vscode/mcp.json in workspace)
            
        Returns:
            str: Path to the saved file
        """
        if path is None:
            vscode_dir = os.path.join(self.workspace_folder, ".vscode")
            os.makedirs(vscode_dir, exist_ok=True)
            path = os.path.join(vscode_dir, "mcp.json")
        
        # Load existing config if it exists
        if os.path.exists(path):
            with open(path, 'r') as f:
                existing_config = json.load(f)
        else:
            existing_config = {}
        
        # Generate new config
        new_config = self.generate_mcp_json()
        
        # Merge configs
        if "inputs" in existing_config:
            if "inputs" not in new_config:
                new_config["inputs"] = []
            
            # Add inputs that don't already exist
            existing_input_ids = [inp["id"] for inp in existing_config["inputs"]]
            for inp in new_config["inputs"]:
                if inp["id"] not in existing_input_ids:
                    existing_config["inputs"].append(inp)
        elif "inputs" in new_config:
            existing_config["inputs"] = new_config["inputs"]
        
        # Add or update server
        if "servers" not in existing_config:
            existing_config["servers"] = {}
            
        existing_config["servers"][self.name] = new_config["servers"][self.name]
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(existing_config, f, indent=2)
            
        print(f"MCP configuration saved to: {path}")
        return path
    
    def generate_server_script(self):
        """
        Generate a Python script for the MCP server.
        
        Returns:
            str: Python script content
        """
        script = """#!/usr/bin/env python3
# MCP Server for VS Code: {name}
# Generated by VSCodeMCP

from fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("{name}")

""".format(name=self.name)
        
        # Add tools
        for tool in self.tools:
            script += """
@mcp.tool()
def {name}(params):
    \"""
    {description}
    \"""
    # Tool implementation
    return {{"content": "This is a placeholder implementation for {name}. Replace this with your actual implementation."}}
""".format(name=tool["name"], description=tool["description"])
        
        # Add resources
        for resource in self.resources:
            script += """
@mcp.resource("{uri}")
def {name}(params):
    \"""
    {description}
    \"""
    # Resource implementation
    return "This is a placeholder implementation for {uri}. Replace this with your actual implementation."
""".format(name=resource["uri"].split("/")[-1].split(":")[-1], uri=resource["uri"], description=resource["description"])
        
        # Add prompts
        for prompt in self.prompts:
            script += """
@mcp.prompt()
def {name}(params):
    \"""
    {description}
    \"""
    # Prompt implementation
    return "This is a placeholder implementation for {name}. Replace this with your actual implementation."
""".format(name=prompt["name"], description=prompt["description"])
        
        # Add server run code
        script += """
if __name__ == "__main__":
    mcp.run()
"""
        
        self._server_script = script
        return script
    
    def save_server_script(self, path=None):
        """
        Save the server script to a file.
        
        Args:
            path: Path to save the file (defaults to server.py in current directory)
            
        Returns:
            str: Path to the saved file
        """
        if path is None:
            path = os.path.join(self.workspace_folder, f"{self.name.lower().replace(' ', '_')}_server.py")
            
        if self._server_script is None:
            self.generate_server_script()
            
        with open(path, 'w') as f:
            f.write(self._server_script)
            
        # Make the script executable
        os.chmod(path, 0o755)
            
        print(f"MCP server script saved to: {path}")
        return path
    
    def install_to_vscode(self, user_profile=False):
        """
        Install the MCP server to VS Code using the command line interface.
        
        Args:
            user_profile: Whether to install to user profile instead of workspace
            
        Returns:
            bool: True if successful, False otherwise
        """
        server_config = {
            "name": self.name
        }
        
        stdio_config = self._generate_server_config()
        for key, value in stdio_config.items():
            server_config[key] = value
            
        # Create the JSON string for the VS Code CLI
        config_str = json.dumps(server_config).replace('"', '\\"')
        
        # Check if VS Code is installed
        code_command = None
        for cmd in ["code", "code-insiders"]:
            try:
                subprocess.run([cmd, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                code_command = cmd
                break
            except FileNotFoundError:
                continue
                
        if code_command is None:
            print("Error: Visual Studio Code (code or code-insiders) not found in PATH.")
            return False
            
        # Run the VS Code command to add the MCP server
        try:
            subprocess.run([code_command, "--add-mcp", f"{config_str}"])
            print(f"MCP server '{self.name}' installed to VS Code {('user profile' if user_profile else 'workspace')}.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing MCP server to VS Code: {e}")
            return False
    
    def generate_vscode_url(self):
        """
        Generate a VS Code URL to install the MCP server.
        
        Returns:
            str: VS Code URL
        """
        server_config = {
            "name": self.name
        }
        
        stdio_config = self._generate_server_config()
        for key, value in stdio_config.items():
            server_config[key] = value
            
        # Create the JSON string for the URL
        config_str = json.dumps(server_config)
        
        # Create the URL
        url = f"vscode:mcp/install?{config_str}"
        
        return url
    
    def setup_vscode_workspace(self, path=None, create_server_script=True):
        """
        Set up the VS Code workspace with MCP configuration.
        
        Args:
            path: Path to the workspace folder (defaults to current directory)
            create_server_script: Whether to create the server script
            
        Returns:
            tuple: (mcp_json_path, server_script_path)
        """
        workspace_path = path or self.workspace_folder
        self.workspace_folder = workspace_path
        
        # Create the .vscode directory
        vscode_dir = os.path.join(workspace_path, ".vscode")
        os.makedirs(vscode_dir, exist_ok=True)
        
        # Save the mcp.json file
        mcp_json_path = self.save_mcp_json()
        
        server_script_path = None
        if create_server_script:
            # Save the server script
            server_script_path = self.save_server_script()
            
            # Update the args in mcp.json to use the server script
            with open(mcp_json_path, 'r') as f:
                mcp_config = json.load(f)
                
            # Get the relative path from workspace folder to server script
            rel_path = os.path.relpath(server_script_path, workspace_path)
            
            # Update the args
            if self.server_type == ServerType.STDIO:
                mcp_config["servers"][self.name]["args"] = [rel_path]
                
                # Save the updated config
                with open(mcp_json_path, 'w') as f:
                    json.dump(mcp_config, f, indent=2)
        
        return (mcp_json_path, server_script_path)

def create_fastmcp_server(name, description=None, path=None):
    """
    Create a FastMCP server for VS Code.
    
    Args:
        name: Name of the server
        description: Description of the server
        path: Path to save the server files
        
    Returns:
        VSCodeMCP: VSCodeMCP instance
    """
    # Check if fastmcp is installed
    try:
        import fastmcp
    except ImportError:
        print("Warning: 'fastmcp' package not found. You'll need to install it with:")
        print("  pip install fastmcp")
    
    # Create VSCodeMCP instance
    server = VSCodeMCP(
        name=name,
        description=description,
        server_type=ServerType.STDIO,
        command="python",
        workspace_folder=path or os.getcwd()
    )
    
    return server

def cli():
    """
    Command-line interface for VSCodeMCP.
    """
    parser = argparse.ArgumentParser(description="VSCodeMCP - Create and manage MCP servers for VS Code")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new MCP server")
    create_parser.add_argument("name", help="Name of the server")
    create_parser.add_argument("--description", "-d", help="Description of the server")
    create_parser.add_argument("--path", "-p", help="Path to save the server files")
    create_parser.add_argument("--tool", "-t", action="append", nargs=2, metavar=("NAME", "DESCRIPTION"), 
                              help="Add a tool to the server")
    create_parser.add_argument("--input", "-i", action="append", nargs=3, metavar=("ID", "DESCRIPTION", "PASSWORD"),
                              help="Add an input to the server")
    create_parser.add_argument("--install", action="store_true", help="Install the server to VS Code")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install an existing MCP server to VS Code")
    install_parser.add_argument("path", help="Path to the .vscode/mcp.json file or server script")
    install_parser.add_argument("--user", "-u", action="store_true", help="Install to user profile instead of workspace")
    
    args = parser.parse_args()
    
    if args.command == "create":
        # Create a new server
        server = create_fastmcp_server(args.name, args.description, args.path)
        
        # Add tools
        if args.tool:
            for name, description in args.tool:
                server.add_tool(name, description)
                
        # Add inputs
        if args.input:
            for input_id, description, password in args.input:
                server.add_input(input_id, description, password.lower() == "true")
                
        # Set up the workspace
        mcp_json_path, server_script_path = server.setup_vscode_workspace()
        
        # Install to VS Code if requested
        if args.install:
            server.install_to_vscode()
            
        print("\nSetup complete! To use your MCP server in VS Code:")
        print("1. Open the workspace folder in VS Code")
        print("2. In VS Code, go to View > Command Palette (Ctrl+Shift+P)")
        print("3. Run the command 'MCP: List Servers'")
        print("4. Select your server and click 'Start Server'")
        print("5. Open the Chat View (Ctrl+Alt+I) and select 'Agent mode'")
        print("6. Use the 'Tools' button to select your MCP tools\n")
            
    elif args.command == "install":
        # Install an existing server
        path = os.path.abspath(args.path)
        
        if os.path.isdir(path):
            # Check for .vscode/mcp.json
            mcp_json_path = os.path.join(path, ".vscode", "mcp.json")
            if os.path.exists(mcp_json_path):
                path = mcp_json_path
            else:
                print(f"Error: Could not find .vscode/mcp.json in {path}")
                return
                
        if os.path.isfile(path):
            if path.endswith("mcp.json"):
                with open(path, 'r') as f:
                    config = json.load(f)
                    
                if "servers" in config and config["servers"]:
                    server_name = list(config["servers"].keys())[0]
                    server_config = config["servers"][server_name]
                    
                    # Create a VSCodeMCP instance from the config
                    server = VSCodeMCP(
                        name=server_name,
                        server_type=ServerType.STDIO if server_config.get("type") == "stdio" else ServerType.SSE,
                        command=server_config.get("command"),
                        args=server_config.get("args", []),
                        env=server_config.get("env", {}),
                        url=server_config.get("url"),
                        headers=server_config.get("headers", {}),
                        env_file=server_config.get("envFile"),
                        workspace_folder=os.path.dirname(os.path.dirname(path))
                    )
                    
                    # Install to VS Code
                    server.install_to_vscode(args.user)
                else:
                    print(f"Error: No servers found in {path}")
            else:
                # Assume it's a server script
                server_name = os.path.basename(path).replace(".py", "")
                
                server = VSCodeMCP(
                    name=server_name,
                    server_type=ServerType.STDIO,
                    command="python",
                    args=[os.path.basename(path)],
                    workspace_folder=os.path.dirname(path)
                )
                
                # Install to VS Code
                server.install_to_vscode(args.user)
        else:
            print(f"Error: Path not found: {path}")
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()
