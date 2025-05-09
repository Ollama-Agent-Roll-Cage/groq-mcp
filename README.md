# VSCodeMCP


TODO RETIRE OR TURN INTO JUST A VSCODE SCRIPT IF THE OTHER GROQ-MCP-SERVER FOR DOESNT WORK OUT THE BOX WITH VSCODE COPILOT THEN WE NEED TO FIGURE OUT WHAT THAT ISSUE IS

A Python module for creating and managing MCP (Model Context Protocol) servers for Visual Studio Code's Copilot agent feature, with optional Groq LLM integration.

## Overview

The VSCodeMCP module makes it easy to:

1. Define MCP tools, resources, and prompts for VS Code's Copilot agent
2. Generate configuration files for VS Code
3. Install servers using VS Code's command line interface
4. Create shareable configuration files for your workspace 
5. Integrate with Groq LLMs for powerful AI-driven tools

## Features

- 🛠️ **Easy Tool Definition**: Define MCP tools with simple Python functions
- 📊 **VS Code Configuration**: Generate VS Code config files automatically
- 🔗 **FastMCP Integration**: Seamless integration with the FastMCP library
- 🧠 **Groq LLM Support**: Use Groq's powerful LLMs in your MCP tools
- 📱 **URL Generation**: Generate VS Code URLs for easy server installation
- 🔐 **Secure Input Handling**: Manage API keys and sensitive inputs securely
- 📝 **Auto-Generated Scripts**: Automatic generation of server scripts

## Installation

```bash
# Install required dependencies
pip install fastmcp

# For Groq integration (optional)
pip install groq
```

## Basic Usage

### Standard MCP Server

```python
from vscode_mcp import VSCodeMCP, ServerType

# Create a simple MCP server
server = VSCodeMCP(
    name="MyMCPServer",
    description="My awesome MCP server",
    server_type=ServerType.STDIO,
    command="python"
)

# Add some tools
server.add_tool(
    name="hello_world",
    description="Responds with a friendly greeting"
)

server.add_tool(
    name="fetch_data",
    description="Fetches data from an external API"
)

# Set up the VS Code workspace
mcp_json_path, server_script_path = server.setup_vscode_workspace()

print(f"MCP configuration saved to: {mcp_json_path}")
print(f"Server script saved to: {server_script_path}")
```

### Groq-Powered MCP Server

```python
from groq_vscode_mcp import GroqVSCodeMCP, GroqModel

# Create a Groq-powered MCP server
server = GroqVSCodeMCP(
    name="GroqTools",
    description="Tools powered by Groq LLMs",
    default_model=GroqModel.LLAMA_3_70B
)

# Add a chat tool that uses Groq for text generation
server.add_chat_tool(
    name="ask_groq",
    description="Ask Groq a question",
    system_prompt="You are a helpful, knowledgeable assistant.",
    model=GroqModel.LLAMA_3_70B
)

# Add a code generation tool that uses Groq
server.add_code_tool(
    name="code_with_groq",
    description="Generate code using Groq",
    system_prompt="You are an expert programmer.",
    model=GroqModel.DEEPSEEK_DISTILL_LLAMA_70B  # Using a specialized code model
)

# Set up the VS Code workspace
mcp_json_path, server_script_path = server.setup_vscode_workspace()
```

## Advanced Usage with FastMCP

The module is designed to work seamlessly with FastMCP for implementing custom tool functionality:

```python
from fastmcp import FastMCP
from vscode_mcp import VSCodeMCP, ServerType

# Create FastMCP instance for implementing tools
mcp = FastMCP("MyMCPServer")

@mcp.tool()
def hello_world(name="World"):
    """
    Responds with a friendly greeting
    """
    return {
        "content": f"Hello, {name}!"
    }

@mcp.tool()
def search_files(pattern, directory="."):
    """
    Search for files matching a pattern
    """
    import glob
    import os
    
    files = glob.glob(os.path.join(directory, pattern))
    return {
        "content": f"Found {len(files)} files: {', '.join(files)}"
    }

# Run the server
if __name__ == "__main__":
    mcp.run()
```

## Using the Command Line Interface

The module includes a simple CLI for creating and managing MCP servers:

```bash
# Create a new MCP server
python -m vscode_mcp create "My MCP Server" --tool hello_world "Responds with a greeting" --install

# Install an existing MCP server to VS Code
python -m vscode_mcp install path/to/server.py --user
```

## VS Code Integration

After setting up your MCP server:

1. Open VS Code and run the command `MCP: List Servers`
2. Select your server and click "Start Server"
3. Open the Chat View (Ctrl+Alt+I) and select "Agent mode"
4. Use the "Tools" button to select your MCP tools

## Available Groq Models

When using the `GroqVSCodeMCP` class, you can choose from these models:

```python
# Text models
GroqModel.LLAMA_3_8B       # Fast and efficient text generation
GroqModel.LLAMA_3_70B      # High-quality text generation
GroqModel.LLAMA_3_1_8B     # Updated 8B text model
GroqModel.LLAMA_3_1_70B    # Updated 70B text model
GroqModel.LLAMA_3_2_11B    # Latest 11B text model

# Vision models (for image analysis)
GroqModel.LLAMA_3_1_8B_VISION   # Efficient multimodal model
GroqModel.LLAMA_3_1_70B_VISION  # High-quality multimodal model
GroqModel.LLAMA_3_2_11B_VISION  # Latest multimodal model

# Code models
GroqModel.QWEN_CODER                 # Specialized code generation
GroqModel.DEEPSEEK_DISTILL_LLAMA_70B # Powerful code model
```

## Examples

Check out the sample files for more detailed examples:

- `sample_mcp_server.py`: A simple MCP server setup
- `advanced_mcp_server.py`: An advanced example with custom tool implementations
- `example_groq_mcp_server.py`: Example using Groq LLMs

## How It Works

The VSCodeMCP module:

1. Generates a `.vscode/mcp.json` file with the server configuration
2. Creates a Python script with FastMCP tool implementations
3. Sets up the necessary files for VS Code to discover and use your MCP server

The GroqVSCodeMCP extension:

1. Connects to the Groq API using your API key
2. Creates FastMCP tools that use Groq LLMs for their functionality
3. Manages the configuration and script generation automatically

## Requirements

- Python 3.7+
- fastmcp library
- Visual Studio Code with Copilot
- groq library (optional, for Groq integration)
- A Groq API key (for Groq integration)

## Project Structure

- `vscode_mcp.py` - Core module for VS Code MCP integration
- `groq_vscode_mcp.py` - Extension module that adds Groq LLM support
- `examples/` - Example MCP servers
  - `sample_mcp_server.py` - Basic MCP server example
  - `advanced_mcp_server.py` - Advanced example with custom tools
  - `example_groq_mcp_server.py` - Example using Groq LLMs

## Why Use Groq with MCP?

- **High Performance**: Groq's infrastructure delivers extremely fast inference speeds
- **Cost Efficiency**: Potentially lower per-token costs compared to other LLM providers
- **Model Variety**: Access to specialized models for different tasks
- **Local Control**: Direct integration without third-party dependencies
- **Customization**: Fine-tune each tool's behavior with custom system prompts

## License

MIT License

---

Made with ❤️ by [Your Name]
