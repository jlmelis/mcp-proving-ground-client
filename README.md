# MCP Client

## Overview

The MCP Client is a Python-based client for interacting with the MCP API. This project is primarily being developed to provide a simple MCP Client that can be used to test custom MCP servers, helping to avoid message limits encountered with other clients like Claude Desktop.

## Installation

1. Install [uv](https://github.com/astral-sh/uv)
2. Clone this repository
3. Run `uv venv` to create a virtual environment
4. Activate the virtual environment
5. Run `uv pip install -r requirements.txt`
6. Create a `.env` file and add your Deepseek API key:
   ```bash
   DEEPSEEK_API_KEY=your_api_key_here
   ```

## Usage

Run the client with:

```bash
uv run main.py
```

## Configuration

Edit `config.yaml` to configure the MCP Server.

## Sample Server

A sample server `weather.py` is included in this repository, based on the quickstart example from the [Model Context Protocol documentation](https://modelcontextprotocol.io/quickstart/server). This server demonstrates basic MCP server functionality and can be used for testing and experimentation.

## Development Status

This project is currently in early active development. There may be bugs, missing features, and breaking changes. Please report any issues you encounter.

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request