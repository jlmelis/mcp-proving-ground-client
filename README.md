# MCP Client

## Overview

The MCP Client is a Python-based client for interacting with the MCP API. This project is primarily being developed to provide a simple MCP Client that can be used to test custom MCP servers, helping to avoid message limits encountered with other clients like Claude Desktop.

## Installation

1. Install [uv](https://github.com/astral-sh/uv)
2. Clone this repository
3. Run `uv venv` to create a virtual environment
4. Activate the virtual environment
5. Run `uv pip install -r requirements.txt`

## Usage

Run the client with:

```bash
uv run main.py
```

## Configuration

Edit `config.yaml` to configure the MCP Server.

## Development Status

This project is currently in early active development. There may be bugs, missing features, and breaking changes. Please report any issues you encounter.

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request