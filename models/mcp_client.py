import asyncio
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import json
import yaml
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env

class InvalidServerScriptError(Exception):
    """Raised when the server script path is invalid."""
    pass

class APIConnectionError(Exception):
    """Raised when there is an issue connecting to the API."""
    pass

class MCPClient:
    def __init__(self) -> None:
        """Initialize the MCPClient instance.

        Initializes the session, exit stack, and OpenAI client with Deepseek API configuration.

        Raises:
            APIConnectionError: If the DEEPSEEK_API_KEY environment variable is not set
        """
        # Load configuration
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise APIConnectionError("API key not found. Please ensure the DEEPSEEK_API_KEY environment variable is set.")
        self.client: OpenAI = OpenAI(
                    base_url="https://api.deepseek.com/v1",
                    api_key=api_key
                )

    async def connect_to_server(self, script_path: str = None) -> None:
        """Connect to an MCP server.

        Args:
            script_path (str): Path to the server script (.py or .js)

        Raises:
            InvalidServerScriptError: If the server script path is invalid
            APIConnectionError: If there is an issue connecting to the API
        """
        # Use provided script path or config value
        script_path = script_path or self.config['server']['args'][-1]
        if not script_path:
            raise ValueError('Server script path must be provided either via command line or config file')

        # Determine script type
        is_python = script_path.endswith(self.config['server']['python_ext'])
        is_js = script_path.endswith(self.config['server']['js_ext'])
        if not (is_python or is_js):
            raise InvalidServerScriptError(f"Server script must be a {self.config['server']['python_ext']} or {self.config['server']['js_ext']} file: {script_path}")

        # Update script path in args
        args = self.config['server']['args'].copy()
        args[-1] = script_path

        server_params = StdioServerParameters(
            command=self.config['server']['command'],
            args=args,
            env=self.config['server']['env']
        )
        
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            await self.session.initialize()
            
            # List available tools
            response = await self.session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])
        except Exception as e:
            raise APIConnectionError(f"Failed to connect to server: {str(e)}")

    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]], messages: List[Dict[str, str]]) -> List[str]:
        """Handle tool calls and update messages.

        Args:
            tool_calls (List[Dict[str, Any]]): List of tool calls to handle
            messages (List[Dict[str, str]]): List of messages to update

        Returns:
            List[str]: List of response strings from handling tool calls
        """
        tool_results: List[Dict[str, Any]] = []
        final_text: List[str] = []

        for tool_call in tool_calls:
            
            tool_name: str = tool_call.function.name
            tool_args: Dict[str, Any] = (
                json.loads(tool_call.function.arguments)
                if isinstance(tool_call.function.arguments, str)
                else tool_call.function.arguments
            )

            try:
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                
                messages.append({
                    "role": "assistant",
                    "content": tool_call.function.name
                })
                messages.append({
                    "role": "user", 
                    "content": json.dumps(result.content[0].text) if result.content and len(result.content) > 0 and hasattr(result.content[0], 'text') else ""
                })

                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                )

                final_text.append(response.choices[0].message.content)
            except Exception as e:
                print(f"Error executing tool {tool_name}: {str(e)}")
                final_text.append(f"[Error executing tool {tool_name}: {str(e)}]")

        return final_text

    async def _prepare_messages(self, query: str) -> List[Dict[str, str]]:
        """Prepare initial messages for the query.

        Args:
            query (str): The user query

        Returns:
            List[Dict[str, str]]: List of initial messages
        """
        return [
            {
                "role": "user",
                "content": query
            }
        ]

    async def _process_response(self, first_response: Any, messages: List[Dict[str, str]]) -> str:
        """Process the first response and handle tool calls if necessary.

        Args:
            first_response (Any): The initial response from the API
            messages (List[Dict[str, str]]): List of messages

        Returns:
            str: The final response text

        Raises:
            ValueError: If the stop reason is unknown
        """
        final_text: List[str] = []

        stop_reason: str = (
            "tool_calls"
            if first_response.choices[0].message.tool_calls is not None
            else first_response.choices[0].finish_reason
        )

        if stop_reason == "stop":
            final_text.append(first_response.choices[0].message.content)
        elif stop_reason == "tool_calls":
            final_text.extend(await self._handle_tool_calls(first_response.choices[0].message.tool_calls, messages))
        else:
            raise ValueError(f"Unknown stop reason: {stop_reason}")

        return "\n".join(final_text)

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools.

        Args:
            query (str): The user query to process

        Returns:
            str: The response from processing the query
        """
        messages = await self._prepare_messages(query)

        response = await self.session.list_tools()
        
        mcp_tools: List[Dict[str, Any]] = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        available_tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": tool['input_schema']
                }
            }
            for tool in mcp_tools
        ]

        first_response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=available_tools
        )

        return await self._process_response(first_response, messages)

    async def chat_loop(self) -> None:
        """Run an interactive chat loop.

        Continuously prompts for user input and processes queries until 'quit' is entered.
        """
        import sys
        import time
        from itertools import cycle
        from threading import Thread, Event

        def spinner(stop_event):
            for frame in cycle(['|', '/', '-', '\\']):
                if stop_event.is_set():
                    break
                sys.stdout.write(f'\rWaiting for response {frame}')
                sys.stdout.flush()
                time.sleep(0.1)
            sys.stdout.write('\r' + ' ' * 20 + '\r')

        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query: str = input("\n\033[1;32mYou:\033[0m ").strip()

                if query.lower() == 'quit':
                    break

                stop_event = Event()
                spinner_thread = Thread(target=spinner, args=(stop_event,))
                spinner_thread.start()

                response: str = await self.process_query(query)
                stop_event.set()
                spinner_thread.join()

                print(f'\033[1;34mMCP:\033[0m {response}')

            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self) -> None:
        """Clean up resources.

        Closes the async exit stack and releases all resources.
        """
        await self.exit_stack.aclose()
