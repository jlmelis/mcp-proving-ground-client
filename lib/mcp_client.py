from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import json
import yaml
from pathlib import Path
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv
import os
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

        Initializes the session, exit stack, and API clients based on available environment variables.

        Raises:
            APIConnectionError: If no API keys are available
        """
        # Load configuration
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()

        # Check for available API keys
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        self.active_api = None

        if deepseek_api_key:
            self.active_api = "deepseek"
            self.client = OpenAI(
                base_url="https://api.deepseek.com/v1",
                api_key=deepseek_api_key
            )
        elif anthropic_api_key:
            self.active_api = "anthropic"
            self.client = Anthropic(api_key=anthropic_api_key)
        else:
            raise APIConnectionError(
                "No API keys found. Please ensure either DEEPSEEK_API_KEY or ANTHROPIC_API_KEY environment variable is set.")

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
            raise InvalidServerScriptError(
                f"Server script must be a {self.config['server']['python_ext']} or {self.config['server']['js_ext']} file: {script_path}")

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
            logging.info("\nConnected to server with tools: %s", [tool.name for tool in tools])
            logging.info("Using %s API", self.active_api.capitalize())
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

        # Debug the tool_calls structure
        logging.debug("Tool calls type: %s", type(tool_calls))
        logging.debug("Tool calls structure: %s",
                      dir(tool_calls) if hasattr(tool_calls, '__dir__') else 'No dir available')

        # Handle different formats between Deepseek and Anthropic
        if not isinstance(tool_calls, list):
            try:
                # If it's not a list, check if it has an items attribute (like Anthropic might have)
                if hasattr(tool_calls, 'items') and callable(tool_calls.items):
                    tool_calls = [tool_calls]
                else:
                    logging.warning("Unexpected tool_calls format: %s", tool_calls)
                    return ["[Error: Unexpected tool_calls format]"]
            except Exception as e:
                logging.error("Error processing tool_calls: %s", str(e))
                return [f"[Error processing tool_calls: {str(e)}]"]

        for tool_call in tool_calls:
            try:
                # Extract tool name and arguments based on the API
                if self.active_api == "deepseek":
                    tool_name = tool_call.function.name
                    tool_args = (
                        json.loads(tool_call.function.arguments)
                        if isinstance(tool_call.function.arguments, str)
                        else tool_call.function.arguments
                    )
                elif self.active_api == "anthropic":
                    # Handle Anthropic tool call structure
                    if hasattr(tool_call, 'name'):
                        tool_name = tool_call.name
                    elif hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                        tool_name = tool_call.function.name
                    else:
                        logging.warning("Could not find tool name in: %s", tool_call)
                        continue

                    if hasattr(tool_call, 'input'):
                        tool_args = tool_call.input
                    elif hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                        tool_args = (
                            json.loads(tool_call.function.arguments)
                            if isinstance(tool_call.function.arguments, str)
                            else tool_call.function.arguments
                        )
                    else:
                        logging.warning("Could not find tool arguments in: %s", tool_call)
                        continue
                else:
                    logging.warning("Unknown API type: %s", self.active_api)
                    continue

                logging.debug("Executing tool: %s with args: %s", tool_name, tool_args)

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Format the result content
                result_content = ""
                if result.content and len(result.content) > 0:
                    if hasattr(result.content[0], 'text'):
                        result_content = result.content[0].text
                    else:
                        result_content = str(result.content[0])

                # Append messages for follow-up
                messages.append({
                    "role": "assistant",
                    "content": tool_name
                })
                messages.append({
                    "role": "user",
                    "content": json.dumps(result_content) if result_content else ""
                })

                # Get follow-up response from the model
                if self.active_api == "deepseek":
                    response = self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                    )
                    final_text.append(response.choices[0].message.content)
                elif self.active_api == "anthropic":
                    response = self.client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=4096,
                        messages=messages,
                    )
                    final_text.append(response.content[0].text)
            except Exception as e:
                logging.error("Error handling tool call: %s", str(e))
                import traceback
                logging.debug("Traceback: %s", traceback.format_exc())
                final_text.append(f"[Error handling tool call: {str(e)}]")

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

        try:
            if self.active_api == "deepseek":
                stop_reason: str = (
                    "tool_calls"
                    if first_response.choices[0].message.tool_calls is not None
                    else first_response.choices[0].finish_reason
                )

                if stop_reason == "stop":
                    final_text.append(first_response.choices[0].message.content)
                elif stop_reason == "tool_calls":
                    final_text.extend(
                        await self._handle_tool_calls(first_response.choices[0].message.tool_calls, messages))
                else:
                    raise ValueError(f"Unknown stop reason: {stop_reason}")
            elif self.active_api == "anthropic":
                # Debug anthropic response structure
                logging.debug("Response type: %s", type(first_response))
                logging.debug("Response attributes: %s", dir(first_response))

                # Check for tool_calls in the Anthropic response
                if hasattr(first_response, 'content') and first_response.content:
                    # First check if this is a response with content
                    final_text.append(first_response.content[0].text)

                # Look for tool_calls - the attribute name may vary
                tool_calls_attribute = None
                for attr in ['tool_calls', 'tool_use', 'tools']:
                    if hasattr(first_response, attr):
                        tool_calls_attribute = attr
                        break

                if tool_calls_attribute and getattr(first_response, tool_calls_attribute):
                    logging.debug("Found tool calls in attribute: %s", tool_calls_attribute)
                    tool_calls = getattr(first_response, tool_calls_attribute)
                    final_text.extend(await self._handle_tool_calls(tool_calls, messages))
                elif not final_text:  # If we haven't added any text yet
                    logging.debug("Response structure: %s", first_response)
                    final_text.append("Received response from Anthropic API but unable to extract content.")
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            final_text.append(f"Error processing response: {str(e)}")

        return "\n".join(final_text)

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools.

        Args:
            query (str): The user query to process

        Returns:
            str: The response from processing the query
        """
        try:
            messages = await self._prepare_messages(query)

            response = await self.session.list_tools()

            mcp_tools: List[Dict[str, Any]] = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in response.tools]

            # Simple first approach with Anthropic: use without tools first
            if self.active_api == "deepseek":
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
            elif self.active_api == "anthropic":
                logging.debug("Sending request to Anthropic API with tools...")
                # Convert MCP tools to Anthropic tools format
                anthropic_tools = []

                for tool in mcp_tools:
                    # Format tools for Anthropic API
                    try:
                        anthropic_tool = {
                            "name": tool["name"],
                            "description": tool["description"],
                            "input_schema": tool["input_schema"]
                        }
                        anthropic_tools.append(anthropic_tool)
                    except Exception as e:
                        logging.debug("Error formatting tool %s: %s", tool['name'], str(e))

                try:
                    # First attempt with tools
                    first_response = self.client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=4096,
                        messages=messages,
                        tools=anthropic_tools
                    )
                    logging.debug("Received response from Anthropic API with tools")
                except Exception as e:
                    logging.warning("Error with tools, falling back to basic mode: %s", str(e))
                    # Fallback to no tools if the tools call fails
                    first_response = self.client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=4096,
                        messages=messages
                    )
                    logging.debug("Received response from Anthropic API in fallback mode")

            return await self._process_response(first_response, messages)
        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            return f"Error processing query: {str(e)}"

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

        logging.info("\nMCP Client Started!")
        logging.info("Using %s API", self.active_api.capitalize())
        print("\nMCP Client Started!")
        print(f"Using {self.active_api.capitalize()} API")
        print("Type your queries or 'quit' to exit. Type 'debug' for debug info.")

        while True:
            try:
                query: str = input("\n\033[1;32mYou:\033[0m ").strip()

                if query.lower() == 'quit':
                    break

                if query.lower() == 'debug':
                    print(f"Active API: {self.active_api}")
                    print(f"Client type: {type(self.client)}")
                    print(f"Session initialized: {self.session is not None}")
                    print("For more detailed logs, set logging level to DEBUG or type 'debug on'|'debug off'")
                    continue

                if query.lower() == 'debug on':
                    logging.getLogger().setLevel(logging.DEBUG)
                    print("Debug logging enabled")
                    continue

                if query.lower() == 'debug off':
                    logging.getLogger().setLevel(logging.INFO)
                    print("Debug logging disabled")
                    continue

                stop_event = Event()
                spinner_thread = Thread(target=spinner, args=(stop_event,))
                spinner_thread.start()

                # Set a timeout for the query
                import asyncio
                try:
                    response = await asyncio.wait_for(self.process_query(query), timeout=60.0)
                except asyncio.TimeoutError:
                    response = "Request timed out after 60 seconds."

                stop_event.set()
                spinner_thread.join()

                print(f'\033[1;34mMCP:\033[0m {response}')

            except Exception as e:
                logging.error("Error in chat_loop: %s", str(e))
                import traceback
                logging.debug("Traceback: %s", traceback.format_exc())
                print(f"\nError in chat_loop: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up resources.

        Closes the async exit stack and releases all resources.
        """
        await self.exit_stack.aclose()