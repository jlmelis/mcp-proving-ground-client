import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
                    base_url="https://api.deepseek.com/v1",
                    api_key=os.getenv("DEEPSEEK_API_KEY")
                )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file" + server_script_path)
            
        command = "uv" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[
                "run",
                server_script_path
            ],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        
        mcp_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        available_tools = [
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

        
        # Process response and handle tool calls
        tool_results = []
        final_text = []

        stop_reason = (
            "tool_calls"
            if first_response.choices[0].message.tool_calls is not None
            else first_response.choices[0].finish_reason
        )
        
        if stop_reason == "stop":
            final_text.append(first_response.choices[0].message.content)
        elif stop_reason == "tool_calls":
            #handle tool calls
            for tool_call in first_response.choices[0].message.tool_calls:
                print(f"Tool call detected: {tool_call.function.name}")
                tool_name = tool_call.function.name
                tool_args = arguments = (
                    json.loads(tool_call.function.arguments)
                    if isinstance(tool_call.function.arguments, str)
                    else tool_call.function.arguments
                )

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                
                
                messages.append({
                    "role": "assistant",
                    "content": first_response.choices[0].message.content
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
        else:
            raise ValueError(f"Unknown stop reason: {stop_reason}")


        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())