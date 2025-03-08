import streamlit as st
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

# Load environment variables
load_dotenv()

PROVIDER_CONFIG = {
    "Deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-coder"],
        "key_env": "DEEPSEEK_API_KEY",
        "default_key": os.getenv("DEEPSEEK_API_KEY", "")
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4-turbo", "gpt-3.5-turbo"],
        "key_env": "OPENAI_API_KEY",
        "default_key": os.getenv("OPENAI_API_KEY", "")
    }
}

class MCPClient:
    def __init__(self, server_script_path: str):
        self.session = None
        self.server_script_path = server_script_path
        self.available_tools = []
        
    async def connect(self):
        """Connect to MCP server"""
        is_python = self.server_script_path.endswith('.py')
        command = "python" if is_python else "node"
        
        server_params = StdioServerParameters(
            command=command,
            args=[self.server_script_path],
            env=None
        )
        
        stdio_transport = await stdio_client(server_params)
        self.session = ClientSession(stdio_transport[0], stdio_transport[1])
        await self.session.initialize()
        
        # Get available tools
        response = await self.session.list_tools()
        self.available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

    async def process_query(self, messages, provider_config, api_key, model):
        """Process query with selected provider and MCP tools"""
        try:
            client = OpenAI(
                base_url=provider_config["base_url"],
                api_key=api_key
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=self.available_tools,
                tool_choice="auto",
                stream=True
            )
            
            full_response = []
            tool_responses = []
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response.append(chunk.choices[0].delta.content)
                    yield "".join(full_response)
                
                if chunk.choices[0].delta.tool_calls:
                    tool_calls = chunk.choices[0].delta.tool_calls
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        # Execute tool call
                        result = await self.session.call_tool(tool_name, tool_args)
                        tool_responses.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": result.content
                        })
            
            if tool_responses:
                yield "\n[Processing tool responses...]"
                messages.extend(tool_responses)
                async for chunk in self.process_query(messages, provider_config, api_key, model):
                    yield chunk
                    
        except Exception as e:
            yield f"Error: {str(e)}"

def render_sidebar():
    """Render provider/model selection sidebar"""
    with st.sidebar:
        st.title("⚙️ LLM Settings")
        
        # Provider selection
        new_provider = st.selectbox(
            "Provider",
            options=list(PROVIDER_CONFIG.keys()),
            index=list(PROVIDER_CONFIG.keys()).index(st.session_state.provider)
        )
        
        # Reset model if provider changes
        if new_provider != st.session_state.provider:
            st.session_state.provider = new_provider
            st.session_state.model = PROVIDER_CONFIG[new_provider]["models"][0]
        
        # Model selection
        st.session_state.model = st.selectbox(
            "Model",
            options=PROVIDER_CONFIG[st.session_state.provider]["models"],
            index=PROVIDER_CONFIG[st.session_state.provider]["models"].index(st.session_state.model)
        )
        
        # API Key input
        st.session_state.api_key = st.text_input(
            "API Key",
            type="password",
            value=PROVIDER_CONFIG[st.session_state.provider]["default_key"],
            help=f"Find your API key at {PROVIDER_CONFIG[st.session_state.provider]['base_url']}"
        )

def main():
    st.set_page_config(
        page_title="Multi-LLM MCP Chat",
        page_icon="🤖",
        layout="centered"
    )
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "provider" not in st.session_state:
        st.session_state.provider = "Deepseek"
    if "model" not in st.session_state:
        st.session_state.model = PROVIDER_CONFIG["Deepseek"]["models"][0]
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "mcp_client" not in st.session_state:
        server_script = os.getenv("MCP_SERVER_SCRIPT", "./server.py")
        st.session_state.mcp_client = MCPClient(server_script)
        asyncio.run(st.session_state.mcp_client.connect())
    
    render_sidebar()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            async def stream_response():
                nonlocal full_response
                provider_config = PROVIDER_CONFIG[st.session_state.provider]
                async for chunk in st.session_state.mcp_client.process_query(
                    st.session_state.messages,
                    provider_config,
                    st.session_state.api_key,
                    st.session_state.model
                ):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                
                response_container.markdown(full_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
            
            asyncio.run(stream_response())

if __name__ == "__main__":
    main()
