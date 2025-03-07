import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

 # Load environment variables
load_dotenv()

# Provider configuration
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

def initialize_session():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "provider" not in st.session_state:
        st.session_state.provider = "Deepseek"
    if "model" not in st.session_state:
        st.session_state.model = PROVIDER_CONFIG["Deepseek"]["models"][0]
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

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

def render_chat():
    """Main chat interface"""
    st.title("💬 Multi-LLM Chat")
    st.caption("🚀 Switch providers in the sidebar")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                # Show initial thinking indicator
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("▌")  # Blinking cursor
                
                client = OpenAI(
                    base_url=PROVIDER_CONFIG[st.session_state.provider]["base_url"],
                    api_key=st.session_state.api_key
                )
                
                stream = client.chat.completions.create(
                    model=st.session_state.model,
                    messages=st.session_state.messages,
                    stream=True
                )
                
                full_response = []
                response_container = st.empty()
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        # Clear thinking indicator on first content
                        if not full_response:
                            thinking_placeholder.empty()
                        full_response.append(chunk.choices[0].delta.content)
                        # Show text with blinking cursor during stream
                        response_container.markdown("".join(full_response) + "▌")
                
                # Finalize without cursor
                response_container.markdown("".join(full_response))
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "".join(full_response)
                })
                
            except Exception as e:
                thinking_placeholder.empty()
                st.error(f"❌ Error: {str(e)}")
                st.session_state.messages.pop()

def main():
    st.set_page_config(
        page_title="Multi-LLM Chat",
        page_icon="🤖",
        layout="centered"
    )
    
    initialize_session()
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()
