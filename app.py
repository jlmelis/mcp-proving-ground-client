import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration for different LLM providers
PROVIDER_CONFIG = {
    "Deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-coder"],
        "key_env": "DEEPSEEK_API_KEY"
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4-turbo", "gpt-3.5-turbo"],
        "key_env": "OPENAI_API_KEY"
    }
}

def initialize_chat():
    """Initialize chat session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_api_key" not in st.session_state:
        st.session_state.current_api_key = ""

def render_sidebar():
    """Render provider/model selection sidebar"""
    with st.sidebar:
        st.title("LLM Settings")
        provider = st.selectbox(
            "Provider",
            options=list(PROVIDER_CONFIG.keys()),
            key="provider"
        )
        model = st.selectbox(
            "Model",
            options=PROVIDER_CONFIG[provider]["models"],
            key="model"
        )
        api_key = st.text_input(
            "API Key",
            type="password",
            value=os.getenv(PROVIDER_CONFIG[provider]["key_env"]) or "",
            help="API key for the selected provider"
        )
        st.session_state.current_api_key = api_key

def render_chat_interface():
    """Main chat interface rendering"""
    st.title("Multi-LLM Chat")

    # Display chat messages
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
                client = OpenAI(
                    base_url=PROVIDER_CONFIG[st.session_state.provider]["base_url"],
                    api_key=st.session_state.current_api_key
                )

                stream = client.chat.completions.create(
                    model=st.session_state.model,
                    messages=st.session_state.messages,
                    stream=True,
                    
                )

                response = st.write_stream(
                    chunk.choices[0].delta.content for chunk in stream
                    if chunk.choices[0].delta.content
                )

                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.pop()  # Remove failed message

def main():
    initialize_chat()
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()