import streamlit as st
import requests
import json
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="AssistantHub Chat",
    page_icon="🤖",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8003"
CHAT_ENDPOINT = f"{API_URL}/api/chat_response"
HEALTH_ENDPOINT = f"{API_URL}/health"


def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def send_message(query, language="en"):
    """Send message to the chat API"""
    try:
        payload = {
            "query": query,
            "language": language
        }
        response = requests.post(
            CHAT_ENDPOINT,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["bot_reply"]
        else:
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Connection error: {str(e)}"


def main():
    # Header
    st.title("🤖 AssistantHub Chat")
    st.markdown("---")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        language = st.selectbox(
            "Language",
            options=["en", "bn"],
            index=0,
            help="Select your preferred language"
        )

        # st.markdown("---")

        # # API Status
        # st.subheader("📊 API Status")
        # if check_api_health():
        #     st.success("✅ API is online")
        # else:
        #     st.error("❌ API is offline")
        #     st.warning(
        #         "Make sure the FastAPI server is running on localhost:8003")

        st.markdown("---")

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages with improved styling
    chat_container = st.container()

    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                # User message - right side
                col1, col2 = st.columns([1, 3])
                with col2:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #007acc;
                            color: white;
                            padding: 10px;
                            border-radius: 10px;
                            margin: 5px 0;
                            text-align: left;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">
                            <strong>You:</strong> {message["content"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                # Assistant message - left side
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f0f2f6;
                            color: #262730;
                            padding: 10px;
                            border-radius: 10px;
                            margin: 5px 0;
                            text-align: left;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            border-left: 4px solid #ff6b6b;
                        ">
                            <strong>🤖 Assistant:</strong> {message["content"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Add timestamp for assistant messages
                    if "timestamp" in message:
                        st.caption(f"⏰ {message['timestamp']}")

    # Chat input
    if prompt := st.chat_input("Ask me anything about your knowledge base..."):
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Get assistant response
        with st.spinner("Processing your request..."):
            response = send_message(prompt, language)

        # Add assistant message to chat history with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": timestamp
        })

        # Refresh to show the new messages
        st.rerun()

    # # Footer - right after chat input
    # st.markdown("---")
    # col1, col2, col3 = st.columns([1, 2, 1])
    # with col2:
    #     st.markdown(
    #         "<p style='text-align: center; color: #666;'>"
    #         "Powered by AssistantHub | Built with Streamlit & FastAPI"
    #         "</p>",
    #         unsafe_allow_html=True
    #     )


if __name__ == "__main__":
    main()
