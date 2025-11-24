"""
GPT-like Chat Interface with Streaming and Cost Telemetry
"""

import streamlit as st
import requests
import uuid
import json
import os

st.set_page_config(
    page_title="Chat",
    page_icon="💬",
    layout="wide",
)

st.title("Chat")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# API configuration - use Docker service name when running in container
default_api_url = os.getenv("API_BASE_URL", "http://app:8000")
API_BASE_URL = st.sidebar.text_input("API Base URL", value=default_api_url, help="Base URL for the chat API")

# New chat button
if st.sidebar.button("New Chat"):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

st.sidebar.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "stats" in message and message["stats"]:
            stats = message["stats"]
            st.caption(
                f"[stats] prompt={stats.get('prompt_tokens', 0)} "
                f"completion={stats.get('completion_tokens', 0)} "
                f"cost=${stats.get('cost', 0):.6f} "
                f"latency={stats.get('latency_ms', 0):.0f}ms"
            )

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        stats_placeholder = st.empty()
        full_response = ""
        stats_data = None

        try:
            # Stream response from API
            response = requests.post(
                f"{API_BASE_URL}/api/chat/stream",
                json={"message": prompt, "session_id": st.session_state.session_id},
                stream=True,
                timeout=60,
            )
            response.raise_for_status()

            current_event = None
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")

                    if line_str.startswith("event: "):
                        current_event = line_str[7:]
                    elif line_str.startswith("data: "):
                        data = line_str[6:]

                        if current_event == "stats":
                            # Parse stats JSON
                            try:
                                stats_data = json.loads(data)
                            except json.JSONDecodeError:
                                pass
                            current_event = None
                        else:
                            # Regular content - accumulate and display
                            full_response += data
                            # Display with cursor during streaming
                            message_placeholder.markdown(full_response + "▌")

            # Once streaming is complete, render as markdown
            message_placeholder.markdown(full_response)

            # Display stats
            if stats_data:
                stats_placeholder.caption(
                    f"[stats] prompt={stats_data.get('prompt_tokens', 0)} "
                    f"completion={stats_data.get('completion_tokens', 0)} "
                    f"cost=${stats_data.get('cost', 0):.6f} "
                    f"latency={stats_data.get('latency_ms', 0):.0f}ms"
                )

        except requests.exceptions.ConnectionError:
            full_response = "Error: Could not connect to the API. Make sure the server is running."
            message_placeholder.error(full_response)
        except requests.exceptions.Timeout:
            full_response = "Error: Request timed out."
            message_placeholder.error(full_response)
        except Exception as e:
            full_response = f"Error: {str(e)}"
            message_placeholder.error(full_response)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response, "stats": stats_data})
