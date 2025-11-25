"""
Self-Healing Code Assistant Interface

This page provides a UI for generating and self-healing code based on natural language descriptions.
Supports Python and Rust with automatic language detection, test execution, and iterative error correction.
"""

import os

import requests
import streamlit as st

st.set_page_config(
    page_title="Code Healing",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Self-Healing Code Assistant")
st.markdown("Generate working code with tests from natural language descriptions. **Language is auto-detected!**")

# API configuration
default_api_url = os.getenv("API_BASE_URL", "http://app:8000")
API_BASE_URL = st.sidebar.text_input("API Base URL", value=default_api_url, help="Base URL for the code healing API")

# Sidebar configuration
st.sidebar.markdown("### ℹ️ Auto-Detection")
st.sidebar.info(
    "Programming language is automatically detected from your task description. "
    "Mention 'Python' or 'Rust' in your description for best results."
)

# Example tasks
st.sidebar.markdown("### Example Tasks")
example_tasks = {
    "Python - Quicksort": "write a quicksort function in Python with comprehensive tests",
    "Python - Binary Search": "implement binary search in Python with edge case tests",
    "Python - Fibonacci": "write a fibonacci function in Python with memoization and tests",
    "Rust - Quicksort": "write a quicksort function in Rust with comprehensive tests",
    "Rust - Binary Search": "implement binary search in Rust with edge case tests",
    "Rust - Palindrome Check": "write a function to check if a string is a palindrome in Rust with tests",
}

selected_example = st.sidebar.selectbox("Load Example", [""] + list(example_tasks.keys()))

# Main interface
st.markdown("---")

# Task description input
task_description = st.text_area(
    "Task Description",
    value=example_tasks.get(selected_example, ""),
    height=100,
    placeholder="Describe the code you want to generate, e.g., 'write quicksort in Python'",
    help="Provide a natural language description of what code you want generated",
)

# Generate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button("🚀 Generate & Heal Code", use_container_width=True, type="primary")

st.markdown("---")

# Processing and results
if generate_button:
    if not task_description.strip():
        st.error("Please enter a task description")
    else:
        # Create containers for live updates
        status_container = st.container()
        code_container = st.container()
        results_container = st.container()

        with status_container:
            status_placeholder = st.empty()
            status_placeholder.info(f"🔄 Starting code generation for: {task_description}")

        try:
            # Call the API (language is auto-detected)
            with st.spinner("Detecting language and generating code..."):
                response = requests.post(
                    f"{API_BASE_URL}/api/heal-code",
                    json={"task_description": task_description},
                    timeout=300,  # 5 minutes timeout for complex code generation
                )
                response.raise_for_status()
                result = response.json()

            # Display results based on success
            if result["success"]:
                status_placeholder.success(f"✅ {result['message']} (Attempts: {result['attempts']})")

                # Display final code
                with code_container:
                    st.subheader("Generated Code")
                    # Try to detect language from code for syntax highlighting
                    detected_lang = (
                        "python"
                        if ".py" in result.get("working_directory", "") or "def " in result["final_code"]
                        else "rust"
                    )
                    st.code(result["final_code"], language=detected_lang)

                    # Download button
                    file_ext = "py" if detected_lang == "python" else "rs"
                    st.download_button(
                        label="📥 Download Code",
                        data=result["final_code"],
                        file_name=f"generated_code.{file_ext}",
                        mime="text/plain",
                    )

                # Display test output
                with results_container:
                    st.subheader("Test Results")
                    with st.expander("View Test Output", expanded=False):
                        st.text(result["test_output"])

                    st.info(f"📁 Working directory: `{result['working_directory']}`")

            else:
                status_placeholder.error(f"❌ {result['message']}")

                # Display final code attempt
                with code_container:
                    st.subheader("Last Code Attempt")
                    if result["final_code"]:
                        # Try to detect language from code for syntax highlighting
                        detected_lang = (
                            "python"
                            if ".py" in result.get("working_directory", "") or "def " in result["final_code"]
                            else "rust"
                        )
                        st.code(result["final_code"], language=detected_lang)
                    else:
                        st.warning("No code was generated")

                # Display error details
                with results_container:
                    st.subheader("Error Details")
                    st.error(f"Failed after {result['attempts']} attempts")

                    with st.expander("View Error Output", expanded=True):
                        st.text(result["test_output"])

                    st.info(f"📁 Working directory: `{result['working_directory']}`")

        except requests.exceptions.ConnectionError:
            status_placeholder.error("❌ Could not connect to the API. Make sure the server is running.")
        except requests.exceptions.Timeout:
            status_placeholder.error(
                "❌ Request timed out. Code generation may take several minutes for complex tasks."
            )
        except requests.exceptions.HTTPError as e:
            status_placeholder.error(f"❌ API Error: {e.response.status_code}")
            try:
                error_detail = e.response.json()
                st.error(f"Details: {error_detail.get('detail', 'Unknown error')}")
            except Exception:
                st.error(f"Details: {str(e)}")
        except Exception as e:
            status_placeholder.error(f"❌ Unexpected error: {str(e)}")

# Information section
st.markdown("---")
st.markdown("### How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
    **1. Generate**
    - Describes your task in natural language
    - LLM generates code with tests
    - Code is written to disk
    """
    )

with col2:
    st.markdown(
        """
    **2. Test**
    - Runs `pytest` (Python) or `cargo test` (Rust)
    - Captures compilation/test errors
    - Analyzes failure reasons
    """
    )

with col3:
    st.markdown(
        """
    **3. Heal**
    - Iteratively fixes errors
    - Max 3 attempts
    - Returns working code or detailed errors
    """
    )

st.markdown("---")
st.markdown("### Supported Languages")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
    **Python** 🐍
    - Uses `pytest` for testing
    - Generates separate test files
    - Supports standard library only
    - Examples: algorithms, data structures, utilities
    """
    )

with col2:
    st.markdown(
        """
    **Rust** 🦀
    - Uses `cargo test` for testing
    - Includes inline tests with `#[cfg(test)]`
    - Standard library only
    - Examples: algorithms, data structures, parsers
    """
    )

st.markdown("---")
st.markdown(
    """
### Tips for Best Results

- **Be specific**: "write quicksort with partition logic" is better than "write sorting"
- **Mention tests**: "with comprehensive tests" or "including edge case tests"
- **Keep it simple**: Stick to standard library features
- **One task at a time**: Focus on a single function or algorithm

### Known Limitations

- Maximum 3 self-healing attempts
- Standard library only (no external dependencies)
- Generated code is stored in temporary directories
- May require manual review for complex logic
"""
)
