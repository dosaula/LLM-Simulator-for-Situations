import streamlit as st
from src.interface import show_login_message, show_sidebar
from src.conversation_modes import chatbot, select_mode, conversation_simulation, simulate_roles, initialize_session_state, select_user
import os

# Main execution function
def main():
    """
    Main execution function with improved flow control.
    Mode selection only shows at appropriate times.
    """
    initialize_session_state()
    show_sidebar()

    if "username" not in st.session_state or not st.session_state["username"]:
        local_path = os.path.join("config", "local") 
        os.makedirs(local_path, exist_ok=True)
        show_login_message()
        return

    # Initialize simulation state if it doesn't exist
    if "simulation" not in st.session_state:
        st.session_state.simulation = {}

    # If we're already in a chatbot session, go directly to chatbot
    if "user_info" in st.session_state and st.session_state.get("selected_mode") == "Talk with an agent":
        chatbot()
        return
    
    # If we're in a simulation conversation, go directly to it
    if st.session_state.simulation.get("conversation_started", False):
        conversation_simulation()
        return

    # Only show mode selection if we're not in an active session
    selected_mode = select_mode()
    
    # Handle the interface based on selected mode
    if selected_mode == "Talk with an agent":
        if "user_info" not in st.session_state:
            select_user()
    else:  # simulate mode
        simulate_roles()

# Entry point for script execution
if __name__ == "__main__":
    main()
