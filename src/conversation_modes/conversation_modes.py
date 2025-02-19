import streamlit as st
import time
import os
import uuid
from src.agents import create_agents_crewai, improve_role_description
from src.message_handling import text_to_speech, transcribe_audio, upload_and_process_documents, process_message, \
    save_feedback, load_history, save_history, clear_history, get_existing_roles, get_existing_comb, \
    load_goal_from_file, \
    summarize_text_with_crewai


def initialize_session_state():
    """
    Initializes all necessary session state variables at the start of the application.
    This function ensures that important state variables exist to prevent errors during execution.
    """
    if "username" not in st.session_state or st.session_state["username"] is None:
        st.session_state["username"] = None  # Ensure a valid default value

    if "user_selected" not in st.session_state:
        st.session_state.user_selected = False  # Track if the user has made a selection

    if "selection_type" not in st.session_state:
        st.session_state.selection_type = None  # Stores the selection type

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}  # Stores chat messages


def reset_chat():
    """
    Cleans up the chat history and related states in the session.
    This ensures a fresh start for new conversations.
    """
    # Remove message history
    if "messages" in st.session_state:
        del st.session_state.messages
    # Remove any stored user input
    if "user_input" in st.session_state:
        del st.session_state.user_input
    # Remove processing flag
    if "processing" in st.session_state:
        del st.session_state.processing


def generate_greeting(client_type_text: str, role: str) -> str:
    """
    Generates a contextual greeting based on client type and role.

    Args:
        client_type_text (str): Type of client ('client', 'no_client', or other)
        role (str): Role of the agent generating the greeting

    Returns:
        str: Appropriate greeting message
    """
    greetings = {
        "Already client": f"¡Hola! ¿Qué tal va todo? ¿La familia bien? ¿Qué ofertas me traes?",
        "New client": f"Hola buenas, soy {role} ¿Qué necesita?",
        "default": f"Hola, soy {role}. ¿Sobre qué tema te gustaría conversar?"
    }
    return greetings.get(client_type_text, greetings["default"])


def select_mode():
    """
    Mode selection function that returns the selected mode immediately.
    """
    mode = st.radio(
        "What would you like to do with the chatbot?",
        ["Talk with an agent", "Simulate a conversation between agents"],
        key="mode_selection"
    )
    
    # Clear irrelevant state when mode changes
    if mode != st.session_state.get("selected_mode"):
        st.session_state.selected_mode = mode
        if mode == "Talk with an agent":
            st.session_state.simulation = {}
            if "conversation_started" in st.session_state.simulation:
                del st.session_state.simulation["conversation_started"]
        elif mode == "Simulate a conversation between agents":
            if "user_info" in st.session_state:
                del st.session_state.user_info
    
    return mode

def ensure_directory_exists(path):
    """
    Ensures that a directory exists, creates it if it doesn't.

    Args:
        path (str): Path to the directory to check/create

    Returns:
        bool: True if directory already existed, False if it was created
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return False
    return True


def save_and_continue(role, objective, tasks, user_type, uploaded_file):
    """
    Saves role description, configures session state, and prepares for chat interaction.

    Args:
        role (str): Role name
        objective (str): Role objective
        tasks (str): Role tasks
        user_type (str): Type of user
        uploaded_file: File object containing additional context

    Returns:
        bool: Success status of the operation
    """
    try:
        # Create safe role name and directory
        safe_role_name = "".join(c for c in role if c.isalnum() or c in (' ', '-', '_')).strip()
        role_dir = os.path.join("config", "base", "personalidad", safe_role_name)
        ensure_directory_exists(role_dir)
        role_file_path = os.path.join(role_dir, "goal.txt")

        # Handle context file if provided
        context_file_path = None
        if uploaded_file is not None:
            excel_path = os.path.join(role_dir, "context.xlsx")
            with open(excel_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            context_file_path = excel_path
            st.success(f"Excel file saved in {excel_path}")

        # Save role information
        save_goal_to_file(role_file_path, objective, tasks)

        # Configure session state
        combined_goal = f"{objective}\n{tasks}"
        st.session_state.user_info = {
            "role": role,
            "goal": combined_goal,
            "objective": objective,
            "chroma": f"./vectordb1_{user_type.lower()}",
            "tasks": tasks,
            "context_file": context_file_path
        }

        # Create agents
        if context_file_path:
            create_agents_crewai(role, combined_goal, f"./vectordb1_{user_type.lower()}",
                                 context_file=context_file_path)
        else:
            create_agents_crewai(role, combined_goal, f"./vectordb1_{user_type.lower()}")

        st.session_state.config_completed = True

        return True  # Indica que el guardado fue exitoso

    except Exception as e:
        st.error(f"Error saving the role: {str(e)}")
        return False


def save_goal_to_file(file_path, objective, tasks):
    """
    Saves objective and tasks to a goal file.

    Args:
        file_path (str): Path where to save the goal file
        objective (str): The objective text
        tasks (str): The tasks text
    """
    # Ensure directory exists
    directory = os.path.dirname(file_path)
    ensure_directory_exists(directory)

    # Save content with separator
    content = f"{objective}\n---\n{tasks}"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def select_user():
    """
    Main function for user role selection and configuration in the Streamlit interface.
    Handles role selection, configuration, and agent creation process.

    This function creates the main user interface for:
    - Selecting predefined roles or creating custom ones
    - Configuring COM-B parameters
    - Handling file uploads
    - Managing interaction settings
    - Creating and configuring agents
    """

    st.title("Select the agent's role")

    # Configure base path for roles
    base_path = os.path.join("config", "base", "personalidad")
    comb_path = os.path.join(base_path, "comb")
    client_type_path = os.path.join(base_path, "client_type")  # For customer types
    ensure_directory_exists(base_path)
    ensure_directory_exists(comb_path)
    ensure_directory_exists(client_type_path)

    # Get existing role names
    role_names = get_existing_roles(base_path)
    all_options = role_names + ["Customize a new role"]

    # Role selector
    user_type = st.selectbox(
        "Choose from the following options or personalize your agent",
        all_options,
        key="user_role_selectbox"
    )

    # Get available customer types
    client_type_files = [f.replace(".txt", "") for f in os.listdir(client_type_path) if f.endswith(".txt")]
    client_type_files = ["Select a client type"] + client_type_files  # Add default option

    # Selector for client type
    client_type = st.selectbox(
        "Select a client type:",
        client_type_files,
        key="client_type_selectbox"
    )

    # Variables to store role information
    max_interactions = ""
    combined_goal = ""
    objective = ""
    tasks = ""
    goal_files = []

    # Get COM-B options
    comb_options = get_existing_comb()
    capacidad_options = ["undefined"] + comb_options.get("capacity", [])
    oportunidad_options = ["undefined"] + comb_options.get("opportunity", [])
    motivacion_options = ["undefined"] + comb_options.get("motivation", [])
    comportamiento_options = ["undefined"] + comb_options.get("tone", [])

    # Function to read files based on selection
    def read_file_if_exists(category, file_name):
        if file_name == "undefined":
            return ""

        file_path = os.path.join(comb_path, category, f"{file_name}.txt")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            return "No additional information available."

    # Function to read client type files
    def read_client_type_file(file_name):
        if file_name == "Select a client type":
            return ""

        file_path = os.path.join(client_type_path, f"{file_name}.txt")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            return "Client type information not available."

    # Branch to customize new role
    if user_type == "Customize a new role":

        # Create dropdown for interaction duration selection
        interaction_duration = st.selectbox(
            "Select interaction duration:",
            ["Fast (3 interactions)", "Medium (5 interactions)", "Long (10 interactions)", "Customized", "No limit"],
            key="interaction_duration_selectbox"
        )

        # Set max interactions based on selection
        if interaction_duration == "Fast (3 interactions)":
            max_interactions = 3
        elif interaction_duration == "Medium (5 interactions)":
            max_interactions = 5
        elif interaction_duration == "Long (10 interactions)":
            max_interactions = 10
        elif interaction_duration == "No limit":
            max_interactions = 999999
        else:
            # Custom number input for interactions
            max_interactions = st.number_input(
                "Maximum number of interactions with the agent:",
                min_value=3,
                max_value=100,
                value=5
            )


        # Styles for the text area
        st.markdown("""<style>.stTextArea textarea { color: #1c58b5; }</style>""", unsafe_allow_html=True)
        
        # Input fields for new role
        role = st.text_input("Define the name of the agent:", key="custom_role_input")

        # Styles for the text area
        st.markdown("""<style>.stTextArea textarea { color: #1c58b5; }</style>""", unsafe_allow_html=True)

        # Goal and task fields
        objective = st.text_area("Define the goal of the agent:", key="custom_objective_input",
                                 help="The description should accurately and comprehensively represent the agent's purpose and goals")

        tasks = st.text_area("Define the tasks of the agent:", key="custom_tasks_input",
                             help="Lists the specific tasks to be performed by the agent")

        # Conversation parameters configuration
        st.markdown("<h3 style='text-align: center; font-size: 28px;'>Configure conversation parameters</h3>",
                    unsafe_allow_html=True)

        # Split into two columns
        col1, col2 = st.columns(2)

        with col1:
            capacidad = st.selectbox("Capacity", capacidad_options, key="capacidad_select",
                                     help="Capacity refers to the physical and psychological abilities that a person needs to carry out a specific behavior.")
            oportunidad = st.selectbox("Opportunity", oportunidad_options, key="oportunidad_select",
                                       help="Opportunity refers to the external factors, such as the social and physical environment, that make it easier or more difficult to perform a behavior.")

        with col2:
            motivacion = st.selectbox("Motivation", motivacion_options, key="motivacion_select",
                                      help="Motivation is the internal process that drives action, including habits, emotions, and decision-making processes.")
            comportamiento = st.selectbox("Tone", comportamiento_options, key="comportamiento_select",
                                          help="Tone is the way they communicate, reflecting their attitude, emotion, and level of formality to adapt to the situation and the customer.")
    
        # Read the selected text files
        capacidad_text = read_file_if_exists("capacity", capacidad)
        oportunidad_text = read_file_if_exists("opportunity", oportunidad)
        motivacion_text = read_file_if_exists("motivation", motivacion)
        comportamiento_text = read_file_if_exists("tone", comportamiento)

        combined_goal = f"{objective}\n"
        combined_goal += f"\n{capacidad_text}\n{oportunidad_text}\n{motivacion_text}\n{comportamiento_text}"

        # Read the file from the client type if one is selected
        client_type_text = read_client_type_file(client_type)
        combined_goal += f"\n{client_type_text}"

        # Uploader for context Excel file
        uploaded_file = st.file_uploader("Upload Excel file to enrich the agent's context (optional)",
                                         type=['xlsx', 'xls'])

        if st.button("Continue", key="custom_continue_button") and role and objective and tasks:
            #  Normalize role name
            safe_role_name = "".join(c for c in role if c.isalnum() or c in (' ', '-', '_')).strip()
            role_dir = os.path.join(base_path, safe_role_name)
            ensure_directory_exists(role_dir)

            # Improve role description
            improved_objective, improved_tasks = improve_role_description(role, combined_goal, tasks)

            st.session_state.custom_role_pending = {
                "role": role,
                "original_objective": objective,
                "original_tasks": tasks,
                "improved_objective": improved_objective,
                "improved_tasks": improved_tasks,
                "role_dir": role_dir,
                "uploaded_file": uploaded_file,
                "max_interactions": max_interactions,
                "client_type": client_type

            }
            st.rerun()

    # Check if there's a pending custom role in the session state
    if "custom_role_pending" in st.session_state:
        # Get the pending role data from session state
        pending = st.session_state.custom_role_pending

        # Create header for comparison section
        st.markdown("### Compare and choose the description you prefer")
        # Create two columns for side-by-side comparison
        col1, col2 = st.columns(2)

        # Left column: Original version
        with col1:
            st.markdown("#### Original Description")
            # Display original objective in disabled text area
            hola = st.text_area("Original objective", pending["original_objective"], disabled=True)
            # Display original tasks in disabled text area
            st.text_area("Original tasks", pending["original_tasks"], disabled=True)
            # Button to select original version
            if st.button("Select Original Version"):
                selected_objective = pending["original_objective"]
                selected_tasks = pending["original_tasks"]
                # Store selection in session state
                st.session_state.selected_version = "original"

        # Right column: Improved version
        with col2:
            st.markdown("#### Improved Description")
            # Display improved objective in disabled text area
            st.text_area("Improved objective", pending["improved_objective"], disabled=True)
            # Display improved tasks in disabled text area
            st.text_area("Improved tasks", pending["improved_tasks"], disabled=True)
            # Button to select improved version
            if st.button("Select Improved Version"):
                selected_objective = pending["improved_objective"]
                selected_tasks = pending["improved_tasks"]
                # Store selection in session state
                st.session_state.selected_version = "improved"

        # Process the selected version if one has been chosen
        if "selected_version" in st.session_state:
            # Save the chosen version to a file
            save_goal_to_file(
                os.path.join(pending["role_dir"], "goal.txt"),
                selected_objective,
                selected_tasks
            )

            # Handle context file if it exists
            context_file_path = None
            if pending["uploaded_file"] is not None:
                # Create path for Excel file
                excel_path = os.path.join(pending["role_dir"], "context.xlsx")
                # Save uploaded file to disk
                with open(excel_path, "wb") as f:
                    f.write(pending["uploaded_file"].getvalue())
                context_file_path = excel_path

            # Combine objective and tasks for the goal
            combined_goal = f"{selected_objective}\n{selected_tasks}"

            # Update session state with final user information
            st.session_state.user_info = {
                "role": pending["role"],
                "goal": combined_goal,
                "objective": selected_objective,
                "tasks": selected_tasks,
                "context_file": context_file_path,
                "client_type": client_type,
                "max_interactions": max_interactions
            }

            # Create agents with the chosen version
            create_agents_crewai(
                pending["role"],
                combined_goal,
                f"./chromadb_agents/vectordb_{pending['role'].lower()}",
                context_file=context_file_path
            )

            # Clear status and start conversation
            del st.session_state.custom_role_pending
            del st.session_state.selected_version
            st.rerun()
            
            chatbot()

            # Manage Excel context file
            context_file_path = None
            if uploaded_file is not None:
                excel_path = os.path.join(role_dir, "context.xlsx")
                with open(excel_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                context_file_path = excel_path
                st.success(f"Excel file saved in {excel_path}")

            # Improved save role file
            save_goal_to_file(role_file_path, improved_objective, improved_tasks)

            # Prepare session information
            combined_goal = f"{improved_objective}\n{improved_tasks}"
            st.session_state.user_info = {
                "role": role,
                "goal": combined_goal,
                "objective": improved_objective,
                "chroma": f"./chromadb_agents/vectordb_{user_type.lower()}",
                "tasks": improved_tasks,
                "context_file": context_file_path,
                "client_type": client_type,
                "max_interactions": max_interactions
            }

            # Create agents with enhanced description
            if context_file_path:
                create_agents_crewai(role, combined_goal, f"./chromadb_agents/vectordb_{user_type.lower()}",
                                     context_file=context_file_path)
            else:
                create_agents_crewai(role, combined_goal, f"./chromadb_agents/vectordb_{user_type.lower()}")
            st.rerun()

    else:
        # Set up the role directory path
        role_dir = os.path.join(base_path, user_type)
        if os.path.exists(role_dir):
            # Look for goal.txt file in the role directory
            goal_files = [f for f in os.listdir(role_dir) if f == 'goal.txt']
        else:
            goal_files = []
            return

    # Process if goal file exists
    if goal_files:
        # Get the full path to the goal file
        selected_file = os.path.join(role_dir, goal_files[0])
        try:
            # Load the goal and tasks from file
            objective, tasks = load_goal_from_file(selected_file)
            combined_goal = f"{objective}\n{tasks}"

            # Create dropdown for interaction duration selection
            interaction_duration = st.selectbox(
                "Select interaction duration:",
                ["Fast (3 interactions)", "Medium (5 interactions)", "Long (10 interactions)", "Customized",
                 "No limit"],
                key="interaction_duration_selectbox"
            )

            # Set max interactions based on selection
            if interaction_duration == "Fast (3 interactions)":
                max_interactions = 3
            elif interaction_duration == "Medium (5 interactions)":
                max_interactions = 5
            elif interaction_duration == "Long (10 interactions)":
                max_interactions = 10
            elif interaction_duration == "No limit":
                max_interactions = 999999
            else:
                # Custom number input for interactions
                max_interactions = st.number_input(
                    "Maximum number of interactions with the agent:",
                    min_value=3,
                    max_value=100,
                    value=5
                )

            # Configure COM-B parameters if user type exists
            if user_type:
                # Display header for conversation parameters
                st.markdown(
                    "<h3 style='text-align: center; font-size: 28px;'>Configure conversation parameters</h3>",
                    unsafe_allow_html=True)

                # Create two-column layout
                col1, col2 = st.columns(2)

                # Left column: Capacity and Opportunity settings
                with col1:
                    capacidad = st.selectbox("Capacity", capacidad_options, key="capacidad_select",
                                             help="Capacity refers to the physical and psychological abilities that a person needs to carry out a specific behavior.")
                    oportunidad = st.selectbox("Opportunity", oportunidad_options, key="oportunidad_select",
                                               help="Opportunity refers to the external factors, such as the social and physical environment, that make it easier or more difficult to perform a behavior.")

                # Right column: Motivation and Tone settings
                with col2:
                    motivacion = st.selectbox("Motivation", motivacion_options, key="motivacion_select",
                                              help="Motivation is the internal process that drives action, including habits, emotions, and decision-making processes.")
                    comportamiento = st.selectbox("Tone", comportamiento_options, key="comportamiento_select",
                                                  help="Tone is the way they communicate, reflecting their attitude, emotion, and level of formality to adapt to the situation and the customer.")

                # Read selected parameter files
                capacidad_text = read_file_if_exists("capacity", capacidad)
                oportunidad_text = read_file_if_exists("opportunity", oportunidad)
                motivacion_text = read_file_if_exists("motivation", motivacion)
                comportamiento_text = read_file_if_exists("tone", comportamiento)

                # Combine all parameters into goal
                combined_goal += f"\n{capacidad_text}\n{oportunidad_text}\n{motivacion_text}\n{comportamiento_text}"

                # Add client type information if selected
                client_type_text = read_client_type_file(client_type)
                combined_goal += f"\n{client_type_text}"

            # Generate and display role summary
            try:
                brief_summary_objective = summarize_text_with_crewai(combined_goal)
                # Create styled container for summary display
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #1c58b5;
                        padding: 15px;
                        border-radius: 10px;
                        background-color: #f9f9f9;
                        margin-top: 20px;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h4 style="color: #1c58b5; margin-bottom: 15px; font-weight: bold; text-align: center;">Know more about the agent</h4>
                        <p style="color: #1c58b5; margin: 0; line-height: 2;">{brief_summary_objective}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                brief_summary_objective = combined_goal

            # File uploader for context Excel file
            uploaded_file = st.file_uploader("Upload Excel file to enrich the agent's context (optional)",
                                             type=['xlsx', 'xls'])

            # Continue button and processing
            if st.button("Continue", key=f"continue_{user_type}"):
                # Handle context file if uploaded
                context_file_path = None
                if uploaded_file is not None:
                    excel_path = os.path.join(role_dir, "context.xlsx")
                    with open(excel_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    context_file_path = excel_path
                    st.success(f"Excel file saved in {excel_path}")

                if "conversation_active" in st.session_state: 
                    del st.session_state.conversation_active
                if "should_process" in st.session_state: 
                    del st.session_state.should_process

                # Set up session state with configuration
                st.session_state.user_info = {
                    "role": user_type,
                    "goal": combined_goal,
                    "objective": objective,
                    "chroma": f"./chromadb_agents/vectordb_{user_type.lower()}",
                    "tasks": tasks,
                    "context_file": context_file_path,
                    "max_interactions": max_interactions,
                    "client_type": client_type
                }

                # Create agents with appropriate configuration
                if context_file_path:
                    create_agents_crewai(user_type, combined_goal,
                                         f"./chromadb_agents/vectordb_{user_type.lower()}",
                                         context_file=context_file_path)
                else:
                    create_agents_crewai(user_type, combined_goal,
                                         f"./chromadb_agents/vectordb_{user_type.lower()}")

                st.rerun()
        except Exception as e:
            st.error(f"Error saving the role: {str(e)}")

    # Section for starting conversation
    if "user_info" in st.session_state:
        st.subheader("Upload Excel file to enrich the agent's context (optional):")
        if st.button("Start a conversation"):
            chatbot()


def chatbot():
    """
    Main chatbot function that handles the conversation flow.
    Includes:
    - User interface setup
    - Message processing
    - Audio handling
    - Document upload
    - Conversation limits
    - Feedback collection
    """
    # Get user information from session state
    user_info = st.session_state.user_info
    role, goal = user_info["role"], user_info["goal"]
    st.title(f"Sandoz ChatBot ({role})")
    st.write("---")
    client_type = user_info.get("client_type", "default")

    # Initialize agent and database
    agent_persist_directory = f"./chromadb_agents/vectordb_{role.lower()}"
    agent, chroma_db, tool_xml = create_agents_crewai(role, goal, persist_directory=agent_persist_directory)
    retriever = chroma_db.as_retriever(search_type="mmr")

    # Initialize tracking variables at session start
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())

    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = time.time()

    # Generate custom greeting
    greeting_message = generate_greeting(client_type, role)

    # Initialize session state variables if not present
    if "messages" not in st.session_state:
        st.session_state.messages = load_history() or [
            {"role": role, "content": greeting_message}
        ]
    
    # Initialize conversation state
    if "should_process" not in st.session_state:
        st.session_state.should_process = False
    if "conversation_active" not in st.session_state:
        st.session_state.conversation_active = True
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    # Initialize interaction counter only if it doesn't exist AND conversation is active
    if "interaction_count" not in st.session_state and st.session_state.conversation_active:
        st.session_state.interaction_count = 0

    # Set maximum interactions from user info
    max_interactions = st.session_state.user_info.get("max_interactions", False)

    st.session_state.user_info["loading_documents"] = False
    st.session_state.user_info["conversation_started"] = True

    # Main conversation loop and UI components
    if st.session_state.user_info.get("conversation_started", False):
        # UI Containers for chat history and user input
        chat_container = st.container()
        input_container = st.container()

        # Display conversation history
        with chat_container:
            st.write("### Conversation history")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "audio" in message:
                        st.audio(message["audio"])

        # User input section
        with input_container:
            st.write("---")
            st.subheader("Type your message or record an audio:")
            col1, col2 = st.columns([4, 1])

            def submit():
                if st.session_state.user_input.strip():
                    st.session_state.should_process = True
                    st.session_state.current_message = st.session_state.user_input
                    st.session_state.user_input = ""

            with col1:
                with st.form("my-form"):
                    st.markdown("""<style>.stTextArea textarea { color: #1c58b5; }</style>""", unsafe_allow_html=True)
                    
                    user_input = st.text_input("Type your message here:", key="user_input")
                    col_form1, col_form2 = st.columns([3, 2])
                    with col_form1:
                        submit_button = st.form_submit_button("Send", on_click=submit)
                    with col_form2:
                        if max_interactions != 999999 and st.session_state.conversation_active:
                            remaining_interactions = max_interactions - st.session_state.interaction_count
                            st.markdown(
                                f"<p style='color: #1c58b5; margin-top: 5px;'>Remaining interactions: {remaining_interactions}</p>",
                                unsafe_allow_html=True)

            with col2:
                with st.container():
                    if st.button("🎤", key="voice_input", use_container_width=True):
                        audio_input = transcribe_audio()
                        if audio_input:
                            st.session_state.current_message = audio_input
                            st.session_state.should_process = True
                    if st.button("PDF Upload", use_container_width=True):
                        st.session_state.show_pdf_upload = not st.session_state.get("show_pdf_upload", False)

            # PDF upload section
            if st.session_state.get("show_pdf_upload", False):
                st.write("---")
                col_upload1, col_upload2 = st.columns([3, 1])
            
                with col_upload1:
                    with st.spinner("Processing document..."):
                        upload_and_process_documents(chroma_db, agent.role, key_prefix="default")
                        st.session_state.user_info["loading_documents"] = True
            
                with col_upload2:
                    st.write("")
                    st.write("")
                    if st.button("❌ Close", key="close_uploader"):
                        st.session_state.show_pdf_upload = False
                        st.rerun()

            st.write("---")
            # End conversation button
            if st.button("End conversation", key="end_conversation_button", use_container_width=True):
                st.session_state["showing_feedback"] = True
                st.session_state.conversation_active = False
                st.session_state.interaction_count = 0
                clear_history()
                st.rerun()

            # Feedback form
            if st.session_state.get("showing_feedback", False):
                with st.form("feedback_form"):
                    st.write("Please provide your feedback:")
                    rating = st.slider("How would you rate your experience?", 1, 5, 3)
                    comments = st.text_area("Any additional comments?")

                    submit_feedback = st.form_submit_button("Submit Feedback")

                    if submit_feedback:
                        success, message = save_feedback(rating, comments)
                        if success:
                            st.success(message)
                            # Clear all relevant session state
                            for key in ["user_info", "agent", "showing_feedback", "messages", 
                                      "interaction_count", "conversation_active"]:
                                if key in st.session_state:
                                    st.session_state.pop(key)
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(message)

        # Process messages and enforce limits
        if st.session_state.should_process and st.session_state.conversation_active:
            current_message = st.session_state.current_message
            st.session_state.messages.append({"role": "user", "content": current_message})
            save_history(st.session_state.messages)

            st.session_state.interaction_count += 1
            final_interaction = st.session_state.interaction_count >= max_interactions

            with st.spinner("Loading..."):
                answer = process_message(agent, current_message, tool_xml, retriever, chroma_db, use_mmr=True, top_k=5,
                                      final_interaction=final_interaction)
                audio_response = text_to_speech(answer)

                st.session_state.messages.append({
                    "role": role,
                    "content": answer,
                    "audio": audio_response
                })
                save_history(st.session_state.messages)

            if final_interaction:
                st.warning("The negotiation has reached its limit. The agent has made a final decision.")
                # Reset conversation state
                st.session_state.conversation_active = False
                st.session_state.interaction_count = 0
                clear_history()
                # Ensure these changes persist
                st.session_state["reset_complete"] = True

            st.session_state.should_process = False
            st.session_state.current_message = ""
            st.rerun()

def simulate_roles():
    """
    Manages the setup and initialization of a conversation simulation between two agents.
    Features:
    - Selection of existing agents or creation of custom agents
    - Definition of conversation topics
    - Dynamic agent configuration
    - Personality and goal management
    - Automatic improvement of role descriptions
    """
    
    # Set base path for personality configuration files
    base_path = os.path.join("config", "base", "personalidad")

    # Get list of existing roles and add custom role option
    role_names = get_existing_roles(base_path)
    all_options = role_names + ["Customize a new role"]

    # Check if simulation setup is already completed
    if "simulation" not in st.session_state or not st.session_state.simulation.get("setup_complete", False):
        with st.form("setup_form"):
            st.title("Conversation Simulation between Agents")
            # Agent selection interface
            role_1 = st.selectbox("Select the first agent:", all_options, key="agent_1")
            role_2 = st.selectbox("Select the second agent:", all_options, key="agent_2")

            # Styles for the text area
            st.markdown("""<style>.stTextArea textarea { color: #1c58b5; }</style>""", unsafe_allow_html=True)

            # Define conversation topic
            topic = st.text_input("Define the conversation topic:")

            # Storage for custom role configurations
            custom_role_data = {}

            # Handle custom role creation for both agents
            for i, role in enumerate([role_1, role_2], start=1):
                if role == "Customize a new role":
                    custom_role_data[f"custom_role_{i}"] = st.text_input(f"Agent {i} name:")
                    custom_role_data[f"custom_goal_{i}"] = st.text_area(f"Agent {i} objective:")
                    custom_role_data[f"custom_tasks_{i}"] = st.text_area(f"Agent {i} tasks:")

            # Validate form submission and required fields
            if st.form_submit_button("Continue") and (
                    (role_1 != "Customize a new role" or
                     (custom_role_data["custom_role_1"] and custom_role_data["custom_goal_1"] and custom_role_data[
                         "custom_tasks_1"])) and
                    (role_2 != "Customize a new role" or
                     (custom_role_data["custom_role_2"] and custom_role_data["custom_goal_2"] and custom_role_data[
                         "custom_tasks_2"]))
            ):
                # Ensure topic is defined
                if not topic:
                    st.warning("Please enter a conversation topic.")
                    return

                # Initialize session storage for simulation
                st.session_state.simulation = {
                    "topic": topic,
                    "setup_complete": True,
                    "messages": [],
                    "current_agent": 1
                }

                # Process and create agents
                for i, role in enumerate([role_1, role_2], start=1):
                    if role == "Customize a new role":
                        # Handle custom role creation
                        custom_role = custom_role_data[f"custom_role_{i}"]
                        custom_goal = custom_role_data[f"custom_goal_{i}"]
                        custom_tasks = custom_role_data[f"custom_tasks_{i}"]

                        # Enhance role description using AI
                        improved_objective, improved_tasks = improve_role_description(
                            custom_role,
                            custom_goal,
                            custom_tasks
                        )

                        # Initialize custom agent
                        agent, chroma_db, tool_xml = create_agents_crewai(
                            custom_role,
                            improved_objective,
                            f"./chromadb_agents/vectordb_{custom_role.lower()}"
                        )

                        # Store custom agent configuration
                        st.session_state.simulation[f"agent_{i}"] = {
                            "role": custom_role,
                            "goal": improved_objective,
                            "tasks": improved_tasks,
                            "agent": agent,
                            "chroma_db": chroma_db
                        }
                    else:
                        # Handle existing role setup
                        role_dir = os.path.join(base_path, role)
                        goal_file = os.path.join(role_dir, "goal.txt")

                        # Load existing role configuration
                        if os.path.exists(goal_file):
                            goal, tasks = load_goal_from_file(goal_file)
                        else:
                            st.warning(f"Goal file not found for {role}.")
                            return

                        # Initialize existing agent
                        agent, chroma_db, tool_xml = create_agents_crewai(
                            role,
                            goal,
                            f"./chromadb_agents/vectordb_{role.lower()}"
                        )

                        # Store existing agent configuration
                        st.session_state.simulation[f"agent_{i}"] = {
                            "role": role,
                            "goal": goal,
                            "tasks": tasks,
                            "agent": agent,
                            "chroma_db": chroma_db
                        }

        if "agent_1" in st.session_state.simulation and "agent_2" in st.session_state.simulation:
            st.rerun()

        # Start conversation if setup is complete
    if st.session_state.simulation.get("setup_complete", False):
        conversation_simulation()


def conversation_simulation():
    """
    Manages the simulation of conversations between two agents.
    Features:
    - Initiates conversation based on the specified topic
    - Alternates turns between agents
    - Handles message history and audio responses
    - Provides conversation control through continue/end buttons
    """
    # Verify simulation configuration exists
    if "simulation" not in st.session_state:
        st.error("Please configure the simulation in 'simulate_roles' first.")
        return

    simulation = st.session_state.simulation

    # Validate agents and topic configuration
    if "agent_1" not in simulation or "agent_2" not in simulation:
        st.error("Agent configuration error. Please reconfigure the simulation.")
        return
    if "topic" not in simulation or not simulation["topic"]:
        st.error("No conversation topic has been defined.")
        return

    # Get agent configurations
    agent_1 = simulation["agent_1"]
    agent_2 = simulation["agent_2"]

    # Set up the UI
    st.title("Agent Conversation Simulation")
    st.subheader(f"Topic: {simulation['topic']}")

    # Display conversation history with audio if available
    for msg in simulation["messages"]:
        st.write(f"**{msg['role']}:** {msg['content']}")
        if "audio" in msg:
            st.audio(msg["audio"])

    # Initialize conversation if no previous messages exist
    if not simulation["messages"]:
        # Show loading spinner while first agent generates response
        with st.spinner(f"{agent_1['role']} is starting the conversation..."):
            # Generate initial message
            response = process_message(
                agent_1["agent"],
                f"Start a conversation about {simulation['topic']} with {agent_2['role']}",
                agent_1["chroma_db"],
                agent_1["chroma_db"]
            )

            # Convert response to audio
            audio_response = text_to_speech(response)

            # Add first message to history
            simulation["messages"].append({
                "role": agent_1["role"],
                "content": response,
                "audio": audio_response
            })

            # Switch turn to second agent
            simulation["current_agent"] = 2
            st.rerun()

    # Create button layout
    col1, col2 = st.columns(2)

    # Continue conversation button
    with col1:
        if st.button("Continue conversation"):
            # Get current agent's turn and data
            current_num = simulation["current_agent"]
            current_agent_data = simulation[f"agent_{current_num}"]
            # Determine the other agent's data
            other_agent_data = simulation[f"agent_{1 if current_num == 2 else 2}"]

            # Process next message
            with st.spinner(f"{current_agent_data['role']} is responding..."):
                # Get the last message from history
                last_message = simulation["messages"][-1]["content"]
                # Generate response to previous message
                response = process_message(
                    current_agent_data["agent"],
                    f"Respond to this message from {other_agent_data['role']}: {last_message}",
                    current_agent_data["chroma_db"],
                    current_agent_data["chroma_db"]
                )

                # Generate audio for response
                audio_response = text_to_speech(response)

                # Add response to conversation history
                simulation["messages"].append({
                    "role": current_agent_data["role"],
                    "content": response,
                    "audio": audio_response
                })

                # Switch turn to the other agent
                simulation["current_agent"] = 1 if current_num == 2 else 2
                st.rerun()

    # End conversation button
    with col2:
        if st.button("End conversation"):
            # Clean up simulation state
            del st.session_state["simulation"]
            clear_history()
            st.success("Simulation completed.")
            st.rerun()
