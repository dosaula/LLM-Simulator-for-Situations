import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import pandas as pd
import os
import json
from textwrap import dedent
import crewai
import re
import time
import tempfile
from pathlib import Path
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.prepare_environment import set_api_keys
from langchain.vectorstores import Chroma
from datetime import datetime
import platform
import uuid

def load_message_from_file(file_path):
    """
    Loads and returns the content of a message file.

    Args:
        file_path (str): Path to the message file

    Returns:
        str: Content of the file, stripped of leading/trailing whitespace
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()
  
  
def get_existing_roles(base_path):
    """
    Retrieves a list of existing roles from the directory and its subdirectories.

    Args:
        base_path (str): Base directory path to search for roles

    Returns:
        list: List of role names found in the directories
    """
    try:
        roles = []
        # Walk through directory structure
        for root, dirs, files in os.walk(base_path):
            for dir_name in dirs:
                role_dir = os.path.join(root, dir_name)
                # Check for goal.txt files to identify valid role directories
                txt_files = [f for f in os.listdir(role_dir) if f == 'goal.txt']
                if txt_files:
                    roles.append(dir_name)
        return roles
    except Exception as e:
        st.error(f"Error when reading the roles: {str(e)}")
        return []
    

def get_existing_comb():
    """
    Retrieves existing COM-B (Capability, Opportunity, Motivation, Behavior) files from the configuration directory.

    Returns:
        dict: Dictionary containing lists of files for each COM-B category
    """
    base_comb_path = os.path.join("config", "base", "personalidad", "comb")
    categories = ["capacity", "opportunity", "motivation", "tone"]
    comb_files = {category: [] for category in categories}

    try:
        for category in categories:
            category_path = os.path.join(base_comb_path, category)
            if os.path.exists(category_path):
                # Get filenames without .txt extension
                txt_files = [f.replace(".txt", "") for f in os.listdir(category_path) if f.endswith(".txt")]
                comb_files[category] = txt_files
        return comb_files
    except Exception as e:
        st.error(f"Error reading COM-B files: {str(e)}")
        return {}
    

def load_goal_from_file(file_path):
    """
    Loads objective and tasks from a goal file.

    Args:
        file_path (str): Path to the goal file

    Returns:
        tuple: (objective, tasks) strings separated by '---' marker
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        # Split content into objective and tasks
        parts = content.split("\n---\n")
        if len(parts) >= 2:
            return parts[0], parts[1]
        return content, ""  # For legacy files without separator

def summarize_text_with_crewai(text):
    """
    Uses a CrewAI agent to generate a summary of the provided text.

    Args:
        text (str): Text to be summarized

    Returns:
        str: Summarized text or error message
    """
    try:
        set_api_keys()
        # Create and configure summarization agent
        agent = crewai.Agent(
            model="gpt-4o-mini",
            api_key=os.environ['OPENAI_API_KEY'],
            api_base=os.environ['OPENAI_API_BASE'],
            role=load_message_from_file(r"src\utils\summarize_text_with_crewai\role.txt"),
            goal=load_message_from_file(r"src\utils\summarize_text_with_crewai\goal.txt"),
            backstory=load_message_from_file(r"src\utils\summarize_text_with_crewai\backstory.txt")
        )
        # Create and execute summarization task
        give_answer = crewai.Task(
            description=load_message_from_file(r"src\utils\summarize_text_with_crewai\description.txt").replace(
                "{text}", text),
            expected_output=load_message_from_file(r"src\utils\summarize_text_with_crewai\expected_output.txt"),
            agent=agent)
        response = agent.execute_task(give_answer)
        return response

    except Exception as e:
        return f"Error al generar el resumen: {e}"
    
    
def transcribe_audio():
    """
    Captures audio from the microphone and converts it into text using Google Speech Recognition.

    Returns:
        str: Transcribed text if successful, otherwise None.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Recording... Speak now.")  # Notify user that recording has started
        audio = recognizer.listen(source)  # Capture audio input

        try:
            # Convert audio to text using Google's speech recognition service
            transcription = recognizer.recognize_google(audio, language="es-ES")
            st.success("Transcription completed.")
            return transcription
        except sr.UnknownValueError:
            st.error("The audio could not be understood. Try again!")  # Error if speech is not recognized
        except sr.RequestError:
            st.error("Error in the voice recognition service. Try again!")  # Error if service request fails
    return None


def text_to_speech(text, language="es"):
    """
    Converts text into speech and returns an audio buffer.

    Args:
        text (str): The text to be converted to speech.
        language (str): The language for text-to-speech (default is Spanish).

    Returns:
        BytesIO: Audio buffer containing the generated speech.
    """
    tts = gTTS(text, lang=language, tld="com" if language == "en" else "es")
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)  # Save the generated audio into the buffer
    audio_buffer.seek(0)  # Reset buffer position for playback
    return audio_buffer


def excel_to_xml(input_excel, output_xml, root_element="dic", item_element="w", from_lang="es", to_lang="xml_format"):
    """
    Converts an Excel file into an XML file, organizing data into a structured format.

    Args:
        input_excel (str): Path to the input Excel file.
        output_xml (str): Path to the output XML file.
        root_element (str): Name of the root XML element (default: "dic").
        item_element (str): Name of the item elements inside the XML (default: "w").
        from_lang (str): Source language (default: "es").
        to_lang (str): Target language/format (default: "xml_format").

    This function reads the Excel file, processes its data, and writes it in an XML format,
    preserving the structure of sheets and columns.
    """
    with open(output_xml, "w", encoding="utf-8") as xml_file:
        # Write XML header
        xml_file.write(f'<?xml version="1.0" encoding="UTF-8"?>\n')
        xml_file.write(f'<{root_element} from="{from_lang}" to="{to_lang}">\n')

        # Retrieve all sheet names from the Excel file
        sheet_names = pd.ExcelFile(input_excel).sheet_names
        for sheet_name in sheet_names:
            # Read each sheet into a DataFrame
            df = pd.read_excel(input_excel, sheet_name=sheet_name)

            # Clean column names for XML compatibility
            df.columns = [
                col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")
                for col in df.columns
            ]

            # Include each sheet as a separate block in the XML
            xml_file.write(f'  <sheet name="{sheet_name}">\n')

            # Process each row in the DataFrame
            for _, row in df.iterrows():
                xml_file.write(f'    <{item_element}>\n')

                # Check if "Medicamento" and "mes" columns exist
                medicamento = row['Medicamento'] if 'Medicamento' in row else ''
                mes = row['mes'] if 'mes' in row else ''

                for col_name, value in row.items():
                    # Ensure values are safe for XML format
                    clean_value = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                    # Construct context based on available columns
                    if medicamento and mes:
                        context = f'{medicamento}_{mes}_{col_name}'
                        xml_file.write(f'<{col_name}>{context} = {clean_value},</{col_name}>\n')
                    elif medicamento:
                        context = f'{medicamento}_{col_name}'
                        xml_file.write(f'<{col_name}>{context} = {clean_value},</{col_name}>\n')
                    else:
                        xml_file.write(f'<{col_name}>{col_name} = {clean_value},</{col_name}>\n')

                xml_file.write(f'    </{item_element}>\n')
            xml_file.write(f'  </sheet>\n')

        # Close root element
        xml_file.write(f'</{root_element}>\n')
    print(f"File processed correctly: {output_xml}")


HISTORY_FILE = "history.json"   
    
def load_history():
    """
    Loads the conversation history from history.json, handling JSON errors.

    Returns:
        list: A list of messages from the conversation history. If the file is empty or corrupt, returns an empty list.
    """
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                data = file.read().strip()
                return json.loads(data) if data else []  # Return empty list if the file is empty
        except (json.JSONDecodeError, ValueError):
            print("⚠️ Error: history.json is corrupted. It will be deleted and reset.")
            clear_history()  # Delete the corrupted history file
            return []
    return []


def save_history(messages):
    """
    Saves the conversation history to history.json, ignoring non-serializable data.

    Args:
        messages (list): A list of message dictionaries to be stored.

    This function filters out non-serializable data (like BytesIO objects for audio) before saving.
    """
    messages_serializable = [
        {key: value for key, value in msg.items() if key != "audio"} for msg in messages
    ]

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(messages_serializable, file, ensure_ascii=False, indent=4)


def clear_history():
    """
    Deletes the conversation history file if it exists.
    """
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        
    st.session_state.pop("session_start_time", None)
    st.session_state.pop("response_times", None)
    st.session_state.pop("last_message_time", None)


def process_message(agent, message, file_read_tool, retriever=None, vectorstore=None, use_mmr=False, top_k=5, final_interaction=False):
    """
    Processes a user message using the agent and conversation history.

    Args:
        agent: The AI agent responsible for generating responses.
        message (str): The user's message.
        file_read_tool: A tool for reading files if required.
        retriever: An optional retriever for fetching additional context (default: None).
        vectorstore: An optional vector store for semantic search (default: None).
        use_mmr (bool): Whether to use Maximal Marginal Relevance for ranking (default: False).
        top_k (int): Number of top-ranked documents to retrieve (default: 5).
        final_interaction (bool): Flag indicating whether it's the final message in the interaction.

    Returns:
        str: The response generated by the AI agent.
    """
    try:
        set_api_keys()  # Ensure API keys are set for external services

        # Validate the file path before using the file read tool
        if file_read_tool and not os.path.exists(file_read_tool.file_path):
            file_read_tool = None

        # Assign tools if available
        tools = [file_read_tool] if file_read_tool else []

        # Load conversation history
        history = load_history()
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])

        # Buscar documentos relevantes si el retriever está disponible
        retrieved_docs = []
        if retriever:
            try:
                retrieved_docs = retriever.get_relevant_documents(message)
            except Exception as e:
                pass  # Ignore errors in document retrieval

        # Concatenar los documentos recuperados en el contexto
        retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

        # Construir el contexto final con la información recuperada
        context = f"{history_text}\n\n[DOCUMENTOS RECUPERADOS]\n{retrieved_text}\n\n[USUARIO]: {message}"

        # Determine the agent's role
        agent_role = agent.role.lower()

        keywords = [
            "farmaceutico", "farmacéutico", "boticario", "químico farmacéutico", "apotecario", 
            "negociador", "mediador", "intermediario", "conciliador", "diplomático", "arbitro", 
            "gestor de conflictos", "facilitador", "moderador", "comerciante", "vendedor", 
            "estratega", "consultor", "representante comercial", "comercial"
        ]

        # Load the task description based on interaction type
        if final_interaction and any(keyword in agent_role.lower() for keyword in keywords):
            task_description = dedent(
                load_message_from_file(r"src\utils\process_message\final_task_description.txt")
            ).replace("{message}", context)
        else:
            task_description = dedent(
                load_message_from_file(r"src\utils\process_message\task_description.txt")
            ).replace("{message}", context)

        # Load expected output format
        expected_output = dedent(
            load_message_from_file(r"src\utils\process_message\expected_output.txt")
        ).replace("{retrieved_info}", retrieved_text if retrieved_text else "No se encontraron documentos relevantes.")

        # Define the AI task
        give_answer = crewai.Task(
            description=task_description,
            expected_output=expected_output,
            agent=agent,
            tools=tools,
            additional_context=context
        )

        # Execute the task and generate a response
        response = agent.execute_task(give_answer)

        # Save the assistant's response in history
        history.append({"role": agent_role, "content": response})
        save_history(history)

        return response

    except Exception as e:
        return f"Error procesando la solicitud: {e}"


def initialize_chroma_db(agent_role, persist_directory):
    """
    Initializes a Chroma DB instance, creating the persistence directory if it doesn't exist.

    Args:
        agent_role (str): The role of the agent.
        persist_directory (str): Path to the directory where Chroma data will be stored.

    Returns:
        Chroma: An initialized instance of Chroma DB.

    Raises:
        OSError: If there are issues creating the persistence directory.
    """
    try:
        set_api_keys()
        api_key = os.environ['OPENAI_API_KEY']
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        # Ensure the persistence directory exists
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        # Create a persistent Chroma client
        persistent_client = chromadb.PersistentClient(persist_directory)

        # Initialize the Chroma database
        chrom_adb = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings
        )
        return chrom_adb

    except OSError as e:
        raise OSError(f"Error creating the persistence directory: {e}")


def upload_and_process_documents(chroma_db, agent_role, key_prefix="default"):
    """
    Processes documents to add to the agent's context.
    If the agent already has context, it processes only relevant documents.

    Args:
        chroma_db (Chroma): The Chroma database instance.
        agent_role (str): The role of the agent.
        key_prefix (str): A prefix for file uploader keys to avoid conflicts.

    Returns:
        bool: True if all files were processed successfully, False otherwise.
    """
    agent_role = agent_role.strip() if agent_role else None

    def clean_documents(split_documents):
        """
        Cleans extracted text from documents.

        Args:
            split_documents (list): List of document chunks.

        Returns:
            tuple: A list of cleaned text content and corresponding metadata.
        """
        documents_content = []
        metadata = []
        for doc in split_documents:
            if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
                cleaned_text = doc.page_content.replace("\n", " ").replace("\r", "").replace("\xa0", " ").strip()
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with a single space
                documents_content.append(cleaned_text)
            else:
                st.warning(f"The document has no valid content in page_content.")
            metadata.append(doc.metadata.get('source', 'unknown'))
        return documents_content, metadata

    def get_existing_paths(chroma_db):
        """
        Retrieves existing document paths from the Chroma database.

        Args:
            chroma_db (Chroma): The Chroma database instance.

        Returns:
            list: A list of unique document paths.
        """
        metadatas = chroma_db.get()["metadatas"]
        existing_paths = [item["source"] for item in metadatas if isinstance(item, dict) and "source" in item]
        return list(set(existing_paths))

    col = st.container()

    with col:
        documents_content = []
        metadata = []

        # Define the agent's document storage path
        base_agent_path = os.path.join("config", "base", "personalidad", agent_role) if agent_role else None

        # Get existing documents in the agent's database
        existing_paths = get_existing_paths(chroma_db)

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your text or PDF files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key=f"file_uploader_{key_prefix}"
        )

        if uploaded_files:
            all_successful = True
            for uploaded_file in uploaded_files:
                st.write(f"Processing file: {uploaded_file.name}")
                original_filename = uploaded_file.name
                tmp_file_path = None

                try:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_file_path = tmp_file.name

                    # Process the uploaded file
                    if uploaded_file.type == "text/plain":
                        loader = TextLoader(tmp_file_path)
                    elif uploaded_file.type == "application/pdf":
                        loader = PyPDFLoader(tmp_file_path)
                    else:
                        st.error(f"Unsupported file format for {uploaded_file.name}")
                        all_successful = False
                        continue

                    documents = loader.load()
                    for doc in documents:
                        if 'source' in doc.metadata:
                            doc.metadata['source'] = original_filename

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
                    )
                    split_documents = text_splitter.split_documents(documents)
                    if not documents:
                        st.error(f"Failed to extract text from {uploaded_file.name}. Check the format.")
                        all_successful = False
                        continue

                    st.write(f"Successfully read content from {uploaded_file.name}.")

                    # Clean text and filter unique documents
                    documents_content, metadata = clean_documents(split_documents)
                    documents_to_add = []
                    metadata_to_add = []

                    for doc, meta in zip(documents_content, metadata):
                        if meta not in existing_paths:
                            metadata_to_add.append({"source": meta})
                            documents_to_add.append(doc)

                    # Save documents to the agent's folder and add them to the database
                    if documents_to_add:
                        chroma_db.add_texts(texts=documents_to_add, metadatas=metadata_to_add)
                        st.success("Documents added to the agent's context and persisted.")

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    all_successful = False
                finally:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

            if all_successful:
                st.success("All files have been processed successfully.")
                return True
            else:
                st.warning("Some files could not be processed successfully.")
                return False
        else:
            st.markdown("No documents uploaded.")
        return False


def save_feedback(rating, comments):
    """
    Save user feedback in a JSON file including session metrics and COMB personality.
    """
    try:
        feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        feedback_file = os.path.join(feedback_dir, "feedback.json")
        
        # Interaction count
        interaction_count = len(st.session_state.get("messages", [])) // 2  # Divide by 2 because each interaction has a user message and assistant response
        
        # Get COMB values from user_info
        comb_info = {
            "Capacity": st.session_state.user_info.get("capacity", "Undefined"),
            "Opportunity": st.session_state.user_info.get("opportunity", "Undefined"),
            "Motivation": st.session_state.user_info.get("motivation", "Undefined")
        }
        
        # Generate unique ID for the conversation
        conversation_id = st.session_state.get("conversation_id", str(uuid.uuid4()))
        
        feedback_entry = {
            "conversation_id": conversation_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rating": rating,
            "comments": comments,
            "username": st.session_state.get("username", "Anonymous"),
            "role": st.session_state.user_info.get("role", "Unknown"),
            "comb": comb_info,
            "session_duration": time.time() - st.session_state.get("session_start_time", time.time()),
            "interaction_count": interaction_count,
            "documents_uploaded": bool(st.session_state.user_info.get("loading_documents", False)),
            "browser_info": {
                "platform": platform.system(),
                "browser": st.session_state.get("browser_info", "Unknown")
            }
        }

        try:
            if os.path.exists(feedback_file):
                with open(feedback_file, "r", encoding="utf-8") as f:
                    feedbacks = json.load(f)
            else:
                feedbacks = []
        except json.JSONDecodeError:
            feedbacks = []

        feedbacks.append(feedback_entry)

        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, indent=4, ensure_ascii=False)

        return True, "Feedback saved successfully!"

    except Exception as e:
        return False, f"Error saving feedback: {str(e)}"