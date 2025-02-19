import os
import crewai
import streamlit as st
from pathlib import Path
from crewai_tools import FileReadTool
from langchain.vectorstores import Chroma
from src.message_handling import excel_to_xml, initialize_chroma_db, load_message_from_file
from src.utils.prepare_environment import set_api_keys


def create_improvement_agent():
    """
    Creates and configures an agent specifically for improving role descriptions.

    Returns:
        crewai.Agent: Configured improvement agent
    """
    return crewai.Agent(
        model="gpt-4o-mini",
        api_key=os.getenv('OPENAI_API_KEY'),
        api_base=os.getenv('OPENAI_API_BASE'),
        role=load_message_from_file(r"src\utils\create_improvement_agent\role.txt"),
        goal=load_message_from_file(r"src\utils\create_improvement_agent\goal.txt"),
        backstory=load_message_from_file(r"src\utils\create_improvement_agent\backstory.txt")
    )


def improve_role_description(role, objective, tasks):
    """
    Improves the description and tasks of a role using an AI agent.

    Args:
        role (str): Name of the role
        objective (str): Role objective
        tasks (str): Role tasks

    Returns:
        tuple: (improved_objective, improved_tasks)
    """
    try:
        # Validate input parameters
        if not all([role.strip(), objective.strip(), tasks.strip()]):
            return "Error: All fields must contain valid information.", ""

        # Create improvement agent
        improvement_agent = create_improvement_agent()

        # Process objective improvement
        objective_text = load_message_from_file(r"src\utils\improve_role_description\objective_prompt.txt")

        if not objective_text:
            return "Error: No se pudo cargar el archivo objective_prompt.txt.", ""

        if not objective:
            return "Error: La variable 'objective' está vacía o no definida.", ""

        objective_prompt = objective_text.format(role=role, objective=objective)

        objective_task = crewai.Task(
            description=objective_prompt,
            expected_output=f"Objetivo mejorado para el rol {role}.",
            agent=improvement_agent
        )

        improved_objective = improvement_agent.execute_task(objective_task)

        # Process tasks improvement
        tasks_text = load_message_from_file(r"src\utils\improve_role_description\tasks_prompt.txt")

        if not tasks_text:
            return "Error: No se pudo cargar el archivo tasks_prompt.txt.", ""

        if not tasks:
            return "Error: La variable 'tasks' está vacía o no definida.", ""

        tasks_prompt = tasks_text.format(role=role, tasks=tasks)

        tasks_task = crewai.Task(
            description=tasks_prompt,
            expected_output=f"Tareas mejoradas para el rol {role}.",
            agent=improvement_agent
        )

        improved_tasks = improvement_agent.execute_task(tasks_task)

        return improved_objective, improved_tasks

    except KeyError as e:
        return f"Error: Falta la variable de entorno {str(e)}", ""

    except Exception as e:
        return f"Error inesperado: {str(e)}", ""


def create_agents_crewai(role, goal, persist_directory, context_file=None):
    """
    Creates a CrewAI agent with role-specific tools and configurations.

    Args:
        role (str): Name of the agent role
        goal (str): Goal/objective for the agent
        persist_directory (str): Directory for persisting agent data
        context_file (str, optional): Path to context file

    Returns:
        tuple: (CrewAI agent, Chroma database, FileReadTool)
    """
    # Set keys
    set_api_keys()

    # Set up base paths and file paths
    base_path = os.path.join("config", "base", "personalidad", role)
    output_xml = os.path.join(base_path, "context_output.xml")
    if context_file and os.path.exists(context_file):
        if not os.path.exists(output_xml) or os.path.getmtime(context_file) > os.path.getmtime(
                output_xml):  ## si no existe o no esta actualizado
            excel_to_xml(context_file, output_xml)

    # Initialize file reading tool if context file exists
    if not os.path.exists(output_xml):
        file_read_tool = None
    else:
        file_read_tool = FileReadTool(file_path=output_xml)

    # Initialize Chroma database
    chroma_db = initialize_chroma_db(role, persist_directory)

    # Create and configure the agent
    agent = crewai.Agent(
        role=role,
        goal=goal,
        backstory=load_message_from_file(r"src\utils\create_agents_crewai\backstory.txt").replace("{goal}", goal),
        tools=[file_read_tool] if file_read_tool else [],
        memory=True,
        llm=crewai.LLM(
            model="gpt-4o-mini",
            api_key=os.environ['OPENAI_API_KEY'],
            api_base=os.environ['OPENAI_API_BASE']
        ),
        verbose=True
    )

    return agent, chroma_db, file_read_tool
