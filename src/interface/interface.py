import streamlit as st
from PIL import Image
import json
import base64
from io import BytesIO
import os
import time
from src.message_handling import clear_history


# Interface start
if "app_stage" not in st.session_state:
    st.session_state["app_stage"] = "select_user"


# Function to convert the image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="WEBP")
    return base64.b64encode(buffered.getvalue()).decode()


# Function to load the CSS file
def load_css(file_name):
    with open(file_name, "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

load_css("styles.css")
logo = Image.open("Sandoz-Logo-Sandoz-Blue-RGB.webp")


def header_with_image(title, image_path):
    """
    Displays a header with an image and a title, centered using CSS classes.
    
    Args:
        title (str): The header title.
        image_path (str): Path or URL of the image.
    """
    img_base64 = image_to_base64(image_path)
    
    st.markdown(f"""
    <style>
        .header-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }}
        .logo-container img {{
            width: 600px;
            height: auto;
        }}
        .header-space {{
            margin-top: 20px;
        }}
    </style>
    <div class="header-container">
        <div class="logo-container">
            <img src="data:image/webp;base64,{img_base64}" alt="Logo">
        </div>
        <div class="header-space">
            <h1>{title}</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
# File path where we will save the users
USERS_DB_FILE = "users_db.json"

# Function to load users from the JSON file
def load_users():
    # Check if the file exists
    if not os.path.exists(USERS_DB_FILE):
        # If it does not exist, create it with an empty dictionary
        with open(USERS_DB_FILE, "w") as f:
            json.dump({}, f)
        return {}
    # If it exists, load the data
    with open(USERS_DB_FILE, "r") as f:
        return json.load(f)


# Function to save users to the JSON file
def save_users(users):
    with open(USERS_DB_FILE, "w") as f:
        json.dump(users, f)


# Load user database at startup
users_db = load_users()


def show_login_message():
    header_with_image("Sandoz ChatBot", logo)
    # CSS to center and style the message
    st.markdown(
        """
        <style>
        .centered-message {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 50vh; /* Adjust the height as needed */
            color: #1c58b5;
            font-size: 28px;
            font-weight: bold;
        }
        </style>
        <div class="centered-message">
            Please, login in the side menu to continue.
        </div>
        """,
        unsafe_allow_html=True
    )


# Function to display the sidebar, including the log-out option
def show_sidebar():


    # Side panel
    st.sidebar.image(logo, use_container_width=True)
    st.sidebar.markdown(
        """
        <style>
            .sidebar-title {
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                color: #1c58b5;
                font-family: 'Roboto', sans-serif;
            }
        </style>
        <div class="sidebar-title">
            LLM
        </div>
        """, unsafe_allow_html=True
    )

    st.sidebar.title("Login")

    # Check if a user is already logged in
    if "username" in st.session_state and st.session_state["username"] not in [None, ""]:
        st.sidebar.write(f"You are logged in as: {st.session_state['username']}")
        time.sleep(0.5)

        if st.sidebar.button("Log out"):
            del st.session_state["username"]
            st.sidebar.success("Session closed. Back to the start...")
            time.sleep(0.5)
            clear_history()
            st.rerun()  # Redirect to restart the session and return to login

    else:
        # Login options
        mode = st.sidebar.radio("Select option:", ["Log in", "Sign up", "Guest"])

        # If "Guest" is selected, no username or password is required
        if mode == "Guest":
            st.session_state["username"] = "Guest"  # This is saved to keep the session active
            st.sidebar.success("Welcome as guest!")
            time.sleep(0.5)
            st.rerun()  # Redirect to update the interface and show the Guest session

        else:
            # Fields for username and password
            username = st.sidebar.text_input("User")
            password = st.sidebar.text_input("Password", type="password")

            if mode == "Log in":
                # Login logic
                if st.sidebar.button("Log in"):
                    if username in users_db and users_db[username] == password:
                        st.session_state["username"] = username
                        st.sidebar.success(f"Welcome back {username}!")
                        st.rerun()  # Redirect to update the interface
                    else:
                        st.sidebar.error("Incorrect username or password.")

            elif mode == "Sign up":
                # New user registration logic
                if st.sidebar.button("Sign up"):
                    if username in users_db:
                        st.sidebar.error("This user already exists.")
                        time.sleep(0.5)
                    else:
                        users_db[username] = password  # Save the new user
                        save_users(users_db)  # Persist user data to file
                        st.session_state["username"] = username  # Automatically log in after registration
                        st.sidebar.success(f"Registration complete! Welcome, {username}.")
                        time.sleep(0.5)
                        st.rerun()  # Redirect to login and keep the session active
