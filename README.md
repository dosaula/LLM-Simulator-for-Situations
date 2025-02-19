# LLM-Simulator-for-Situations
Master's thesis project of students from the Master's in Big Data, Data Science &amp; Business Analytics at UCM.

## Team members
- Alejandro Blanco Rey.
- Amy Ramírez Jurado.
- Bruno Dosaula Ces.
- Germán David Cortés Hernández.
- Nicolás Esteban Spector.

## Description:

### AI-Powered Chatbot for Pharmaceutical Negotiation Scenarios

This project develops an AI-driven chatbot specialized in pharmaceutical negotiation scenarios. It features a modular and integrated architecture that combines advanced Natural Language Processing (NLP) techniques with effective management of user-provided information. 

The chatbot is designed to provide personalized responses, adapting to the context of the negotiation and adjusting to the different user profiles.

## Instructions to Run the Code
1. We recommend creating a specific virtual environment using Anaconda Prompt. To do so, run:
   ```bash
   conda create --name tfm python=3.11
2. Activate the enviroment:
   ```bash
   conda activate tfm
4. Then, navigate to the main `app.py` folder.
5. Install the necessary libraries using the following command:
   ```bash
   pip install -r requirements.txt
6. Navigate to the `config/local` folder and edit the `credentials.yml` file with a valid API KEY:
   ```bash
   OPENAI_API_KEY: 'xxx'
7. Run the `app.py` file:
   ```bash
   streamlit run app.py
