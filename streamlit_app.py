import streamlit as st
from langchain_openai.chat_models import ChatOpenAI

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
from fuzzywuzzy import process
# Load environment variables
load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key=api_key,
)

# Load and preprocess symptoms data
symptoms_df = pd.read_csv('symptoms_data.csv')

def get_initial_symptom():
    return input("Please describe your primary symptom: ")

def retrieve_relevant_data(symptom):
    # Preprocess user symptom for better matching
    user_symptom = symptom.lower().strip()
    
    # List of symptoms from the CSV for comparison
    symptom_list = symptoms_df['symptom'].dropna().str.lower().tolist()
    
    # Find the best matches using fuzzy matching
    matches = process.extract(user_symptom, symptom_list, limit=5)
    
    # Filter matches with a confidence score threshold (e.g., > 60%)
    relevant_symptoms = [match for match, score in matches if score > 60]
    
    if not relevant_symptoms:
        return "No relevant data found."
    
    # Retrieve rows in the dataframe matching the relevant symptoms
    relevant_data = symptoms_df[symptoms_df['symptom'].str.lower().isin(relevant_symptoms)]
    
    # Summarize relevant context from the dataframe
    return relevant_data.to_dict(orient="records")

def query_llm(prompt):
    response = client.chat.completions.create(
        model="llama2:7b",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant with access to structured data."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_follow_up_questions(symptom, context):
    if not context:  # Check if context is empty (None, empty list, empty dict, etc.)
        print("No context provided. Generating questions based on the symptom alone.")
        prompt = f"Patient reports: {symptom}. What follow-up questions would you ask to narrow down the diagnosis?"
    else:
        prompt = f"Patient reports: {symptom}. Using the following context: {context}, what follow-up questions would you ask to narrow down the diagnosis?"
    questions = query_llm(prompt)
    follow_up_questions = extract_follow_up_questions(questions)
    return follow_up_questions

def extract_follow_up_questions(text):
    questions = []
    for line in text.splitlines():
        if line.strip().startswith(tuple(str(i) + '.' for i in range(1, 11))):
            questions.append(line.strip())
    return questions

def collect_user_responses(questions):
    print("The system will now ask some follow-up questions.")
    responses = {}
    for i, question in enumerate(questions, start=1):
        print(f"Agent: {question}")
        response = input(f"User ({i}/{len(questions)}): ").strip().lower()
        responses[question] = response
    return responses

def generate_diagnosis(symptom, responses, context):
    follow_up_info = ' '.join([f"{q}: {a}." for q, a in responses.items()])
    prompt = f"Patient reports: {symptom}. {follow_up_info} Using the following context: {context}, what is the most likely diagnosis?"
    diagnosis = query_llm(prompt)
    return diagnosis

def provide_summary(diagnosis):
    print(f"Based on the information provided, you may have: {diagnosis}")
    print("It's recommended to consult a healthcare professional for an accurate diagnosis.")



import streamlit as st

st.title("Healthcare Diagnostic Assistant")

# Initialize session state for button disabled flag if not already set
if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = False
    
# Initialize the disabled state if it doesn't exist
if 'submit_disabled' not in st.session_state:
    st.session_state.submit_disabled = False

# Define the action to disable the button after submission
def disable_button():
    st.session_state.button_disabled = True

def enable_button():
    st.session_state.button_disabled = False
    
# Initialize session state for button disabled flag if not already set
if 'submit_disabled' not in st.session_state:
    st.session_state.submit_disabled = False

# Define the action to disable the button after submission
def disable_submit_button():
    st.session_state.submit_disabled = True

def enable_submit_button():
    st.session_state.submit_disabled = False

def reset_symptom():
    # Reset the symptom-related state to allow new entry
    st.session_state.button_disabled = False
    st.session_state.submit_disabled = False
    st.session_state.pop("symptom", None)
    st.session_state.pop("context", None)
    st.session_state.pop("follow_up_questions", None)
    st.session_state.pop("responses", None)
    st.session_state.pop("diagnosis", None)
    st.rerun()  # Force a rerun to reset the form immediately

# Symptom input form
with st.form("symptom_form"):
    symptom = st.text_input("Please describe your primary symptom:", value=st.session_state.get("symptom", ""))
    # Button is disabled based on session state
    
    symptom_submitted = st.form_submit_button(
        label="Submit", 
        disabled=st.session_state.button_disabled if symptom else False, 
        on_click=disable_button  # Disable the button once clicked
    )

# Store state to avoid resetting data

if symptom_submitted:
    if not symptom:
        st.error("Please enter a symptom.") 
    else:
        st.session_state["symptom"] = symptom
        with st.spinner('Follow-up questions processing...'):
            st.session_state["context"] = retrieve_relevant_data(symptom)  # Replace with your actual function
            st.session_state["follow_up_questions"] = generate_follow_up_questions(symptom, st.session_state["context"])  # Replace with your actual function

if "symptom" in st.session_state:
    # Retrieve saved state
    symptom = st.session_state["symptom"]
    context = st.session_state["context"]
    follow_up_questions = st.session_state["follow_up_questions"]

    if follow_up_questions:
        st.write("Follow-up questions:")

        # Form to collect responses to follow-up questions
        with st.form("follow_up_form"):
            responses = {}
            for question in follow_up_questions:
                responses[question] = st.text_input(question)
            follow_up_submitted = st.form_submit_button(
                label="Submit Responses", 
                disabled=st.session_state.submit_disabled,
                on_click=disable_submit_button  # Disable the button once clicked
            )
         

        if follow_up_submitted:
                # Save responses in session_state
                st.session_state["responses"] = responses

                with st.spinner('Generating diagnosis...'): 
                    # Generate diagnosis (Replace with your actual function)
                    diagnosis = generate_diagnosis(
                        symptom, 
                        st.session_state["responses"], 
                        context 
                    )
                st.session_state["diagnosis"] = diagnosis

if "diagnosis" in st.session_state:
    st.subheader("Diagnosis")
    st.write(st.session_state["diagnosis"])

    # Button to add a new symptom
    if st.button("Add New Symptom"):
        reset_symptom()  # Reset form to allow new symptom input
