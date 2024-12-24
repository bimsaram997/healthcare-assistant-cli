# healthcare-assistant-cli

# Healthcare Diagnostic Assistant CLI

This is a Python-based Healthcare Diagnostic Assistant that uses machine learning and fuzzy string matching to interactively gather user symptoms, retrieve relevant context, and provide possible diagnoses based on structured data and an AI language model.

## Features

- **Interactive Symptom Input**: Users can describe their primary symptom, and the system will suggest relevant context.
- **Fuzzy Matching**: Matches user input to similar symptoms in a structured dataset, even if phrased differently.
- **AI-Powered Questions**: Dynamically generates follow-up questions to narrow down potential diagnoses.
- **Context-Aware Diagnoses**: Combines user responses and symptom data to provide possible diagnoses.
- **Friendly CLI Interface**: Easy-to-use command-line interaction.

## Prerequisites

- Python 3.7 or higher
- Required Python libraries:
  - `pandas`
  - `dotenv`
  - `fuzzywuzzy`
  - `openai`
- An OpenAI-compatible server running locally (or replace with a valid base URL).
- `symptoms_data.csv` file containing structured symptom-related data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bimsaram997/healthcare-assistant-cli.git
   cd healthcare-assistant-cli
2. Install the dependencies:
  pip install -r requirements.txt
3. Set up your .env file with your OpenAI API key:
  OPENAI_API_KEY=your_openai_api_key_here
4. Pull from Ollama
  ollama pull llama2

# Usage
1. Run the script:
  python health.py
2. Follow the prompts:
  -Enter your primary symptom.
  -Respond to follow-up questions.
  -Receive a potential diagnosis.

