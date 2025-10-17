"""
smol_dev.py

A simple developer script to generate an AI agent from a blueprint prompt.
This script reads a detailed prompt from 'main.prompt', sends it to the
Google Gemini model, and saves the generated Python code to
'generated_agent/agent.py'. It includes logic to sanitize the model's
output to ensure it is valid Python code.
"""

import os
import re
import sys
import google.generativeai as genai
from google.api_core import exceptions as google_api_exceptions
from dotenv import load_dotenv


def sanitize_python_code(response_text):
    """
    Cleans the raw text response from the AI model to ensure it's valid Python.
    - Strips leading/trailing whitespace.
    - Removes markdown code fences (```python ... ```).
    """
    # Regex to find content within ```python ... ```, covering multi-line responses.
    match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        # If a markdown block is found, extract its content.
        code = match.group(1)
    else:
        # Otherwise, assume the whole response is code but might have fences.
        # This handles cases where the AI might forget the opening 'python' tag.
        code = response_text.replace("```python", "").replace("```", "")

    return code.strip()


def main():
    """
    Main function to drive the agent generation process.
    """
    # 1. Load environment variables from a .env file
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # 2. Configure the Google Gemini API
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        print("Google Gemini API configured successfully.")
    except (ValueError, google_api_exceptions.GoogleAPICallError) as e:
        print(f"Error configuring Gemini API: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Read the master prompt from the 'main.prompt' file
    try:
        with open("main.prompt", "r", encoding="utf-8") as f:
            prompt_content = f.read()
        print("Successfully loaded 'main.prompt'.")
    except FileNotFoundError:
        print(
            "Error: 'main.prompt' not found in the current directory.", file=sys.stderr
        )
        sys.exit(1)

    # 4. Generate the agent code by calling the AI model
    print("Generating agent code from prompt... (This may take a moment)")
    try:
        response = model.generate_content(prompt_content)
        raw_code = response.text
    except google_api_exceptions.GoogleAPICallError as e:
        print(f"Error during API call to generate content: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Sanitize the response to ensure it's clean, executable Python code
    print("Sanitizing the generated code...")
    clean_code = sanitize_python_code(raw_code)

    if not clean_code:
        print(
            "Error: The generated response was empty after sanitization.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 6. Create the output directory and save the agent file
    try:
        output_dir = "generated_agent"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, "agent.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(clean_code)

        print(f"Agent generated successfully in '{file_path}'!")
    except IOError as e:
        print(f"Error writing the agent file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
