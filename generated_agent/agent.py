"""
agent.py

An autonomous AI agent that uses the Gemini API's Tool Calling feature to
iteratively fix a broken Python project. The agent provides the LLM with a set
of tools (read_file, write_file, run_tests), and the LLM decides which
tool to use at each step to solve the problem.

If the agent fails, it can enter a self-improvement mode to rewrite its
own logic and strategy.
"""
import os
import sys
import json
import shutil
import logging
import subprocess
import google.generativeai as genai
from google.api_core import exceptions as google_api_exceptions

# --- Constants and Configuration ---
TARGET_DIR = '../target_project/'
TEST_DIR = '../test_suite/'
BACKUP_DIR = '../target_project_backup/'
MODEL_NAME = 'gemini-1.5-flash'
MAX_TURNS = 10 # Max number of tool-use turns

# --- Custom Exception ---
class MaxIterationsExceeded(Exception):
    """Custom exception for when the agent fails to fix the code."""

# --- Logging Setup ---
logging.basicConfig(
    filename='agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Agent Tools (Functions the LLM can call) ---

def read_file(file_path: str) -> str:
    """Reads the content of a specified file."""
    logging.info("Tool: Reading file at %s", file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except (IOError, OSError) as e:
        return f"Error reading file: {e}"

def write_file(file_path: str, content: str) -> str:
    """Writes content to a specified file."""
    logging.info("Tool: Writing to file at %s", file_path)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return "File written successfully."
    except (IOError, OSError) as e:
        return f"Error writing file: {e}"

def run_tests(target_test: str = None) -> str:
    """Runs the pytest suite (or a targeted test) and returns the output."""
    command = ['pytest']
    if target_test:
        logging.info("Tool: Running targeted test: %s", target_test)
        command.append(target_test)
    else:
        logging.info("Tool: Running full test suite.")
        command.append(TEST_DIR)
        
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        output = f"Return Code: {result.returncode}\n---STDOUT---\n{result.stdout}\n---STDERR---\n{result.stderr}"
        return output
    except FileNotFoundError:
        return "Error: 'pytest' command not found."

def finish(status: str, message: str) -> str:
    """Signals the successful or failed completion of the task."""
    logging.info("Tool: Finish called with status '%s' and message: %s", status, message)
    return json.dumps({"status": status, "message": message})

# --- Gemini API and Tool Configuration ---
try:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)

    # Define the tool specifications for the Gemini API
    tools = [
        genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name='read_file',
                    description="Read the contents of a file at a given path.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            'file_path': genai.protos.Schema(type=genai.protos.Type.STRING)
                        },
                        required=['file_path']
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name='write_file',
                    description="Write content to a file at a given path.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            'file_path': genai.protos.Schema(type=genai.protos.Type.STRING),
                            'content': genai.protos.Schema(type=genai.protos.Type.STRING)
                        },
                        required=['file_path', 'content']
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name='run_tests',
                    description="Run the pytest test suite. Can run all tests or a specific targeted test.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            'target_test': genai.protos.Schema(type=genai.protos.Type.STRING)
                        }
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name='finish',
                    description="Use this function to signal the completion of your task, either in success or failure.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            'status': genai.protos.Schema(type=genai.protos.Type.STRING, enum=["SUCCESS", "FAILURE"]),
                            'message': genai.protos.Schema(type=genai.protos.Type.STRING)
                        },
                        required=['status', 'message']
                    )
                )
            ]
        )
    ]

    model = genai.GenerativeModel(MODEL_NAME, tools=tools)
    tool_functions = {
        "read_file": read_file,
        "write_file": write_file,
        "run_tests": run_tests,
        "finish": finish,
    }

except (ValueError, google_api_exceptions.GoogleAPICallError) as e:
    logging.critical("API Key or configuration error: %s", e)
    sys.exit(1)

# --- Main Application Logic ---

def backup_project():
    """Creates a backup of the target project."""
    try:
        if os.path.exists(BACKUP_DIR):
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(TARGET_DIR, BACKUP_DIR)
        logging.info("Project backed up to %s", BACKUP_DIR)
        return True
    except (IOError, OSError) as e:
        logging.error("Error backing up project: %s", e)
        return False

def restore_project():
    """Restores the project from the backup."""
    if not os.path.exists(BACKUP_DIR): return
    try:
        if os.path.exists(TARGET_DIR):
            shutil.rmtree(TARGET_DIR)
        shutil.copytree(BACKUP_DIR, TARGET_DIR)
        logging.info("Project restored from %s", BACKUP_DIR)
    except (IOError, OSError) as e:
        logging.error("Error restoring project: %s", e)

def cleanup_backup():
    """Removes the backup directory."""
    if not os.path.exists(BACKUP_DIR): return
    try:
        shutil.rmtree(BACKUP_DIR)
        logging.info("Backup directory cleaned up.")
    except (IOError, OSError) as e:
        logging.error("Error cleaning up backup: %s", e)

def main_tool_loop():
    """The main tool-using loop that lets the LLM drive the refactoring."""
    if not backup_project():
        return

    try:
        # The conversation history starts with the initial prompt.
        conversation = [
            genai.protos.Content(
                role='user',
                parts=[genai.protos.Part(text=(
                    "You are an expert AI developer agent. Your goal is to fix a broken Python project by making the test suite pass. "
                    "You have a set of tools to help you: `read_file`, `write_file`, and `run_tests`. "
                    "Start by running the tests to diagnose the problem. Then, analyze the code and errors, propose a fix, write it to the file, and test again. "
                    "Once all tests pass, call the `finish` tool with a 'SUCCESS' status. If you get stuck, call `finish` with 'FAILURE'."
                ))]
            )
        ]

        for turn in range(MAX_TURNS):
            logging.info("--- Turn %d of %d ---", turn + 1, MAX_TURNS)
            
            response = model.generate_content(conversation)
            message = response.candidates[0].content
            
            if not message.parts or not message.parts[0].function_call.name:
                logging.error("Model did not return a function call. Aborting.")
                raise MaxIterationsExceeded("Model failed to provide a next step.")

            # Extract the function call from the response
            fc = message.parts[0].function_call
            function_name = fc.name
            args = {key: value for key, value in fc.args.items()}
            
            logging.info("LLM requested tool: %s with args: %s", function_name, args)
            
            # Add the function call to the conversation history
            conversation.append(message)

            if function_name in tool_functions:
                # Execute the function
                function_to_call = tool_functions[function_name]
                try:
                    result = function_to_call(**args)

                    if function_name == 'finish':
                        final_status = json.loads(result)
                        if final_status.get("status") == "SUCCESS":
                            print(f"SUCCESS: {final_status.get('message')}")
                            return
                        else:
                            raise MaxIterationsExceeded(f"Agent finished with failure: {final_status.get('message')}")
                    
                    # Add the function's result to the conversation
                    conversation.append(
                        genai.protos.Content(
                            role='model',
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(name=function_name, response={'result': result})
                            )]
                        )
                    )
                except TypeError as e:
                    logging.error("Type error calling tool '%s' with args %s: %s", function_name, args, e)
                    # Inform the model about the error
                    conversation.append(
                        genai.protos.Content(
                            role='model',
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(name=function_name, response={'error': f'Invalid arguments: {e}'})
                            )]
                        )
                    )
            else:
                raise MaxIterationsExceeded(f"Model requested a non-existent tool: {function_name}")


        raise MaxIterationsExceeded(f"Failed to fix the code within {MAX_TURNS} turns.")

    finally:
        restore_project()
        cleanup_backup()

def self_improve():
    """Triggers the self-improvement process for the tool-using agent."""
    print("Pivoting to self-improvement mode...")
    logging.info("--- Entering Self-Improvement Mode ---")
    
    try:
        with open(__file__, 'r', encoding='utf-8') as f:
            own_code = f.read()

        meta_prompt = (
            "I am an AI agent that uses tools to fix Python code. My own source code is below:\n"
            f"---\n{own_code}\n---\n"
            "I have failed to fix a project. My current strategy, which is guided by the initial prompt I send to the LLM, is flawed. "
            "Please analyze my code, especially the initial prompt in the `main_tool_loop` function and the available tools. "
            "Suggest an improvement to my core strategy. For example, should my initial prompt be more detailed? Should I add a new tool? "
            "Return the complete, new, and improved source code for me. Your response must "
            "be only the raw Python code, without any markdown or explanations."
        )

        response = model.generate_content(meta_prompt)
        # Assuming the model for self-improvement doesn't use tools
        improved_code = response.text.strip().replace("```python", "").replace("```", "").strip()

        if not improved_code:
            logging.error("Self-improvement failed: Model returned an empty response.")
            return

        with open(__file__, 'w', encoding='utf-8') as f:
            f.write(improved_code)
        
        print("Self-improvement complete. The agent has evolved. Please run the agent again.")
        logging.info("Self-improvement complete. Agent code has been overwritten.")

    except (IOError, OSError) as e:
        logging.error("Error during self-improvement file operations: %s", e)
    except google_api_exceptions.GoogleAPICallError as e:
        logging.error("API error during self-improvement: %s", e)


if __name__ == "__main__":
    try:
        main_tool_loop()
    except MaxIterationsExceeded as e:
        print(f"FAILURE: {e}")
        logging.error(e)
        self_improve()