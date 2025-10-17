from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app)  # This will allow your React app to make requests to the backend

@app.route('/api/run-agent', methods=['POST'])
def run_agent():
    """
    This endpoint receives Python code from the frontend,
    writes it to the target file, runs the agent, and returns the result.
    """
    data = request.get_json()
    user_code = data.get('code')

    if not user_code:
        return jsonify({"error": "No code provided"}), 400

    try:
        # Define the path to the target file that the agent will fix
        target_file_path = os.path.join('target_project', 'app.py')

        # Write the user's code to the target file
        with open(target_file_path, 'w', encoding='utf-8') as f:
            f.write(user_code)

        # Run the agent script
        agent_path = os.path.join('generated_agent', 'agent.py')
        result = subprocess.run(
            ['python', agent_path],
            capture_output=True,
            text=True,
            check=False  # We don't want to raise an exception if the script fails
        )

        # The agent's output will be in stdout or stderr
        output = result.stdout or result.stderr

        return jsonify({"output": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
